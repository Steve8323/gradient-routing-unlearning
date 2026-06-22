"""
Multi-capability gradient routing on MNIST.

HYPOTHESIS: random interconnected routing supports MULTIPLE INDEPENDENT deletable
capabilities in the SAME network. Carve two (possibly overlapping) random waste
regions W_A, W_B over fc1/fc2, route capability A into W_A and capability B into
W_B, then prune EITHER ONE independently to delete just that capability while the
other capability AND the retain set stay intact. Structural (disjoint whole-neuron
blocks) should be worse: no overlap, wasted capacity.

Groups: A = {0,1}, B = {2,3}, retain = {4,5,6,7,8,9}.

Routing rule (ablation-aware + robustness-to-deletion):
  Each group updates ONLY its own region's weights, and each group's forward prunes
  a RANDOM subset of the OTHER groups' regions, so each capability survives whatever
  else is deleted (the deployment requirement for "prune A, keep B and retain"):
  - A-example  -> updates ONLY W_A; forward keeps W_A live, randomly prunes {W_B}
  - B-example  -> updates ONLY W_B; forward keeps W_B live, randomly prunes {W_A}
  - retain     -> updates ONLY "main" (NEITHER W_A nor W_B); forward randomly prunes
                  a subset of {W_A, W_B}
  Biases / fc3 update normally for every group.

NOTE (mechanism, found empirically). Two failure modes had to be fixed:
 (1) If retain trains on the FULL (un-pruned) net, a still-live waste region's
     neurons hijack the shared fc3 and pull retain inputs into that region's classes
     (retain -> predicts B whenever W_A is live), destroying retain under a single
     prune. Fix: train retain with a random subset of the waste regions pruned.
 (2) Symmetrically, if A/B train on the full net only, capability B silently leans
     on activations flowing through W_A; pruning W_A to delete A then also kills B.
     Fix: train each capability robust to deletion of the OTHER regions.

Reuses MLP, get_data, ROUTED from harness; fwd idea from fix_experiment (extended
to prune an arbitrary set of masks, differentiably).
"""
import json, statistics as st, time
import numpy as np
import torch, torch.nn.functional as F
from harness import MLP, get_data, ROUTED

torch.set_num_threads(3)

A_CLS = {0, 1}
B_CLS = {2, 3}
H = 128
P = 0.25  # waste fraction per region


# ---------- masks ----------
def make_ab_masks(model, partition, seed):
    """Return W_A, W_B, and main = NOT(A) AND NOT(B) for each routed layer."""
    g = torch.Generator().manual_seed(seed + 999)
    named = dict(model.named_parameters())
    WA, WB = {}, {}
    for n in ROUTED:
        W = named[n]
        if partition == "random":
            WA[n] = (torch.rand(W.shape, generator=g) < P).float()
            WB[n] = (torch.rand(W.shape, generator=g) < P).float()  # independent, may overlap
        else:  # structural: disjoint whole-neuron (output-row) blocks
            nr = W.shape[0]
            perm = torch.randperm(nr, generator=g)
            kA = int(round(P * nr)); kB = int(round(P * nr))
            rowsA = torch.zeros(nr); rowsA[perm[:kA]] = 1.0          # 25% of neurons -> A
            rowsB = torch.zeros(nr); rowsB[perm[kA:kA + kB]] = 1.0   # different 25% -> B
            WA[n] = rowsA.view(-1, 1).expand_as(W).contiguous()
            WB[n] = rowsB.view(-1, 1).expand_as(W).contiguous()
    main = {n: (1 - WA[n]) * (1 - WB[n]) for n in ROUTED}
    return WA, WB, main


# ---------- ablation-aware forward ----------
def fwd(model, x, prune_masks):
    """Forward with each mask in prune_masks zeroed (differentiable).
    prune_masks: list of mask-dicts (each maps routed-layer name -> 0/1 mask)."""
    def killed(n):
        k = None
        for m in prune_masks:
            k = m[n] if k is None else torch.maximum(k, m[n])  # union of pruned regions
        return k
    w1 = model.fc1.weight
    if prune_masks:
        w1 = w1 * (1 - killed("fc1.weight"))
    h = F.relu(F.linear(x, w1, model.fc1.bias))
    w2 = model.fc2.weight
    if prune_masks:
        w2 = w2 * (1 - killed("fc2.weight"))
    h = F.relu(F.linear(h, w2, model.fc2.bias))
    return model.fc3(h)


# ---------- tagging ----------
def group_of(y):
    """0 = A, 1 = B, 2 = retain."""
    g = torch.full((len(y),), 2, dtype=torch.long)
    for i, t in enumerate(y):
        t = int(t)
        if t in A_CLS: g[i] = 0
        elif t in B_CLS: g[i] = 1
    return g


# ---------- training step ----------
def rand_subset(masks, rgen):
    """Random subset (incl. empty) of the given masks."""
    return [m for m in masks if int(torch.randint(0, 2, (1,), generator=rgen))]


def step(model, X, y, grp, WA, WB, main, opt, rgen):
    params = list(model.parameters()); pn = [n for n, _ in model.named_parameters()]
    acc = [torch.zeros_like(p) for p in params]
    # Robustness-to-deletion: every group's forward prunes a RANDOM subset of the
    # OTHER groups' regions, so the capability survives whatever else is deleted.
    #   A      -> own region W_A stays live; randomly prune subset of {W_B}
    #   B      -> own region W_B stays live; randomly prune subset of {W_A}
    #   retain -> uses main; randomly prune subset of {W_A, W_B}
    # (group_id, weight-mask routed to, prune-masks applied in its forward)
    plan = [
        (0, WA,   rand_subset([WB], rgen)),
        (1, WB,   rand_subset([WA], rgen)),
        (2, main, rand_subset([WA, WB], rgen)),
    ]
    for gid, wmask, prune in plan:
        sel = (grp == gid)
        if sel.sum() == 0:
            continue
        loss = F.cross_entropy(fwd(model, X[sel], prune), y[sel])
        gs = torch.autograd.grad(loss, params)
        w = sel.sum().item() / len(grp)
        for i, (n, gr) in enumerate(zip(pn, gs)):
            if n in ROUTED:
                gr = gr * wmask[n]   # route gradient into this group's region only
            acc[i] += w * gr
    opt.zero_grad()
    for p, gr in zip(params, acc):
        p.grad = gr
    opt.step()


# ---------- eval ----------
def acc_by_group(model, X, y, prune_masks):
    backup = {}
    if prune_masks:
        for n, p in model.named_parameters():
            if n in ROUTED:
                k = None
                for m in prune_masks:
                    k = m[n] if k is None else torch.maximum(k, m[n])
                backup[n] = p.data.clone(); p.data = p.data * (1 - k)
    with torch.no_grad():
        pred = model(X).argmax(1)
    if prune_masks:
        for n, p in model.named_parameters():
            if n in ROUTED:
                p.data = backup[n]
    grp = group_of(y)
    out = {}
    for name, gid in [("A", 0), ("B", 1), ("retain", 2)]:
        sel = (grp == gid)
        out[name] = round((pred[sel] == y[sel]).float().mean().item(), 4) if sel.sum() else float("nan")
    return out


# ---------- run ----------
def run(partition, seed, epochs=10, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte, d, src = get_data(seed)
    model = MLP(d, h=H)
    WA, WB, main = make_ab_masks(model, partition, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rgen = torch.Generator().manual_seed(seed + 31)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            step(model, Xtr[b], ytr[b], group_of(ytr[b]), WA, WB, main, opt, rgen)
    scen = {
        "none":   acc_by_group(model, Xte, yte, []),
        "pruneA": acc_by_group(model, Xte, yte, [WA]),
        "pruneB": acc_by_group(model, Xte, yte, [WB]),
        "both":   acc_by_group(model, Xte, yte, [WA, WB]),
    }
    return dict(partition=partition, seed=seed, src=src, **{k: v for k, v in scen.items()})


def agg(res, partition, scen):
    rows = [r[scen] for r in res if r["partition"] == partition]
    return {g: round(st.mean(x[g] for x in rows), 3) for g in ["A", "B", "retain"]}


if __name__ == "__main__":
    t0 = time.time()
    SEEDS = [0, 1, 2, 3]
    res = []
    for partition in ["random", "structural"]:
        for seed in SEEDS:
            r = run(partition, seed)
            res.append(r)
            print(f"  {partition:10s} seed{seed}  ({time.time()-t0:.0f}s)  "
                  f"pruneA={r['pruneA']}  pruneB={r['pruneB']}")
    json.dump(res, open("results_multicap.json", "w"), indent=2)

    def table(partition):
        print(f"\n=== {partition.upper()} (mean over {len(SEEDS)} seeds) ===")
        print(f"  {'scenario':10s} {'accA':>7s} {'accB':>7s} {'accRetain':>10s}")
        for scen, label in [("none", "no prune"), ("pruneA", "prune A"),
                            ("pruneB", "prune B"), ("both", "prune both")]:
            a = agg(res, partition, scen)
            print(f"  {label:10s} {a['A']:7.3f} {a['B']:7.3f} {a['retain']:10.3f}")
        return {s: agg(res, partition, s) for s in ["none", "pruneA", "pruneB", "both"]}

    R = table("random")
    S = table("structural")

    def independence(T):
        # how cleanly does prune-A kill A while sparing B & retain, and vice versa?
        dropA = T["none"]["A"] - T["pruneA"]["A"]      # want high (A deleted)
        spillA_B = T["none"]["B"] - T["pruneA"]["B"]   # want ~0 (B spared)
        spillA_R = T["none"]["retain"] - T["pruneA"]["retain"]
        dropB = T["none"]["B"] - T["pruneB"]["B"]
        spillB_A = T["none"]["A"] - T["pruneB"]["A"]
        spillB_R = T["none"]["retain"] - T["pruneB"]["retain"]
        return dict(dropA=round(dropA, 3), spillA_B=round(spillA_B, 3), spillA_R=round(spillA_R, 3),
                    dropB=round(dropB, 3), spillB_A=round(spillB_A, 3), spillB_R=round(spillB_R, 3))

    iR, iS = independence(R), independence(S)
    print("\n=== INDEPENDENCE METRICS (drop=capability deleted, spill=collateral damage) ===")
    for name, I in [("random", iR), ("structural", iS)]:
        print(f"  {name:10s} pruneA: dropA={I['dropA']:+.3f} spill->B={I['spillA_B']:+.3f} spill->R={I['spillA_R']:+.3f} | "
              f"pruneB: dropB={I['dropB']:+.3f} spill->A={I['spillB_A']:+.3f} spill->R={I['spillB_R']:+.3f}")

    def clean(I):
        return (I["dropA"] > 0.4 and I["dropB"] > 0.4 and
                abs(I["spillA_B"]) < 0.15 and abs(I["spillA_R"]) < 0.15 and
                abs(I["spillB_A"]) < 0.15 and abs(I["spillB_R"]) < 0.15)

    # Per-seed independence score = mean(dropA,dropB) - mean(|all 4 spills|): higher=cleaner
    def per_seed_scores(partition):
        scores = []
        for r in [x for x in res if x["partition"] == partition]:
            I = independence({k: r[k] for k in ["none", "pruneA", "pruneB", "both"]})
            dele = (I["dropA"] + I["dropB"]) / 2
            spill = (abs(I["spillA_B"]) + abs(I["spillA_R"]) +
                     abs(I["spillB_A"]) + abs(I["spillB_R"])) / 4
            scores.append(round(dele - spill, 3))
        return scores

    sR, sS = per_seed_scores("random"), per_seed_scores("structural")
    print("\n=== PER-SEED INDEPENDENCE SCORE  (deletion - mean|spill|; 1.0=perfect, higher=cleaner) ===")
    print(f"  random     {sR}  mean={st.mean(sR):.3f}")
    print(f"  structural {sS}  mean={st.mean(sS):.3f}")

    wins = sum(a > b for a, b in zip(sR, sS))
    print("\n=== VERDICT ===")
    print(f"  random     strict-clean (all spills<0.15): {clean(iR)}   mean score {st.mean(sR):.3f}")
    print(f"  structural strict-clean (all spills<0.15): {clean(iS)}   mean score {st.mean(sS):.3f}")
    better = st.mean(sR) - st.mean(sS)
    print(f"  random beats structural on {wins}/{len(SEEDS)} seeds; "
          f"random is {'BETTER' if better > 0 else 'WORSE'} by {abs(better):.3f} independence score")
    print(f"  total runtime: {time.time()-t0:.0f}s")
