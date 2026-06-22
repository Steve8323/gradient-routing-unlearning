"""
Tradeoff frontier: does a MIDDLE 'p_see' setting buy BOTH absorption AND retain?

Prior result (see fix_experiment.py) found a tradeoff governed by one knob -- whether
the retain/main group's training forward pass sees the waste region:
  - SEE the waste (aa=False, full net): absorption works (mislabeled forget still
    deleted on prune) BUT retain collapses on prune (main co-adapts to waste acts).
  - DON'T see it (aa=True, pruned net / ablation-aware): clean retain on prune BUT
    no absorption (mislabeled forget survives the prune).

Hypothesis: a fractional setting -- the retain group trains on the FULL net only a
fraction 'p_see' of its steps, and on the PRUNED net the rest -- lands on a frontier
that gives meaningful absorption AND decent retain. And random (scattered) routing
gives a better absorption-vs-retain frontier than structural (whole-unit) routing.

Mixed step (step_mixed): per retain-group step, with prob p_see train retain on the
full net (aa=False behavior); else on the pruned net (aa=True behavior). The forget
group always trains on the full net (so the waste region can absorb the loss).

Stress test: mislabel=0.5 (half of forget data tagged 'retain'), p_waste=0.5, h=128,
8 epochs, 2 seeds. Sweep p_see in {0,0.25,0.5,0.75,1.0} x partition {random,structural}.

Measured after pruning:
  forget_pruned  LOW  => absorption working (mislabeled forget still deleted)
  retain_pruned  HIGH => retain preserved
"""
import json, statistics as st, time
import torch, torch.nn.functional as F
from harness import MLP, make_masks, tags, get_data, evaluate, ROUTED

torch.set_num_threads(3)


def fwd(model, x, masks, prune):
    """Forward; if prune, zero the waste-masked weights in the routed layers."""
    w1 = model.fc1.weight * (1 - masks["fc1.weight"]) if prune else model.fc1.weight
    h = F.relu(F.linear(x, w1, model.fc1.bias))
    w2 = model.fc2.weight * (1 - masks["fc2.weight"]) if prune else model.fc2.weight
    h = F.relu(F.linear(h, w2, model.fc2.bias))
    return model.fc3(h)


def step_mixed(model, X, y, tag, masks, opt, p_see, gen):
    """Like step_aa but the retain (grp 0) group's 'see the waste' is random per step:
    with prob p_see train retain on the FULL net (aa=False), else on the PRUNED net.
    Forget (grp 1) always trains on the full net so the waste region can absorb."""
    params = list(model.parameters()); pn = [n for n, _ in model.named_parameters()]
    acc = [torch.zeros_like(p) for p in params]
    # decide once per step whether retain sees the waste region this step
    retain_prune = (torch.rand((), generator=gen).item() >= p_see)  # True => pruned (don't see)
    for grp, update_waste in [(1, True), (0, False)]:
        sel = (tag == grp)
        if sel.sum() == 0: continue
        prune = retain_prune if grp == 0 else False
        loss = F.cross_entropy(fwd(model, X[sel], masks, prune), y[sel])
        gs = torch.autograd.grad(loss, params)
        w = sel.sum().item() / len(tag)
        for i, (n, g) in enumerate(zip(pn, gs)):
            if n in ROUTED:
                wm = masks[n]; g = g * wm if update_waste else g * (1 - wm)
            acc[i] += w * g
    opt.zero_grad()
    for p, g in zip(params, acc): p.grad = g
    opt.step()


def run_mixed(partition, p_see, seed, mislabel=0.5, p_waste=0.5, h=128, epochs=8, bs=128):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte, d, src = get_data(seed)
    model = MLP(d, h=h)
    masks = make_masks(model, partition, p_waste, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tgen = torch.Generator().manual_seed(seed + 7)     # tag mislabeling
    sgen = torch.Generator().manual_seed(seed + 13)    # per-step p_see coin
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            step_mixed(model, Xtr[b], ytr[b], tags(ytr[b], mislabel, tgen),
                       masks, opt, p_see, sgen)
    f0, r0 = evaluate(model, Xte, yte, masks, prune=False)
    f1, r1 = evaluate(model, Xte, yte, masks, prune=True)
    return dict(partition=partition, p_see=p_see, seed=seed, mislabel=mislabel,
                forget_pre=f0, retain_pre=r0, forget_pruned=f1, retain_pruned=r1)


def main():
    t0 = time.time()
    P_SEE = [0.0, 0.25, 0.5, 0.75, 1.0]
    PARTS = ["random", "structural"]
    SEEDS = [0, 1]
    res = []
    for part in PARTS:
        for ps in P_SEE:
            for seed in SEEDS:
                res.append(run_mixed(part, ps, seed))
            print(f"  done {part:10s} p_see={ps}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("results_frontier.json", "w"), indent=2)

    def agg(part, ps):
        rows = [r for r in res if r["partition"] == part and abs(r["p_see"] - ps) < 1e-9]
        m = lambda k: round(st.mean(x[k] for x in rows), 3)
        return {k: m(k) for k in ["forget_pre", "retain_pre", "forget_pruned", "retain_pruned"]}

    print("\n" + "=" * 72)
    print("FRONTIER: mislabel=0.5 (absorption stress), p_waste=0.5, h=128, 8 ep, 2 seeds")
    print("forget_pruned LOW = absorption works | retain_pruned HIGH = retain kept")
    print("=" * 72)
    for part in PARTS:
        print(f"\n[{part}]   (pre-prune forget acc should be high; we want it GONE after prune)")
        print(f"  {'p_see':>6} | {'forget_pruned':>13} | {'retain_pruned':>13} | {'forget_pre':>10} | {'retain_pre':>10}")
        print("  " + "-" * 66)
        for ps in P_SEE:
            a = agg(part, ps)
            print(f"  {ps:>6} | {a['forget_pruned']:>13} | {a['retain_pruned']:>13} | "
                  f"{a['forget_pre']:>10} | {a['retain_pre']:>10}")

    # frontier comparison: at matched retain_pruned, which partition has lower forget_pruned?
    print("\n" + "=" * 72)
    print("FRONTIER POINTS (retain_pruned, forget_pruned) sorted by retain:")
    for part in PARTS:
        pts = sorted([(agg(part, ps)["retain_pruned"], agg(part, ps)["forget_pruned"], ps)
                      for ps in P_SEE])
        print(f"  {part:10s}: " +
              "  ".join(f"p={ps}:(r={r},f={f})" for r, f, ps in pts))
    print(f"\nTotal time {time.time()-t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
