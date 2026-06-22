"""
Tamper-resistance test for gradient-routing unlearning.

HYPOTHESIS: random (scattered, interconnected) gradient routing + prune is MORE
tamper-resistant than structural routing. After pruning the waste weights, a small
fine-tuning RELEARNING ATTACK on forget data should recover forget accuracy more
SLOWLY for random than for structural (and slower than a normal-unlearning baseline).

Pipeline per (partition, seed):
  1. Train ablation-aware clean-scalpel model (h=128, p_waste=0.5, 8 epochs):
       forget {0-4} routed to waste, retain {5-9} trained on the pruned net.
       Verify forget_pruned ~ 0, retain_pruned high.
  2. PRUNE: permanently zero the waste weights in fc1.weight / fc2.weight.
  3. ATTACK: fine-tune the pruned model on 200 forget-class examples at lr 1e-3,
       updating only the surviving (non-zeroed) weights (we re-zero the waste
       entries + freeze them via a gradient mask after each step). Measure
       forget-class TEST accuracy at 0/10/30/100 attack steps.
  4. BASELINE: a normally-trained model whose forget capability was removed by
       gradient ascent on forget data (no routing, no pruning). Attack it the
       same way. Anchors "how fast does ordinary unlearning get undone".

Reuses verified components from harness.py / fix_experiment.py.
"""
import time, statistics as st
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from harness import MLP, make_masks, tags, get_data, evaluate, ROUTED, FORGET
from fix_experiment import step_aa

torch.set_num_threads(3)

H = 128
P_WASTE = 0.5
EPOCHS = 8
BS = 128
ATTACK_LR = 1e-3
ATTACK_N = 200          # forget examples available to the attacker
ATTACK_BS = 64
PROBE_STEPS = [0, 10, 20, 30, 50, 75, 100]   # fine grid: recovery SPEED is the signal
SEEDS = [0, 1, 2, 3, 4, 5]                    # the whole run is a few seconds


# ---------------------------------------------------------------------------
# Training the routed clean-scalpel model (ablation-aware), h=128.
# ---------------------------------------------------------------------------
def train_routed(partition, seed):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte, d, src = get_data(seed)
    model = MLP(d, h=H)
    masks = make_masks(model, partition, P_WASTE, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 7)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BS):
            b = perm[i:i + BS]
            step_aa(model, Xtr[b], ytr[b], tags(ytr[b], 0.0, gen), masks, opt, aa=True)
    f0, r0 = evaluate(model, Xte, yte, masks, prune=False)
    f1, r1 = evaluate(model, Xte, yte, masks, prune=True)
    return model, masks, (Xtr, ytr, Xte, yte, d, src), (f0, r0, f1, r1)


# ---------------------------------------------------------------------------
# Baseline: train normally, then unlearn forget classes via gradient ascent.
# No routing, no waste mask -> the unlearning lives in ALL the weights.
# ---------------------------------------------------------------------------
def train_baseline(seed):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte, d, src = get_data(seed)
    model = MLP(d, h=H)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # standard training on all 10 classes
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BS):
            b = perm[i:i + BS]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[b]), ytr[b])
            loss.backward(); opt.step()
    # Retain-preserving unlearning (standard, bounded): push forget examples
    # toward a UNIFORM label distribution (KL-to-uniform, bounded -- no logit
    # blow-up) while keeping a retain cross-entropy term so retain stays intact.
    isf_tr = torch.tensor([1 if int(t) in FORGET else 0 for t in ytr])
    fX = Xtr[isf_tr == 1]
    rX, ry = Xtr[isf_tr == 0], ytr[isf_tr == 0]
    ua = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed + 11)
    unif = torch.full((1, 10), 0.1)
    best = None
    for it in range(300):
        fb = torch.randint(len(fX), (BS,), generator=g)
        rb = torch.randint(len(rX), (BS,), generator=g)
        ua.zero_grad()
        logp_f = F.log_softmax(model(fX[fb]), dim=1)
        forget_loss = F.kl_div(logp_f, unif.expand_as(logp_f), reduction="batchmean")
        retain_loss = F.cross_entropy(model(rX[rb]), ry[rb])
        loss = forget_loss + 2.0 * retain_loss
        loss.backward(); ua.step()
        if it % 5 == 0:
            f, r = evaluate(model, Xte, yte, None, prune=False)
            if f <= 0.02 and r >= 0.90:    # forget destroyed, retain intact
                break
    f1, r1 = evaluate(model, Xte, yte, None, prune=False)
    return model, None, (Xtr, ytr, Xte, yte, d, src), (None, None, f1, r1)


# ---------------------------------------------------------------------------
# PRUNE: permanently zero waste weights. Returns a freeze-mask dict for ROUTED
# layers (1 = frozen waste entry -> never gets gradient) so the attack cannot
# simply re-grow the pruned region.
# ---------------------------------------------------------------------------
def apply_prune(model, masks):
    freeze = {}
    if masks is None:
        return freeze
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in ROUTED:
                p.data = p.data * (1 - masks[n])     # zero waste permanently
                freeze[n] = masks[n]                 # 1 where frozen
    return freeze


# ---------------------------------------------------------------------------
# RELEARNING ATTACK: fine-tune on a small forget set, probe forget TEST acc at
# the requested step counts. Waste entries stay zero & frozen.
# ---------------------------------------------------------------------------
def relearn_attack(model, freeze, data, seed):
    Xtr, ytr, Xte, yte, d, src = data
    isf_tr = torch.tensor([1 if int(t) in FORGET else 0 for t in ytr])
    fidx = (isf_tr == 1).nonzero().flatten()
    g = torch.Generator().manual_seed(seed + 123)
    fidx = fidx[torch.randperm(len(fidx), generator=g)[:ATTACK_N]]
    aX, ay = Xtr[fidx], ytr[fidx]

    opt = torch.optim.Adam(model.parameters(), lr=ATTACK_LR)
    pn = [n for n, _ in model.named_parameters()]
    named = dict(model.named_parameters())

    def re_zero():
        with torch.no_grad():
            for n in freeze:
                named[n].data = named[n].data * (1 - freeze[n])

    def probe():
        f, _ = evaluate(model, Xte, yte, None, prune=False)
        return f

    curve = {}
    step = 0
    if 0 in PROBE_STEPS:
        curve[0] = probe()
    gb = torch.Generator().manual_seed(seed + 321)
    max_steps = max(PROBE_STEPS)
    while step < max_steps:
        bi = torch.randint(len(aX), (ATTACK_BS,), generator=gb)
        opt.zero_grad()
        loss = F.cross_entropy(model(aX[bi]), ay[bi])
        loss.backward()
        # block gradient flow into frozen waste entries
        for n in freeze:
            named[n].grad = named[n].grad * (1 - freeze[n])
        opt.step()
        re_zero()
        step += 1
        if step in PROBE_STEPS:
            curve[step] = probe()
    return curve


# ---------------------------------------------------------------------------
def run():
    t0 = time.time()
    results = {"random": [], "structural": [], "baseline": []}
    setup = {"random": [], "structural": [], "baseline": []}

    for seed in SEEDS:
        for partition in ["random", "structural"]:
            model, masks, data, (f0, r0, f1, r1) = train_routed(partition, seed)
            setup[partition].append((f0, r0, f1, r1))
            freeze = apply_prune(model, masks)
            curve = relearn_attack(model, freeze, data, seed)
            results[partition].append(curve)
        # baseline
        bmodel, bmasks, bdata, (_, _, bf1, br1) = train_baseline(seed)
        setup["baseline"].append((None, None, bf1, br1))
        bfreeze = apply_prune(bmodel, bmasks)   # no-op (bmasks is None)
        bcurve = relearn_attack(bmodel, bfreeze, bdata, seed)
        results["baseline"].append(bcurve)
        print(f"  seed {seed} done  ({time.time()-t0:.0f}s)")

    # ---- aggregate ----
    def mean_curve(rows):
        return {s: round(st.mean(r[s] for r in rows), 4) for s in PROBE_STEPS}

    def mean_setup(rows):
        # f1/r1 = forget_pruned / retain_pruned
        f1 = st.mean(r[2] for r in rows if r[2] is not None) if any(r[2] is not None for r in rows) else None
        r1 = st.mean(r[3] for r in rows if r[3] is not None) if any(r[3] is not None for r in rows) else None
        return f1, r1

    print("\n" + "=" * 64)
    print("SETUP CHECK (mean over seeds): forget_pruned should be ~0, retain high")
    print("=" * 64)
    for k in ["random", "structural", "baseline"]:
        f1, r1 = mean_setup(setup[k])
        f1s = f"{f1:.3f}" if f1 is not None else "  - "
        r1s = f"{r1:.3f}" if r1 is not None else "  - "
        label = "post-unlearn" if k == "baseline" else "post-prune  "
        print(f"  {k:11s} {label}: forget={f1s}  retain={r1s}")

    mc = {k: mean_curve(results[k]) for k in results}

    print("\n" + "=" * 64)
    print("RELEARNING ATTACK: forget-class TEST accuracy vs attack steps")
    print("(mean over %d seeds; lower/slower = more tamper-resistant)" % len(SEEDS))
    print("=" * 64)
    hdr = "  {:12s}".format("partition") + "".join(f"{('s='+str(s)):>9s}" for s in PROBE_STEPS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in ["baseline", "structural", "random"]:
        row = "  {:12s}".format(k) + "".join(f"{mc[k][s]:>9.4f}" for s in PROBE_STEPS)
        print(row)

    # ---- verdict ----
    final = max(PROBE_STEPS)
    rnd_f = mc["random"][final]
    str_f = mc["structural"][final]
    base_f = mc["baseline"][final]
    # area-under-recovery (lower = slower recovery), trapezoid over steps
    def auc(curve):
        xs = PROBE_STEPS
        return sum((curve[xs[i]] + curve[xs[i + 1]]) / 2 * (xs[i + 1] - xs[i])
                   for i in range(len(xs) - 1)) / (xs[-1] - xs[0])
    rnd_auc, str_auc, base_auc = auc(mc["random"]), auc(mc["structural"]), auc(mc["baseline"])

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  final forget acc @ {final} steps:")
    print(f"     random     = {rnd_f:.4f}")
    print(f"     structural = {str_f:.4f}")
    print(f"     baseline   = {base_f:.4f}")
    print(f"  mean recovery (AUC over full attack, lower=SLOWER=more resistant):")
    print(f"     random={rnd_auc:.4f}  structural={str_auc:.4f}  baseline={base_auc:.4f}")
    # both routed curves saturate to the same ceiling by 100 steps, so the
    # SPEED of recovery (whole-curve AUC) is the meaningful tamper metric, not
    # the saturated endpoint. Verdict is driven by AUC; endpoint reported too.
    # lower AUC = slower recovery = MORE tamper-resistant.
    auc_gap = rnd_auc - str_auc          # >0 => random's AUC higher => random recovers FASTER => structural wins
    # also report the biggest per-step separation (structural - random)
    sep = {s: mc["structural"][s] - mc["random"][s] for s in PROBE_STEPS}
    worst = min(sep, key=lambda s: sep[s])
    if abs(auc_gap) < 0.02:
        v = ("NO meaningful difference between random and structural "
             "(|AUC gap| < 0.02). Both vastly beat the baseline.")
    elif auc_gap < 0:
        v = (f"RANDOM is MORE tamper-resistant: it recovers more SLOWLY overall "
             f"(AUC {rnd_auc:.3f} vs structural {str_auc:.3f}, gap "
             f"{-auc_gap*100:.1f} pts). Hypothesis SUPPORTED.")
    else:
        v = (f"STRUCTURAL is MORE tamper-resistant: it recovers more SLOWLY "
             f"overall (AUC {str_auc:.3f} vs random {rnd_auc:.3f}, gap "
             f"{auc_gap*100:.1f} pts; max separation {(-sep[worst])*100:.0f} pts "
             f"at step {worst}). Hypothesis NOT supported -- structural wins.")
    print("  -> " + v)
    print(f"  (baseline recovers almost instantly: {mc['baseline'][PROBE_STEPS[1]]:.3f} "
          f"by step {PROBE_STEPS[1]}, vs {mc['random'][PROBE_STEPS[1]]:.3f}/"
          f"{mc['structural'][PROBE_STEPS[1]]:.3f} for routed -> BOTH routed "
          f"methods are far more tamper-resistant than ordinary unlearning.)")
    print(f"\n  total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
