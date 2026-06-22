"""Ablation-aware fix: train the retain (main) group on the ALREADY-PRUNED net,
so main weights cannot co-adapt to waste activations. Predict: forget stays 0,
retain recovers. Reuses verified components from harness.py."""
import json, statistics as st
import torch, torch.nn.functional as F
from harness import MLP, make_masks, tags, get_data, evaluate, ROUTED


def fwd(model, x, masks, prune):
    w1 = model.fc1.weight * (1 - masks["fc1.weight"]) if prune else model.fc1.weight
    h = F.relu(F.linear(x, w1, model.fc1.bias))
    w2 = model.fc2.weight * (1 - masks["fc2.weight"]) if prune else model.fc2.weight
    h = F.relu(F.linear(h, w2, model.fc2.bias))
    return model.fc3(h)


def step_aa(model, X, y, tag, masks, opt, aa):
    params = list(model.parameters()); pn = [n for n, _ in model.named_parameters()]
    acc = [torch.zeros_like(p) for p in params]
    for grp, update_waste in [(1, True), (0, False)]:
        sel = (tag == grp)
        if sel.sum() == 0: continue
        prune = aa and (grp == 0)                      # retain trains on pruned net
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


def run_aa(partition, mislabel, seed, aa, epochs=8, bs=128):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte, d, src = get_data(seed)
    model = MLP(d); masks = make_masks(model, partition, 0.5, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 7)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            step_aa(model, Xtr[b], ytr[b], tags(ytr[b], mislabel, gen), masks, opt, aa)
    f0, r0 = evaluate(model, Xte, yte, masks, False)
    f1, r1 = evaluate(model, Xte, yte, masks, True)
    return dict(partition=partition, mislabel=mislabel, seed=seed, aa=aa,
                forget_pre=f0, retain_pre=r0, forget_pruned=f1, retain_pruned=r1)


if __name__ == "__main__":
    res = []
    for part in ["random", "structural"]:
        for seed in [0, 1, 2]:
            res.append(run_aa(part, 0.0, seed, aa=True))
    for ml in [0.0, 0.5, 0.75]:          # does absorption survive the fix?
        for seed in [0, 1, 2]:
            res.append(run_aa("random", ml, seed, aa=True))
    json.dump(res, open("results_fix.json", "w"), indent=2)

    def agg(part, ml):
        rows = [r for r in res if r["partition"] == part and abs(r["mislabel"] - ml) < 1e-9]
        m = lambda k: round(st.mean(x[k] for x in rows), 3)
        return {k: m(k) for k in ["forget_pre", "retain_pre", "forget_pruned", "retain_pruned"]}
    print("ABLATION-AWARE (mislabel 0):")
    for part in ["random", "structural"]:
        a = agg(part, 0.0)
        print(f"  {part:10s} forget_pruned={a['forget_pruned']}  retain_pruned={a['retain_pruned']}  (pre r={a['retain_pre']})")
    print("ABLATION-AWARE absorption (random):")
    for ml in [0.0, 0.5, 0.75]:
        a = agg("random", ml)
        print(f"  ml{ml}: forget_pruned={a['forget_pruned']}  retain_pruned={a['retain_pruned']}")
