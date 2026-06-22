"""Actual-benefit hunt: random vs structural at varying waste fraction, on a
capacity-tight net (h=64), ablation-aware (both are clean scalpels). Does random
preserve MORE retain capability for a given amount of deletion?"""
import json, statistics as st
import torch
from harness import MLP, make_masks, get_data, evaluate, tags
from fix_experiment import step_aa


def run_pw(partition, p_waste, seed, h=64, epochs=8, bs=128, mislabel=0.0):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte, d, src = get_data(seed)
    model = MLP(d, h=h)
    masks = make_masks(model, partition, p_waste, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 7)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            step_aa(model, Xtr[b], ytr[b], tags(ytr[b], mislabel, gen), masks, opt, aa=True)
    f1, r1 = evaluate(model, Xte, yte, masks, True)
    return dict(partition=partition, p_waste=p_waste, seed=seed, forget_pruned=f1, retain_pruned=r1)


res = []
for part in ["random", "structural"]:
    for pw in [0.3, 0.5, 0.7, 0.9]:
        for seed in [0, 1, 2]:
            res.append(run_pw(part, pw, seed))
json.dump(res, open("results_benefit.json", "w"), indent=2)

def agg(part, pw, k):
    rows = [r for r in res if r["partition"] == part and abs(r["p_waste"] - pw) < 1e-9]
    return round(st.mean(x[k] for x in rows), 3)

print("retain_pruned (HIGH=more capability kept) | forget_pruned (LOW=deleted)")
print(f"{'p_waste':>8} | {'random':>16} | {'structural':>16}")
for pw in [0.3, 0.5, 0.7, 0.9]:
    rr, fr = agg('random', pw, 'retain_pruned'), agg('random', pw, 'forget_pruned')
    rs, fs = agg('structural', pw, 'retain_pruned'), agg('structural', pw, 'forget_pruned')
    print(f"{pw:>8} | retain {rr:.3f} forget {fr:.3f} | retain {rs:.3f} forget {fs:.3f}")
