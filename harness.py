"""
Random-interconnected gradient routing on MNIST.

Question: can you tag a RANDOM (scattered, interconnected) subset of weights as
"waste", route forget-class learning into them while blocking retain-class
learning from them, and PRUNE them at test time to delete the forget capability
-- without a clean structural region?

Routing rule (design 'a'):
  - retain-tagged example  -> updates ONLY main weights in routed layers
  - forget-tagged example  -> updates ONLY waste weights in routed layers
  Both groups use the FULL network in the forward pass (so the waste region can
  'preempt'/absorb the loss -> tests absorption with mislabeled data).
Prune = zero the waste-masked weight entries at eval.

Forget classes = {0,1,2,3,4}; retain = {5,6,7,8,9}.
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = "/Users/steven/gradient_routing_experiment"
FORGET = set(range(5))
ROUTED = ["fc1.weight", "fc2.weight"]
torch.set_num_threads(8)


def get_data(seed):
    try:
        from torchvision import datasets
        tr = datasets.MNIST(ROOT + "/data", train=True, download=True)
        te = datasets.MNIST(ROOT + "/data", train=False, download=True)
        Xtr = tr.data.float().view(-1, 784) / 255.0; ytr = tr.targets.clone()
        Xte = te.data.float().view(-1, 784) / 255.0; yte = te.targets.clone()
        src = "MNIST"
    except Exception as e:
        from sklearn.datasets import load_digits
        d = load_digits()
        X = torch.tensor(d.data, dtype=torch.float32) / 16.0
        y = torch.tensor(d.target, dtype=torch.long)
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
        X, y = X[idx], y[idx]; n = int(0.8 * len(X))
        Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]; src = "sklearn_digits(%s)" % type(e).__name__
    g = torch.Generator().manual_seed(seed)
    def sub(X, y, k):
        if len(X) <= k: return X, y
        i = torch.randperm(len(X), generator=g)[:k]; return X[i], y[i]
    Xtr, ytr = sub(Xtr, ytr, 6000); Xte, yte = sub(Xte, yte, 2000)
    return Xtr, ytr, Xte, yte, Xtr.shape[1], src


class MLP(nn.Module):
    def __init__(self, d, h=256, c=10):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.fc2 = nn.Linear(h, h); self.fc3 = nn.Linear(h, c)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def make_masks(model, partition, p_waste, seed):
    g = torch.Generator().manual_seed(seed + 999)
    named = dict(model.named_parameters()); masks = {}
    for n in ROUTED:
        W = named[n]
        if partition == "random":
            m = (torch.rand(W.shape, generator=g) < p_waste).float()       # scattered per-weight
        else:
            rows = (torch.rand(W.shape[0], generator=g) < p_waste).float()  # whole output units
            m = rows.view(-1, 1).expand_as(W).contiguous()
        masks[n] = m
    return masks


def tags(y, mislabel, gen):
    isf = torch.tensor([1 if int(t) in FORGET else 0 for t in y])
    tag = isf.clone()
    if mislabel > 0:
        fidx = (isf == 1).nonzero().flatten()
        k = int(round(mislabel * len(fidx)))
        if k > 0:
            flip = fidx[torch.randperm(len(fidx), generator=gen)[:k]]
            tag[flip] = 0   # mislabel forget -> retain (cannot update waste)
    return tag


def step(model, X, y, tag, masks, routing, opt):
    params = list(model.parameters()); pn = [n for n, _ in model.named_parameters()]
    acc = [torch.zeros_like(p) for p in params]
    for grp, update_waste in [(1, True), (0, False)]:
        sel = (tag == grp)
        if sel.sum() == 0: continue
        loss = F.cross_entropy(model(X[sel]), y[sel])
        gs = torch.autograd.grad(loss, params)
        w = sel.sum().item() / len(tag)
        for i, (n, g) in enumerate(zip(pn, gs)):
            if routing and n in ROUTED:
                wm = masks[n]
                g = g * wm if update_waste else g * (1 - wm)
            acc[i] += w * g
    opt.zero_grad()
    for p, g in zip(params, acc): p.grad = g
    opt.step()


def evaluate(model, X, y, masks, prune):
    backup = {}
    if prune:
        for n, p in model.named_parameters():
            if n in ROUTED:
                backup[n] = p.data.clone(); p.data = p.data * (1 - masks[n])
    with torch.no_grad():
        pred = model(X).argmax(1)
    if prune:
        for n, p in model.named_parameters():
            if n in ROUTED: p.data = backup[n]
    isf = torch.tensor([1 if int(t) in FORGET else 0 for t in y])
    f = (pred[isf == 1] == y[isf == 1]).float().mean().item()
    r = (pred[isf == 0] == y[isf == 0]).float().mean().item()
    return round(f, 4), round(r, 4)


def run(mode, partition, mislabel, seed, epochs=8, p_waste=0.5, bs=128, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte, d, src = get_data(seed)
    model = MLP(d)
    masks = make_masks(model, partition, p_waste, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 7)
    routing = (mode == "routing")
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            step(model, Xtr[b], ytr[b], tags(ytr[b], mislabel, gen), masks, routing, opt)
    f0, r0 = evaluate(model, Xte, yte, masks, prune=False)
    f1, r1 = evaluate(model, Xte, yte, masks, prune=True)
    out = dict(mode=mode, partition=partition, mislabel=mislabel, seed=seed, src=src,
               forget_pre=f0, retain_pre=r0, forget_pruned=f1, retain_pruned=r1)
    if verbose: print(json.dumps(out))
    return out


def matrix():
    res = []
    t0 = time.time()
    for partition in ["random", "structural"]:
        for seed in [0, 1, 2]:
            for ml in [0.0, 0.25, 0.5, 0.75, 1.0]:
                res.append(run("routing", partition, ml, seed))
            res.append(run("control", partition, 0.0, seed))
            print(f"  done {partition} seed{seed}  ({time.time()-t0:.0f}s)")
    json.dump(res, open(ROOT + "/results.json", "w"), indent=2)
    print("wrote results.json,", len(res), "runs,", f"{time.time()-t0:.0f}s")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--mode", default="routing"); ap.add_argument("--partition", default="random")
    ap.add_argument("--mislabel", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    if a.matrix: matrix()
    else: run(a.mode, a.partition, a.mislabel, a.seed, epochs=a.epochs, verbose=True)
