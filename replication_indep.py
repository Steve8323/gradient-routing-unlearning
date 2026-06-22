"""Independent replication of the gradient-routing unlearning experiment.

Written from spec only; does NOT import or read harness.py.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.set_num_threads(4)
DEVICE = torch.device("cpu")

DATA_ROOT = "/Users/steven/gradient_routing_experiment/data"
FORGET = {0, 1, 2, 3, 4}
RETAIN = {5, 6, 7, 8, 9}
N_TRAIN, N_TEST = 6000, 2000
EPOCHS, BATCH, LR = 8, 128, 1e-3


def load_data():
    tfm = transforms.ToTensor()  # scales to [0,1] i.e. /255
    train = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tfm)
    test = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tfm)

    def to_tensors(ds):
        X = ds.data.float().reshape(len(ds), -1) / 255.0
        y = ds.targets.clone()
        return X, y

    Xtr, ytr = to_tensors(train)
    Xte, yte = to_tensors(test)
    return Xtr, ytr, Xte, yte


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_masks(condition, gen):
    """Return waste masks (1 = waste entry) for fc1.weight (256x784) and fc2.weight (256x256).

    'random'     : Bernoulli(0.5) per element.
    'structural' : whole output rows are waste; Bernoulli(0.5) over the 256 rows.
    """
    if condition == "random":
        m1 = (torch.rand(256, 784, generator=gen) < 0.5).float()
        m2 = (torch.rand(256, 256, generator=gen) < 0.5).float()
    elif condition == "structural":
        r1 = (torch.rand(256, generator=gen) < 0.5).float().unsqueeze(1)  # 256x1
        m1 = r1.expand(256, 784).contiguous()
        r2 = (torch.rand(256, generator=gen) < 0.5).float().unsqueeze(1)
        m2 = r2.expand(256, 256).contiguous()
    else:
        raise ValueError(condition)
    return m1, m2


def train_one(condition, routing, seed, Xtr, ytr, Xte, yte):
    """condition in {'random','structural'} controls mask shape.
    routing=True applies gradient routing; routing=False is the control (all weights update).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    m1, m2 = make_masks(condition, gen)  # 1 = waste
    main1, main2 = 1.0 - m1, 1.0 - m2    # 1 = main (non-waste)

    forget_mask_targets = torch.tensor([int(t.item()) in FORGET for t in ytr])

    n = Xtr.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], ytr[idx]
            is_forget = torch.tensor([int(t.item()) in FORGET for t in yb])
            is_retain = ~is_forget
            bsz = xb.shape[0]

            if not routing:
                # Control: standard full-batch update, no masking.
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
                continue

            # Gradient routing: compute each group's gradient separately.
            params = list(model.parameters())
            # accumulate grads here
            total_grads = [torch.zeros_like(p) for p in params]

            # retain group -> updates ONLY main entries of fc1/fc2
            if is_retain.any():
                xr, yr = xb[is_retain], yb[is_retain]
                frac = xr.shape[0] / bsz
                loss_r = F.cross_entropy(model(xr), yr)
                gr = torch.autograd.grad(loss_r, params, retain_graph=False, create_graph=False)
                for j, (p, g) in enumerate(zip(params, gr)):
                    g = g.clone()
                    if p is model.fc1.weight:
                        g = g * main1
                    elif p is model.fc2.weight:
                        g = g * main2
                    total_grads[j] += frac * g

            # forget group -> updates ONLY waste entries of fc1/fc2
            if is_forget.any():
                xf, yf = xb[is_forget], yb[is_forget]
                frac = xf.shape[0] / bsz
                loss_f = F.cross_entropy(model(xf), yf)
                gf = torch.autograd.grad(loss_f, params, retain_graph=False, create_graph=False)
                for j, (p, g) in enumerate(zip(params, gf)):
                    g = g.clone()
                    if p is model.fc1.weight:
                        g = g * m1
                    elif p is model.fc2.weight:
                        g = g * m2
                    total_grads[j] += frac * g

            opt.zero_grad()
            for p, g in zip(params, total_grads):
                p.grad = g
            opt.step()

    # Evaluation helpers
    def eval_split(model):
        with torch.no_grad():
            logits = model(Xte)
            pred = logits.argmax(1)
        forget_idx = torch.tensor([int(t.item()) in FORGET for t in yte])
        retain_idx = ~forget_idx
        fa = (pred[forget_idx] == yte[forget_idx]).float().mean().item()
        ra = (pred[retain_idx] == yte[retain_idx]).float().mean().item()
        return fa, ra

    forget_pre, retain_pre = eval_split(model)

    # Prune: zero waste entries of fc1/fc2.
    with torch.no_grad():
        w1 = model.fc1.weight.data.clone()
        w2 = model.fc2.weight.data.clone()
        model.fc1.weight.data = w1 * main1
        model.fc2.weight.data = w2 * main2
        forget_pruned, retain_pruned = eval_split(model)
        # restore (not strictly needed)
        model.fc1.weight.data = w1
        model.fc2.weight.data = w2

    return dict(forget_pre=forget_pre, retain_pre=retain_pre,
                forget_pruned=forget_pruned, retain_pruned=retain_pruned)


def subset(X, y, n, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    idx = torch.randperm(X.shape[0], generator=g)[:n]
    return X[idx], y[idx]


def main():
    Xtr_full, ytr_full, Xte_full, yte_full = load_data()

    conditions = {
        # name -> (mask_condition, routing)
        "random_routing": ("random", True),
        "structural_routing": ("structural", True),
        "control_no_routing": ("random", False),
    }

    results = {name: {k: [] for k in ("forget_pre", "retain_pre", "forget_pruned", "retain_pruned")}
               for name in conditions}

    for seed in (0, 1, 2):
        Xtr, ytr = subset(Xtr_full, ytr_full, N_TRAIN, seed)
        Xte, yte = subset(Xte_full, yte_full, N_TEST, seed)
        for name, (cond, routing) in conditions.items():
            r = train_one(cond, routing, seed, Xtr, ytr, Xte, yte)
            for k, v in r.items():
                results[name][k].append(v)
            print(f"seed={seed} {name}: {json.dumps({k: round(v,4) for k,v in r.items()})}")

    table = {name: {k: round(float(np.mean(v)), 4) for k, v in d.items()}
             for name, d in results.items()}
    print("\n=== MEAN OVER SEEDS 0,1,2 ===")
    print(json.dumps(table, indent=2))
    return table


if __name__ == "__main__":
    main()
