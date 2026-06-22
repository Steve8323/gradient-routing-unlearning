# Random-interconnected gradient routing for unlearning

A small, multi-seed, independently-replicated MNIST testbed for one question:

> Can you tag a **random, interconnected** subset of a network's weights as "waste",
> route a forget-capability into them during training, and **prune them at test time**
> to delete that capability — without a clean structural region?

Short answer: **yes**, and scattered weights beat whole-neuron regions on most axes.

## Key findings (toy-scale existence proofs, not frontier claims)

- **Deletion is real, not a pruning artifact.** Pruning the same random 50% of a
  *normally-trained* net leaves forget accuracy at 0.93; with routing it goes to **0.0**.
  The forget capability genuinely concentrates into the prunable weights (control-verified).
- **Naive scatter-routing is a sledgehammer:** forget → 0.0 but retain collapses to 0.20,
  because main weights co-adapt to waste activations.
- **Ablation-aware training is the fix** (train the retain loss on the *already-pruned* net
  so main can't depend on waste): **forget 0.0, retain 0.95** with random scattered weights.
- **Absorption ↔ co-adaptation are one knob** — does retain's training see the waste region?
  See-it → absorption (mislabeled forget still deleted) **and** retain collapse;
  don't-see-it → clean retain **and** no absorption. A middle setting buys partial absorption
  at near-full retain.
- **Random interconnected vs structural (whole-neuron):**
  | advantage | winner | magnitude |
  |---|---|---|
  | heavy-deletion capacity | random | at 90% waste: retain 0.90 vs 0.80 |
  | absorption↔retain frontier | random | at retain≈0.91: forget 0.39 vs 0.83 |
  | multi-capability composability | random | independence 0.83 vs 0.60 (4/4 seeds) |
  | relearning-resistance | structural | recovery AUC 0.65 vs 0.72 (6/6 seeds) |
- **Bonus (bigger than random-vs-structural):** routing-then-prune — *either* partition — is
  far more tamper-resistant than ordinary gradient-ascent unlearning. GA unlearning relearns
  to 0.91 in **10** attack steps; routed+pruned needs ~30 to half-recover. Prune-localized
  deletion resists relearning where suppression (GA/NPO/RMU) does not.

## Files

| file | what |
|---|---|
| `harness.py` | core routing + matrix (control, naive routing, absorption sweep) |
| `replication_indep.py` | independent re-implementation (cross-check, no shared code) |
| `fix_experiment.py` | ablation-aware training (the clean scalpel) |
| `benefit_experiment.py` | p_waste sweep (heavy-deletion capacity) |
| `tradeoff_frontier_experiment.py` | absorption↔retain frontier (p_see mixing) |
| `multi_capability_experiment.py` | two independently-deletable capabilities |
| `tamper_experiment.py` | relearning-attack tamper-resistance |
| `build_report*.py` | HTML report builders |
| `results_*.json` | raw multi-seed results |

## Run

```bash
python3 harness.py --matrix          # full condition matrix -> results.json
python3 fix_experiment.py            # ablation-aware clean scalpel
python3 benefit_experiment.py        # capacity sweep
python3 build_report2.py             # regenerate the HTML report
```

CPU-only, MNIST auto-downloads to `data/` (gitignored). Each experiment runs in seconds–minutes.

## Caveats

Tiny MLP, MNIST, a class-split as a "capability" proxy. Multi-seed and replicated, but these
are mechanism-level existence proofs on a toy — not validated at frontier scale.
