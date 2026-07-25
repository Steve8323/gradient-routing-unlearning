# Random-interconnected gradient routing for unlearning

A small, multi-seed, independently-replicated MNIST testbed for one question:

> Can you tag a **random, interconnected** subset of a network's weights as "waste",
> route a forget-capability into them during training, and **prune them at test time**
> to delete that capability — without a clean structural region?

Short answer: **yes**, and scattered weights beat whole-neuron regions on most axes.

> **Update (2026-07-24):** a follow-up sprint stress-tested these claims. The effects
> below reproduce, but several were resting on the exact-test floor at the saved seed
> counts (p=0.10–0.25) and are only established significant after a fresh n=6 replication.
> The tamper-resistance advantage also weakens under an attacker-tuned LoRA. See
> [Follow-up experiments](#follow-up-experiments-2026-07-24).

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
  | relearning-resistance | structural | recovery AUC 0.65 vs 0.72 (fresh n=6, p=0.0156) |

  Every row above was re-checked in the 2026-07-24 sprint. The saved 3–4-seed runs sit
  at their exact-test floor (p=0.10–0.25): the effects reproduce exactly, but were not
  significant at the saved n. A fresh n=6 replication recovers two of the directions with
  significance — structural better on relearning resistance (recovery AUC 0.736→0.650,
  p=0.0156) and random better on retain at heavy deletion (0.928 vs 0.883, p=0.0156).
- **Bonus (bigger than random-vs-structural), now qualified:** against ordinary
  gradient-ascent relearning, routing-then-prune — *either* partition — is far more
  tamper-resistant (GA relearns to 0.91 in **10** steps; routed+pruned needs ~30 to
  half-recover). But that advantage is attack-specific: an attacker-tuned LoRA (rank 16,
  lr 1e-2) shrinks routed+prune's edge over gradient-ascent from −0.37 to ~−0.06 AUC and
  half-recovers the deleted capability in ~10–20 steps. Prune-localized deletion resists
  *weak* relearning where suppression (GA/NPO/RMU) does not — but any tamper-resistance
  number that has not faced an attacker-tuned LoRA is provisional. See the follow-up
  section.

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

## Follow-up experiments (2026-07-24)

A 30-agent sprint tried to break the claims above rather than confirm them. Four results,
each run in a private research archive (described here, not linked):

1. **Stats downgrade, then recovery.** Every published random-vs-structural comparison was
   sitting at its exact-test floor for the saved 3–4 seeds (p=0.10–0.25). That means the
   *directions* reproduce exactly, but none were established significant at the saved n.
   Re-running at n=6 recovers both headline directions with significance: structural wins
   relearning resistance (recovery AUC 0.736→0.650, p=0.0156), random wins retain at heavy
   deletion (0.928 vs 0.883, p=0.0156). So the effects are real; the original error bars
   were just too small to prove it.

2. **The random↔structural trade is regime-dependent.** A mass-fixed mask-interpolation
   knob (α slides the same waste budget from fully scattered toward whole-neuron) shows the
   two partitions aren't a fixed win/lose — the frontier changes shape with how much you
   delete. At p_waste=0.5, resistance improves monotonically with α at only ~0.6-pt retain
   cost, so pure structural (α=1) dominates. At p_waste=0.9 the frontier collapses to
   effectively two points: resistance shows up *only* at the pure-structural endpoint, and
   that same endpoint is where the 4.5-pt retain cost lands. Heavy deletion is where
   structural stops being free.

3. **Weak as a defense.** The tamper-resistance advantage is only against a weak attacker.
   An attacker-tuned LoRA (rank 16, lr 1e-2) shrinks routed+prune's edge over gradient-ascent
   from −0.37 to ~−0.06 AUC and half-recovers the deleted capability in ~10–20 steps — but
   it does so by destroying retain, i.e. capability *extraction*, not stealthy repair. It
   also out-attacks the repo's own freeze-masked full-FT attack by +0.10 AUC, and a
   masked-LoRA control confirms recovery does *not* require touching the pruned coordinates.
   Honest lesson: any tamper-resistance number that has not faced an attacker-tuned LoRA is
   provisional.

4. **Scope: you can only route what fires.** Routing+prune deletes *demonstrated* behavior,
   but cannot pin a latent worldview that no in-distribution datum exercises. In a 154-param
   worldview testbed it moved the trait no better than doing nothing (p=0.25 vs no
   intervention), while dose-matched fine-tuning on disagreeing data aligned 20/20 — there is
   nothing to route if the trait never fires during training. Separately, the post-unlearning
   replacement behavior is largely convergent: 7/8 seeds adopt the identical forget→retain
   map, with seed-dependence confined to a ~19% boundary fringe.

Net: the mechanism holds, but three of the louder claims — significance, "random beats
structural everywhere", and tamper-resistance-as-defense — needed narrowing.

## Caveats

Tiny MLP, MNIST, a class-split as a "capability" proxy. Multi-seed and replicated, but these
are mechanism-level existence proofs on a toy — not validated at frontier scale.
