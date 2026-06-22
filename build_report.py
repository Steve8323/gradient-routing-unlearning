import json, statistics as st, base64, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = json.load(open("results.json")); Fx = json.load(open("results_fix.json"))

def agg(data, **f):
    rows = [r for r in data if all((abs(r[k]-v) < 1e-9 if isinstance(v, float) else r[k] == v) for k, v in f.items())]
    if not rows: return None
    return {k: round(st.mean(x[k] for x in rows), 3) for k in ["forget_pre","retain_pre","forget_pruned","retain_pruned"]}

MLS = [0.0, 0.25, 0.5, 0.75, 1.0]
condsA = [
    ("Control\n(no routing)",      agg(M, mode="control", partition="random", mislabel=0.0)),
    ("Naive\nrandom",              agg(M, mode="routing", partition="random", mislabel=0.0)),
    ("Naive\nstructural",          agg(M, mode="routing", partition="structural", mislabel=0.0)),
    ("Abl-aware\nrandom",          agg(Fx, partition="random", mislabel=0.0, aa=True)),
    ("Abl-aware\nstructural",      agg(Fx, partition="structural", mislabel=0.0, aa=True)),
]
naive_r  = [agg(M, mode="routing", partition="random", mislabel=ml) for ml in MLS]
naive_s  = [agg(M, mode="routing", partition="structural", mislabel=ml) for ml in MLS]
aa_r     = [agg(Fx, partition="random", mislabel=ml, aa=True) for ml in [0.0, 0.5, 0.75]]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
# Panel A
x = range(len(condsA)); w = 0.38
ax[0].bar([i-w/2 for i in x], [c[1]["forget_pruned"] for c in condsA], w, label="forget (want LOW)", color="#d62728")
ax[0].bar([i+w/2 for i in x], [c[1]["retain_pruned"] for c in condsA], w, label="retain (want HIGH)", color="#2ca02c")
ax[0].axhline(0.2, ls=":", c="gray", lw=1); ax[0].text(4.1, 0.22, "chance", color="gray", fontsize=8)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels([c[0] for c in condsA], fontsize=8)
ax[0].set_ylabel("test accuracy after pruning"); ax[0].set_ylim(0, 1)
ax[0].set_title("A. Delete forget, keep retain? (mislabel=0)", fontsize=10); ax[0].legend(fontsize=8)
# Panel B: absorption (forget_pruned vs mislabel)
ax[1].plot(MLS, [c["forget_pruned"] for c in naive_r], "o-", color="#d62728", label="naive random")
ax[1].plot(MLS, [c["forget_pruned"] for c in naive_s], "s-", color="#ff7f0e", label="naive structural")
ax[1].plot([0.0,0.5,0.75], [c["forget_pruned"] for c in aa_r], "^--", color="#1f77b4", label="ablation-aware random")
ax[1].set_xlabel("fraction of forget data MISLABELED as retain"); ax[1].set_ylabel("forget acc after prune (LOW = deleted)")
ax[1].set_ylim(-0.03, 1); ax[1].set_title("B. Absorption: is mislabeled forget still deleted?", fontsize=10); ax[1].legend(fontsize=8)
ax[1].annotate("naive: stays deleted\nto 75% mislabel\n(absorption)", (0.5, 0.05), (0.15, 0.45), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="#d62728"))
ax[1].annotate("fix breaks\nabsorption", (0.5, 0.92), (0.55, 0.55), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="#1f77b4"))
# Panel C: retain_pruned vs mislabel
ax[2].plot(MLS, [c["retain_pruned"] for c in naive_r], "o-", color="#d62728", label="naive random")
ax[2].plot(MLS, [c["retain_pruned"] for c in naive_s], "s-", color="#ff7f0e", label="naive structural")
ax[2].plot([0.0,0.5,0.75], [c["retain_pruned"] for c in aa_r], "^--", color="#1f77b4", label="ablation-aware random")
ax[2].axhline(0.2, ls=":", c="gray", lw=1)
ax[2].set_xlabel("fraction of forget data MISLABELED as retain"); ax[2].set_ylabel("retain acc after prune (HIGH = preserved)")
ax[2].set_ylim(0, 1); ax[2].set_title("C. Retain preservation", fontsize=10); ax[2].legend(fontsize=8)
plt.tight_layout()
png = "figure.png"; plt.savefig(png, dpi=130)
b64 = base64.b64encode(open(png, "rb").read()).decode()

def row(name, a):
    return f"<tr><td>{name}</td><td>{a['forget_pre']}</td><td>{a['retain_pre']}</td><td><b>{a['forget_pruned']}</b></td><td><b>{a['retain_pruned']}</b></td></tr>"
table = "".join(row(n, c) for n, c in condsA)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
html = f"""<!doctype html><html><head><meta charset=utf-8><title>Gradient Routing micro-experiment</title>
<style>body{{font:15px/1.6 -apple-system,system-ui,sans-serif;max-width:1000px;margin:30px auto;padding:0 18px;color:#1a1a1a}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;margin:12px 0;font-size:14px}} td,th{{border:1px solid #ccc;padding:5px 10px;text-align:center}}
th{{background:#f3f3f3}} .k{{background:#fffae6}} code{{background:#f3f3f3;padding:1px 4px;border-radius:3px}}
.box{{background:#f7f9fc;border-left:4px solid #1f77b4;padding:10px 16px;margin:14px 0}}</style></head><body>
<h1>Random-interconnected gradient routing: does scatter-then-prune delete a capability?</h1>
<p><b>Question (your idea):</b> tag a random, interconnected ~50% of weights as "waste", route forget-class learning into them, block retain-class learning from them, and <b>prune the waste at test time</b> to delete the forget capability. Does it work without a clean structural region?</p>
<p><b>Setup:</b> MNIST, MLP 784-256-256-10, forget={{0..4}}, retain={{5..9}}, 6k train / 2k test, 8 epochs, CPU. 3 seeds. Routing on the two hidden weight matrices. Chance ≈ 0.20 (5 classes). Independently replicated by a second agent with separate code (forget→0.0, retain→0.21 — matches).</p>
<img src="data:image/png;base64,{b64}" style="width:100%;border:1px solid #ddd;border-radius:6px">
<h2>Headline numbers (mislabel = 0, mean over 3 seeds)</h2>
<table><tr><th>condition</th><th>forget pre</th><th>retain pre</th><th>forget pruned</th><th>retain pruned</th></tr>{table}</table>
<h2>What the experiment found</h2>
<ol>
<li><b>The deletion is real, not a pruning artifact.</b> Pruning the same random 50% on a <i>normally-trained</i> net (control) leaves forget at <b>0.93</b> and retain at <b>0.87</b> — nets shrug off random 50% pruning. So routing's forget→<b>0.0</b> is genuine localization: the forget capability really was concentrated into the prunable weights.</li>
<li><b>Naive random routing is a sledgehammer, not a scalpel.</b> It deletes forget (0.0) but collapses retain (0.20 ≈ chance). The control proves this is <i>not</i> the prune's fault (control retain survives at 0.87) — it's <b>co-adaptation</b>: main weights learn to depend on waste activations during training, so zeroing waste pulls the rug out.</li>
<li><b>Ablation-aware training makes it a true scalpel.</b> Train the retain loss on the <i>already-pruned</i> net so main can't depend on waste → <b>forget 0.0, retain 0.95</b>, even with random scattered weights. Clean targeted deletion with a random interconnected mask.</li>
<li><b>Absorption is real and strong — in the naive version.</b> Even with <b>75% of forget data mislabeled</b> as retain (only 25% correctly routed), pruning still deletes all forget (forget_pruned = 0.0); it only returns at 100% mislabel. A minority correctly-routed seed absorbs the rest.</li>
</ol>
<div class=box><b>The tradeoff this surfaced (the interesting bit):</b> the ablation-aware fix that preserves retain <b>destroys absorption</b> — at 50% mislabel, forget_pruned jumps from 0.0 to 0.92. Both effects are the <i>same knob</i>: does main's training see the waste region? <b>If it does</b> → main co-adapts (retain collapses on prune) <b>and</b> waste preempts main from learning mislabeled forget (absorption). <b>If it doesn't</b> (ablation-aware) → no co-adaptation (clean prune) <b>and</b> no preemption (mislabeled forget leaks into main, absorption dies). Co-adaptation-collateral and absorption are two faces of one coin.</div>
<h2>Caveats</h2>
<p>Tiny model (sub-1M params), MNIST, a 5-vs-5 class split as a "capability" proxy; absorption in the naive condition is partly entangled with the co-adaptation collapse (cleanest absorption readout is structural, where retain partly survives: at 75% mislabel, forget 0.0 / retain 0.84). p_waste fixed at 0.5. These are existence-proof dynamics on a toy, not a frontier claim — but they reproduce the SGTM "robust to ~50% label noise" finding and expose a mechanism-level tension worth testing at scale.</p>
<p style="color:#888;font-size:12px">Generated {ts} · harness.py + fix_experiment.py · 3 seeds · independently replicated</p>
</body></html>"""
out = f"/Users/steven/gradient_routing_experiment/report_{ts}.html"
open(out, "w").write(html); print(out)
