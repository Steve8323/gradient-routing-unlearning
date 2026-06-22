import json, statistics as st, base64, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M  = json.load(open("results.json"))
Fx = json.load(open("results_fix.json"))
Bn = json.load(open("results_benefit.json"))

def agg(data, **f):
    rows = [r for r in data if all((abs(r[k]-v) < 1e-9 if isinstance(v, float) else r[k] == v) for k, v in f.items())]
    if not rows: return None
    keys = [k for k in ["forget_pre","retain_pre","forget_pruned","retain_pruned"] if k in rows[0]]
    return {k: round(st.mean(x[k] for x in rows), 3) for k in keys}

MLS = [0.0, 0.25, 0.5, 0.75, 1.0]
# --- hardcoded agent-run means (real numbers from the 3 parallel agents) ---
PSEE = [0.0, 0.25, 0.5, 0.75, 1.0]
FRON = {"random":{"f":[.913,.796,.759,.386,.0],"r":[.908,.916,.906,.909,.299]},
        "structural":{"f":[.908,.889,.893,.832,.0],"r":[.902,.917,.913,.921,.459]}}
STEPS = [0,10,20,30,50,75,100]
TAMP = {"baseline":[.038,.911,.957,.965,.964,.965,.966],
        "structural":[.0,.004,.312,.640,.833,.893,.908],
        "random":[.0,.014,.549,.808,.893,.916,.927]}
MCAP = {"random":{"pruneA":(.0,.651,.823),"pruneB":(.956,.0,.865)},
        "structural":{"pruneA":(.0,.727,.690),"pruneB":(.925,.0,.820)}}
INDEP = {"random":0.828,"structural":0.602}

fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))
# A: original delete-keep
cA = [("Control", agg(M,mode="control",partition="random",mislabel=0.0)),
      ("Naive\nrand", agg(M,mode="routing",partition="random",mislabel=0.0)),
      ("Naive\nstruct", agg(M,mode="routing",partition="structural",mislabel=0.0)),
      ("Abl-aware\nrand", agg(Fx,partition="random",mislabel=0.0,aa=True)),
      ("Abl-aware\nstruct", agg(Fx,partition="structural",mislabel=0.0,aa=True))]
x=range(len(cA)); w=.38
ax[0,0].bar([i-w/2 for i in x],[c[1]["forget_pruned"] for c in cA],w,label="forget (LOW)",color="#d62728")
ax[0,0].bar([i+w/2 for i in x],[c[1]["retain_pruned"] for c in cA],w,label="retain (HIGH)",color="#2ca02c")
ax[0,0].set_xticks(list(x)); ax[0,0].set_xticklabels([c[0] for c in cA],fontsize=8); ax[0,0].set_ylim(0,1)
ax[0,0].set_title("A. Delete forget / keep retain (mislabel 0)",fontsize=10); ax[0,0].legend(fontsize=8); ax[0,0].set_ylabel("acc after prune")
# B: capacity sweep
PW=[0.3,0.5,0.7,0.9]
ax[0,1].plot(PW,[agg(Bn,partition="random",p_waste=p)["retain_pruned"] for p in PW],"o-",color="#1f77b4",label="random")
ax[0,1].plot(PW,[agg(Bn,partition="structural",p_waste=p)["retain_pruned"] for p in PW],"s-",color="#ff7f0e",label="structural")
ax[0,1].set_xlabel("waste fraction (how much you delete)"); ax[0,1].set_ylabel("retain kept after prune"); ax[0,1].set_ylim(.7,1)
ax[0,1].set_title("B. Heavy-deletion capacity (forget=0 both)",fontsize=10); ax[0,1].legend(fontsize=8)
ax[0,1].annotate("random degrades\ngracefully",(0.9,.903),(0.45,.78),fontsize=8,arrowprops=dict(arrowstyle="->",color="#1f77b4"))
# C: frontier
ax[0,2].plot(PSEE,FRON["random"]["f"],"^-",color="#1f77b4",label="random")
ax[0,2].plot(PSEE,FRON["structural"]["f"],"s-",color="#ff7f0e",label="structural")
ax[0,2].set_xlabel("p_see (how often retain sees waste)"); ax[0,2].set_ylabel("mislabeled forget after prune (LOW=absorbed)")
ax[0,2].set_title("C. Absorption frontier (retain≈0.91 for both)",fontsize=10); ax[0,2].legend(fontsize=8); ax[0,2].set_ylim(-.03,1)
ax[0,2].annotate("random absorbs\nmore at same retain",(0.75,.386),(0.05,.30),fontsize=8,arrowprops=dict(arrowstyle="->",color="#1f77b4"))
# D: original absorption vs mislabel
nr=[agg(M,mode="routing",partition="random",mislabel=ml)["forget_pruned"] for ml in MLS]
ns=[agg(M,mode="routing",partition="structural",mislabel=ml)["forget_pruned"] for ml in MLS]
ar=[agg(Fx,partition="random",mislabel=ml,aa=True)["forget_pruned"] for ml in [0.0,0.5,0.75]]
ax[1,0].plot(MLS,nr,"o-",color="#d62728",label="naive random"); ax[1,0].plot(MLS,ns,"s-",color="#ff7f0e",label="naive structural")
ax[1,0].plot([0.0,0.5,0.75],ar,"^--",color="#1f77b4",label="ablation-aware random")
ax[1,0].set_xlabel("fraction of forget MISLABELED"); ax[1,0].set_ylabel("forget after prune (LOW=deleted)"); ax[1,0].set_ylim(-.03,1)
ax[1,0].set_title("D. Naive absorption vs the clean-scalpel fix",fontsize=10); ax[1,0].legend(fontsize=8)
# E: multicap independence
ax[1,1].bar(["random","structural"],[INDEP["random"],INDEP["structural"]],color=["#1f77b4","#ff7f0e"])
ax[1,1].set_ylim(0,1); ax[1,1].set_ylabel("independence score (deletion − spillover)")
ax[1,1].set_title("E. Two independently-deletable capabilities",fontsize=10)
for i,v in enumerate([INDEP["random"],INDEP["structural"]]): ax[1,1].text(i,v+.02,f"{v:.2f}",ha="center",fontsize=10)
# F: tamper recovery
for k,c,ls in [("baseline","#7f7f7f","-"),("structural","#ff7f0e","s-"),("random","#1f77b4","^-")]:
    ax[1,2].plot(STEPS,TAMP[k],ls if len(ls)>1 else "o-",color=c,label=k)
ax[1,2].set_xlabel("relearning-attack steps"); ax[1,2].set_ylabel("forget recovered (LOW=resistant)"); ax[1,2].set_ylim(0,1)
ax[1,2].set_title("F. Tamper-resistance vs ordinary unlearning",fontsize=10); ax[1,2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figure2.png",dpi=125)
b64=base64.b64encode(open("figure2.png","rb").read()).decode()

ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def r4(name,a): return f"<tr><td>{name}</td><td>{a['forget_pruned']}</td><td>{a['retain_pruned']}</td></tr>"
hl="".join(r4(n,c) for n,c in cA)
html=f"""<!doctype html><html><head><meta charset=utf-8><title>Gradient routing: random vs structural</title>
<style>body{{font:15px/1.6 -apple-system,system-ui,sans-serif;max-width:1040px;margin:28px auto;padding:0 18px;color:#1a1a1a}}
h1{{font-size:23px}} h2{{font-size:18px;margin-top:26px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;margin:10px 0;font-size:13.5px}} td,th{{border:1px solid #ccc;padding:4px 9px;text-align:center}}
th{{background:#f3f3f3}} .win{{background:#e8f4ea}} .lose{{background:#fdecea}} code{{background:#f3f3f3;padding:1px 4px;border-radius:3px}}
.box{{background:#f7f9fc;border-left:4px solid #1f77b4;padding:9px 15px;margin:13px 0}}</style></head><body>
<h1>Random-interconnected gradient routing — does scatter-then-prune beat a clean region?</h1>
<p><b>Idea (yours):</b> tag a random, interconnected subset of weights as "waste", route the forget capability into them, prune them at test time. <b>Testbed:</b> MNIST MLP 784-256-256-10 (h=128 for the agent runs), forget={{0..4}} vs retain={{5..9}}, CPU, multi-seed, independently replicated. Chance ≈ 0.20.</p>
<img src="data:image/png;base64,{b64}" style="width:100%;border:1px solid #ddd;border-radius:6px">

<h2>Scorecard: random interconnected vs structural (whole-neuron)</h2>
<table>
<tr><th>advantage</th><th>winner</th><th>magnitude</th></tr>
<tr class=win><td>Heavy-deletion capacity (panel B)</td><td>random</td><td>at 90% waste: retain 0.90 vs 0.80 — structural amputates whole neurons</td></tr>
<tr class=win><td>Absorption↔retain frontier (panel C)</td><td>random</td><td>at retain≈0.91, mislabeled-forget deleted to 0.39 vs structural 0.83</td></tr>
<tr class=win><td>Multi-capability composability (panel E)</td><td>random</td><td>independence 0.83 vs 0.60, wins 4/4 seeds; overlapping prunable regions</td></tr>
<tr class=lose><td>Tamper-resistance / relearning (panel F)</td><td>structural</td><td>relearns slower, AUC 0.65 vs 0.72, 6/6 seeds — fewer surviving re-route paths</td></tr>
</table>
<p><b>3 of 4 favor scattered routing.</b> Random wins wherever the benefit is <i>keeping every neuron usable</i> (deep deletion, soft absorption, packing many capabilities into shared neurons). Structural wins only on relearning-resistance, because cutting whole pathways leaves the attacker fewer channels.</p>

<h2>The core result (mislabel 0, after prune)</h2>
<table><tr><th>condition</th><th>forget</th><th>retain</th></tr>{hl}</table>
<p>Naive scattered routing deletes forget (0.0) but the interconnection collapses retain (0.20). <b>Ablation-aware training</b> — train retain on the already-pruned net so main can't lean on waste — fixes it: <b>forget 0.0, retain 0.95</b> with random weights. Control proves the deletion is real (same prune on a normal net leaves forget at 0.93).</p>

<div class=box><b>Mechanistic punchline:</b> absorption (mislabeled forget still deleted) and co-adaptation collateral (retain collapses on prune) are <i>the same knob</i> — does retain's training see the waste region? See-it → absorption + collateral; don't-see-it → clean + no absorption. The frontier (panel C) shows you can dial a middle setting for partial absorption at near-full retain, and random gives a strictly better frontier than structural.</div>

<h2>Bonus (bigger than random-vs-structural)</h2>
<p>Routing-then-prune — <i>either</i> partition — is dramatically more tamper-resistant than ordinary unlearning (panel F). Gradient-ascent unlearning snaps back to 0.91 forget accuracy in <b>10</b> attack steps; routed+pruned needs ~30 just to half-recover. <b>Prune-localized deletion resists relearning in a way suppression (GA/NPO/RMU) does not.</b></p>

<h2>Multi-capability detail (A={{0,1}}, B={{2,3}}, retain={{4..9}})</h2>
<table><tr><th>partition</th><th>scenario</th><th>acc A</th><th>acc B</th><th>acc retain</th></tr>
<tr><td>random</td><td>prune A</td><td>0.00</td><td>0.65</td><td>0.82</td></tr>
<tr><td>random</td><td>prune B</td><td>0.96</td><td>0.00</td><td>0.87</td></tr>
<tr><td>structural</td><td>prune A</td><td>0.00</td><td>0.73</td><td>0.69</td></tr>
<tr><td>structural</td><td>prune B</td><td>0.93</td><td>0.00</td><td>0.82</td></tr></table>

<h2>Caveats</h2>
<p>Tiny model, MNIST, a class-split as a "capability" proxy. Effects are existence-proof dynamics on a toy, multi-seed and replicated but not a frontier claim. Absorption in the naive condition is partly entangled with the collateral; cleanest absorption readout is the frontier (panel C) where retain is held fixed. Magnitudes are honest means; tamper and multicap gaps are consistent in sign across seeds but modest at saturation.</p>
<p style="color:#888;font-size:12px">Generated {ts} · harness.py, fix_experiment.py, benefit_experiment.py + 3 parallel agents (tamper/multicap/frontier) · independently replicated core result</p>
</body></html>"""
out=f"/Users/steven/gradient_routing_experiment/report_{ts}.html"
open(out,"w").write(html); print(out)
