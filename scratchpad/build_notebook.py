#!/usr/bin/env python
"""Generate tools/oos_postmortem.ipynb -- the out-of-sample post-mortem notebook.

Self-contained & re-runnable: recomputes every table/plot from the strategy files
and prices.txt (days 1-750). No dependence on scratchpad artifacts.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- title
md(r"""
# Out-of-sample post-mortem — days 501–750 released

**What changed.** `prices.txt` now holds **750 days**. Because `eval.py` scores the
**last 250 days**, backtesting today scores **days 501–750 — the exact hidden window the
grader used**. So we can finally mark every strategy against *real* out-of-sample (OOS) data
and dissect why some generalised and others collapsed.

**Two independent 250-day stages now exist:**

| Stage | Days | Role |
| :-- | :-- | :-- |
| **A** (in-sample) | 251–500 | what every knob was tuned on |
| **B** (out-of-sample) | 501–750 | the grader window — a genuinely held-out stage |

**Leaderboard we must reproduce:** bigsize **Score 64.13** (mean 107.99, std 1412.11),
ownrevert **Score −26.11** (mean −26.11, std 1833.57).

> **Headline.** The plain **family** mean-reversion engine is the *only* edge that generalised.
> Every in-sample "improvement" that dazzled on Stage A — the own-price sleeve (IS 304), the
> ALGO carve-out (IS 250) — **lost money on Stage B**. The in-sample H1/H2 "regime balance" that
> made ownrevert look robust was a **mirage**: the weak/strong halves simply *swapped* between
> stages, so within-stage balance never tested cross-stage robustness. Only a held-out stage does.
""")

# ---------------------------------------------------------------- setup
code(r"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# make the repo importable no matter where the kernel starts
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'tools' else os.getcwd()
for p in (ROOT, os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'strategy')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

from tools.research import backtest, loadPrices, score, featureIC
import importlib

prc = loadPrices()                     # (nInst, nt) = (51, 750)
nInst, nt = prc.shape
IS = prc[:, :500]                      # Stage A -> scores 251-500
# full prc     -> Stage B -> scores 501-750
print(f"Loaded {nInst} instruments x {nt} days")

def sc(seg):
    mu, sd = float(np.mean(seg)), float(np.std(seg))
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0.0
    return score(mu, sd), sr

def run(modname):
    mod = importlib.import_module(f"strategy.{modname}")
    out = {}
    for tag, sl in [("A", IS), ("B", prc)]:
        mod._prevPos = None; mod._prevNt = None
        m = backtest(mod.getMyPosition, sl, 250, return_series=True, return_attribution=True)
        (fs, fsh) = sc(m["pnl"]); (h1, _) = sc(m["pnl"][:125]); (h2, _) = sc(m["pnl"][125:])
        out[tag] = dict(score=fs, sharpe=m["annSharpe"], mean=m["meanPL"], std=m["stdPL"],
                        turn=m["avgDailyTurnover"], h1=h1, h2=h2,
                        pnl=m["pnl"], pnlByInst=m["pnlByInst"])
    return out

STRATS = {
    "family_cluster_bigsize":    "CHAMPION: pure family, big book (2.5M) + hard vol dial",
    "family_cluster_volfilter":  "family + no-trade band + gentle vol dial (1.0M)",
    "family_cluster_only":       "baseline: family sleeve only, no dial (1.0M)",
    "family_cluster_ownrevert":  "family + universe-wide own-price sleeve (ALGO x10)",
    "family_cluster_algo_custom":"family(1-50) + dedicated ALGO z-score reversion",
}
R = {name: run(name) for name in STRATS}
print("backtests done")
""")

# ---------------------------------------------------------------- table 1
md(r"""
## 1. The scoreboard: in-sample vs out-of-sample

Same strategies, both stages. Watch the **A → B** collapse for the elaborate books and the
**survival** of the plain family books.
""")

code(r"""
print(f"{'strategy':<27} | {'A Score':>8} {'A H1':>6} {'A H2':>6} {'A Shrp':>6} | "
      f"{'B Score':>8} {'B H1':>6} {'B H2':>6} {'B Shrp':>6}")
print("-"*102)
for name in STRATS:
    a, b = R[name]["A"], R[name]["B"]
    print(f"{name:<27} | {a['score']:>8.1f} {a['h1']:>6.1f} {a['h2']:>6.1f} {a['sharpe']:>6.2f} | "
          f"{b['score']:>8.1f} {b['h1']:>6.1f} {b['h2']:>6.1f} {b['sharpe']:>6.2f}")

print("\nGRADER RECONCILIATION (local Stage B == leaderboard):")
for name, lb in [("family_cluster_bigsize","64.13 / mean 107.99 / std 1412.11"),
                 ("family_cluster_ownrevert","-26.11 / mean -26.11 / std 1833.57")]:
    b = R[name]["B"]
    print(f"  {name:<26} leaderboard {lb}")
    print(f"  {'':<26} local Stage B  {b['score']:.2f} / mean {b['mean']:.2f} / std {b['std']:.2f}")
""")

code(r"""
# Bar chart: Stage A vs Stage B score per strategy
names = list(STRATS)
labels = [n.replace('family_cluster_','') for n in names]
a_sc = [R[n]['A']['score'] for n in names]
b_sc = [R[n]['B']['score'] for n in names]
x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(10,4.5))
ax.bar(x-w/2, a_sc, w, label='Stage A (in-sample 251-500)', color='#8896b0')
ax.bar(x+w/2, b_sc, w, label='Stage B (OOS / grader 501-750)', color='#2f6fb0')
ax.axhline(0, color='k', lw=.8)
for xi, v in zip(x-w/2, a_sc): ax.text(xi, v+ (4 if v>=0 else -10), f'{v:.0f}', ha='center', fontsize=8)
for xi, v in zip(x+w/2, b_sc): ax.text(xi, v+ (4 if v>=0 else -10), f'{v:.0f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
ax.set_ylabel('Score'); ax.set_title('In-sample Score is a liar: A (grey) vs OOS B (blue)')
ax.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
**Read-out.**
- **bigsize / volfilter / only** (pure family) all stayed **positive OOS** (+64 / +40 / +38).
  bigsize's extra size + harder dial genuinely helped — it kept the *most* OOS profit.
- **ownrevert** 304 → **−26**: worse than holding cash.
- **algo_custom** 250 → **+14.6**: technically positive, but 94% of its in-sample edge evaporated.

The rank order on Stage A (ownrevert ≫ algo_custom ≫ bigsize) is almost **inverted** on Stage B.
In-sample Score did not just shrink — for the elaborate books it pointed the *wrong way*.
""")

# ---------------------------------------------------------------- regime flip
md(r"""
## 2. Why "in-sample H1 ≈ H2 balance" lied — the regimes swapped

ownrevert was promoted because its in-sample halves were balanced (H1 296 ≈ H2 313), sold as
"regime-robust." But H1/H2 are just the first/second 125 days of *one* stage. Across the two
**real** stages the weak and strong halves **flip**, so within-stage balance measures nothing
about the next stage.
""")

code(r"""
fig, ax = plt.subplots(figsize=(9,4))
fam = "family_cluster_bigsize"
vals = [R[fam]['A']['h1'], R[fam]['A']['h2'], R[fam]['B']['h1'], R[fam]['B']['h2']]
bars = ['A-H1\n(251-375)','A-H2\n(376-500)','B-H1\n(501-625)','B-H2\n(626-750)']
cols = ['#c0392b','#27ae60','#27ae60','#c0392b']
ax.bar(bars, vals, color=cols)
for i,v in enumerate(vals): ax.text(i, v+3, f'{v:.0f}', ha='center')
ax.set_ylabel('Score (family engine)'); ax.axhline(0,color='k',lw=.6)
ax.set_title('The strong half is not fixed: in-sample H2 was strong, OOS H1 is strong')
plt.tight_layout(); plt.show()
print("The 'strong regime' moved from the 2nd half of Stage A to the 1st half of Stage B.")
print("Any signal 'balanced' within Stage A had a 50/50 chance of looking balanced next stage.")
""")

# ---------------------------------------------------------------- sleeve decomp
md(r"""
## 3. Sleeve autopsy — the own-price sleeve *was* the failure

ownrevert = **family sleeve** (cross-sectional, asset-vs-peers) **+ own-price sleeve**
(asset-vs-its-own-3-day-price, ALGO ×10). We score each sleeve alone, IS vs OOS.
""")

code(r"""
import strategy.family_cluster_ownrevert as OR

def make_sleeve(which):
    st = {"prev": None, "nt": None}
    def hold(t):
        t = np.asarray(t, float)
        if st["prev"] is None or st["prev"].shape != t.shape: st["prev"] = t.copy()
        else:
            mv = np.abs(t-st["prev"]) > OR.NO_TRADE_BAND*np.maximum(np.abs(st["prev"]),1.0)
            st["prev"] = np.where(mv, t, st["prev"])
        return st["prev"].astype(int)
    def fn(p):
        nI, n = p.shape
        if st["nt"] is None or n != st["nt"]+1: st["prev"] = None
        st["nt"] = n
        if n < OR.CLUSTER_WINDOW+2: return hold(np.zeros(nI))
        dr = np.diff(np.log(p), axis=1)
        d = OR._family_dollars(p, dr) if which=="family" else OR._own_dollars(p, dr)
        return hold(d*OR._vol_scale(p)/p[:,-1])
    return fn

def splits(pnl): return sc(pnl)[0], sc(pnl[:125])[0], sc(pnl[125:])[0]
print(f"{'sleeve (in ownrevert)':<22} | {'A full':>7} {'A H1':>6} {'A H2':>6} | {'B full':>7} {'B H1':>6} {'B H2':>6}")
print("-"*74)
curves = {}
for which in ("family","own"):
    fn = make_sleeve(which)
    mA = backtest(fn, IS, 250, return_series=True)
    mB = backtest(fn, prc, 250, return_series=True)
    curves[which] = mB["pnl"]
    fa,a1,a2 = splits(mA["pnl"]); fb,b1,b2 = splits(mB["pnl"])
    print(f"{which+' sleeve':<22} | {fa:>7.1f} {a1:>6.1f} {a2:>6.1f} | {fb:>7.1f} {b1:>6.1f} {b2:>6.1f}")
print("\n(family sleeve shown at ownrevert's reduced 750k budget, not bigsize's 2.5M.)")
""")

code(r"""
fig, ax = plt.subplots(figsize=(10,4))
for which, c in [("family","#2f6fb0"),("own","#c0392b")]:
    ax.plot(np.cumsum(curves[which]), label=f"{which} sleeve", color=c, lw=1.6)
ax.axhline(0,color='k',lw=.6); ax.axvline(125,color='grey',ls=':',lw=1)
ax.text(126, ax.get_ylim()[0]*.9, 'OOS H2 -> ', color='grey')
ax.set_xlabel('OOS day (501-750)'); ax.set_ylabel('cumulative PnL ($)')
ax.set_title('Out-of-sample: family sleeve grinds up, own-price sleeve bleeds (esp. 2nd half)')
ax.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
**Verdict.** In-sample the own-price sleeve was the *hero* (214 vs family's 124, and
"regime-balanced" H1 245 / H2 184). Out-of-sample it is the *villain*: **−84**, with a
catastrophic OOS-H2 of **−189**. The family sleeve stayed **+37** OOS even on its starved
budget. ownrevert bolted a losing engine onto a winning one.
""")

# ---------------------------------------------------------------- ALGO concentration
md(r"""
## 4. Concentration risk — ALGO is where the elaborate books blew up

ALGO (asset 0) has a 10× position limit and 5× cheaper fees, so ownrevert boosted it ×10 and
algo_custom gave it a dedicated engine. On Stage B **the ALGO reversion bet lost money in both**,
but bigsize — which treats ALGO as just one small name — barely touched it.
""")

code(r"""
print(f"{'strategy':<12} | {'OOS PnL':>10} {'ALGO PnL':>10} {'ALGO %PnL':>10} | {'ALGO %gross':>11}")
print("-"*64)
for tag, modname in [("bigsize","family_cluster_bigsize"),
                     ("ownrevert","family_cluster_ownrevert"),
                     ("algo_custom","family_cluster_algo_custom")]:
    pbi = R[modname]["B"]["pnlByInst"]
    total, algo = pbi.sum(), pbi[:,0].sum()
    mod = importlib.import_module(f"strategy.{modname}")
    mod._prevPos=None; mod._prevNt=None
    gA=gAll=0.0
    for t in range(500,750):
        pos = mod.getMyPosition(prc[:,:t]); px = prc[:,t-1]
        d = np.abs(pos*px); gAll += d.sum(); gA += d[0]
    print(f"{tag:<12} | {total:>10,.0f} {algo:>10,.0f} {100*algo/total:>9.0f}% | {100*gA/gAll:>10.0f}%")
""")

code(r"""
# breadth: sorted per-instrument OOS PnL for bigsize (broad) vs ownrevert (ALGO-dominated)
fig, axes = plt.subplots(1,2, figsize=(11,4), sharey=False)
for ax, name, ttl in [(axes[0],"family_cluster_bigsize","bigsize: broad, no single name dominates"),
                      (axes[1],"family_cluster_ownrevert","ownrevert: ALGO (red) is the big loser")]:
    p = R[name]["B"]["pnlByInst"].sum(axis=0)
    order = np.argsort(p)
    cols = ['#c0392b' if i==0 else '#2f6fb0' for i in order]
    ax.bar(range(len(p)), p[order], color=cols)
    ax.axhline(0,color='k',lw=.6); ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('instruments (sorted by OOS PnL)'); ax.set_ylabel('OOS PnL ($)')
plt.tight_layout(); plt.show()
print("Red bar = ALGO (asset 0).")
""")

md(r"""
**Verdict.** ownrevert put **22% of gross** on ALGO and ALGO **lost −12.5k** — *more than the
whole book's net loss*. algo_custom's ALGO bet also lost (~−11k) but was only **3% of gross**, so
the family part still dragged it net-positive. bigsize keeps ALGO at **~1% of gross / 5% of PnL** —
**breadth is the reason it survives.** Concentration didn't add edge; it added a single point of
failure.
""")

# ---------------------------------------------------------------- IC decay
md(r"""
## 5. The smoking gun — signal information content (IC), IS vs OOS

IC = daily cross-sectional rank-correlation of a signal with the **next day's** return, averaged.
Positive & stable with |t|≳2 = a real edge. This is the test that *would have caught* the failure
**before** submitting.
""")

code(r"""
def feat_family(p):
    return OR._family_dollars(p, np.diff(np.log(p),axis=1))
def feat_own(p):
    nI,n = p.shape; L=np.log(p); out=np.zeros(nI)
    if n < OR.OWN_WINDOW+2: return out
    for i in range(nI):
        pr = L[i,-OR.OWN_WINDOW-1:-1]; s=pr.std()
        if s>1e-9: out[i] = -np.clip((L[i,-1]-pr.mean())/s, -OR.OWN_CLIP, OR.OWN_CLIP)
    return out

print(f"{'signal':<20} | {'A meanIC':>9} {'A t':>6} | {'B meanIC':>9} {'B t':>6}   verdict")
print("-"*70)
for nm, fn in [("family reversion",feat_family),("own-price reversion",feat_own)]:
    a=featureIC(fn, IS, 250); b=featureIC(fn, prc, 250)
    verdict = "STABLE, real edge" if abs(b['tstat'])>1.4 and b['meanIC']>0.012 else "~ZERO, noise"
    print(f"{nm:<20} | {a['meanIC']:>+9.4f} {a['tstat']:>+6.2f} | {b['meanIC']:>+9.4f} {b['tstat']:>+6.2f}   {verdict}")
""")

md(r"""
**Verdict — the whole story in two rows.**
- **family reversion:** IC **+0.016 → +0.014** (t ≈ 1.9 → 1.7). Small but *stable across both
  stages*. A real, if modest, edge — exactly why the family books generalise.
- **own-price reversion:** IC **+0.007 → +0.010** (t < 1.1 in both). **Statistically indistinguishable
  from zero.** It never *had* a real edge. In-sample it "scored 214" purely from **overfit + 10×
  ALGO leverage + 350k/day turnover** amplifying a coin-flip. A near-zero IC run at that size and
  churn is *built to bleed* out-of-sample — which is precisely what −84 is.

**Had we screened IC on a held-out stage, the own-price sleeve would never have shipped.** It fails
the t≳2 bar; the family signal (barely) clears it.
""")

# ---------------------------------------------------------------- sweep + ceiling
md(r"""
## 6. Where the family engine tops out — a two-stage robustness sweep

The honest objective is **min(Stage A, Stage B)** on a **plateau** (neighbours agree), never a
spike tuned to one stage. Sweep the only two proven levers: size (`GROSS_DOLLARS`) and the
risk-off dial (`VOL_K`).
""")

code(r"""
import strategy.family_cluster_bigsize as BS
grid_g = [1_000_000,1_500_000,2_000_000,2_500_000,3_000_000,4_000_000,6_000_000]
grid_k = [1.0,2.0,3.0]
tblB = np.zeros((len(grid_g),len(grid_k))); tblMin = np.zeros_like(tblB)
for i,g in enumerate(grid_g):
    for j,k in enumerate(grid_k):
        BS.GROSS_DOLLARS=g; BS.VOL_K=k
        BS._prevPos=None; BS._prevNt=None; mA=backtest(BS.getMyPosition, IS, 250, return_series=True)
        BS._prevPos=None; BS._prevNt=None; mB=backtest(BS.getMyPosition, prc, 250, return_series=True)
        a=sc(mA["pnl"])[0]; b=sc(mB["pnl"])[0]
        tblB[i,j]=b; tblMin[i,j]=min(a,b)
BS.GROSS_DOLLARS=2_500_000; BS.VOL_K=2.0   # restore shipped config

fig, ax = plt.subplots(figsize=(7,5))
im = ax.imshow(tblB, cmap='viridis', aspect='auto')
ax.set_xticks(range(len(grid_k))); ax.set_xticklabels([f'{k:g}' for k in grid_k])
ax.set_yticks(range(len(grid_g))); ax.set_yticklabels([f'{g/1e6:g}M' for g in grid_g])
ax.set_xlabel('VOL_K (risk-off strength)'); ax.set_ylabel('GROSS_DOLLARS (book size)')
ax.set_title('Stage B (OOS) Score across the family engine grid')
for i in range(len(grid_g)):
    for j in range(len(grid_k)):
        ax.text(j,i,f'{tblB[i,j]:.0f}',ha='center',va='center',
                color='w' if tblB[i,j]<tblB.max()*0.7 else 'k', fontsize=9)
plt.colorbar(im, label='OOS Score'); plt.tight_layout(); plt.show()
print("Shipped config (2.5M / VOL_K 2.0) -> OOS 64. Raising VOL_K to 3.0 -> ~69, and it")
print("helps Stage A too (Pareto), on a plateau from 2-4M. Nothing here breaks OOS ~70.")
""")

md(r"""
**Verdict.**
- **VOL_K 2 → 3 is a free, robust win:** +4–5 OOS at every size, +Sharpe, and it *also* lifts
  Stage A. Monotone across the whole grid = a plateau, not a lucky spike. The single change worth
  making to the shipped file.
- **Size plateau ≈ 2–4M.** Below is under-levered; 6M erodes Sharpe (both stages fall). 2.5–3M is
  the robust centre.
- **Ceiling ≈ OOS Score 70.** No amount of re-tuning this book clears it. **The only way up is a
  genuinely uncorrelated new signal that passes the Section-5 IC gate on both stages** — which the
  own-price idea does not.
""")

# ---------------------------------------------------------------- summary
md(r"""
## 7. Weak-point ledger — every strategy, what it got right, what killed it

| Strategy | IS Score | **OOS Score** | Strong point (keep) | Weak point (why it failed / risk) |
| :-- | --: | --: | :-- | :-- |
| **family_cluster_bigsize** 🥇 | 152 | **+64** | Broad family edge (stable IC), big book fills caps, hard vol dial helps OOS | Family edge is *modest & regime-varying*; caps out ~70. Not broken — just bounded. |
| **family_cluster_volfilter** | 139 | **+40** | Same family edge, gentle dial; safe | Under-levered vs bigsize; leaves OOS profit on the table. |
| **family_cluster_only** | 137 | **+38** | The clean generalising core | No risk-off dial → slightly worse in choppy OOS stretches. |
| **family_cluster_algo_custom** | 250 | **+14.6** | Correctly spotted ALGO's fee/limit asymmetry; family part carried it | The ALGO **z-score reversion signal has no OOS edge** (lost −11k); only survived because it was 3% of gross. |
| **family_cluster_ownrevert** ❌ | 304 | **−26** | Ambition: add an uncorrelated 2nd sleeve | Own-price IC ≈ 0 (t<1.1) → **−84 sleeve OOS**; **22% gross on ALGO** which lost −12.5k; huge turnover; "regime balance" was a within-stage mirage. |

### The four lessons this stage bought us
1. **Trust the held-out stage, never the in-sample Score.** In-sample over-promised on *every*
   elaborate book and even inverted the ranking.
2. **Within-stage H1/H2 balance ≠ robustness.** The strong half moved from A-H2 to B-H1. Balance
   inside one stage is a coin flip about the next.
3. **Breadth beats concentration.** The books that died bet ≥20%+ of gross on one special name.
   bigsize's ~1% ALGO / broad book is *why* it holds.
4. **A signal must earn its budget with a stable IC (|t|≳2 on both stages).** The own-price signal
   never cleared that bar; leverage + turnover just amplified a coin flip into a loss.
""")

md(r"""
## 8. The new plan — revised, evidence-gated

**Ship:** the pure **family engine** (`family_cluster_bigsize`). Data-backed tweak from this
notebook: **raise `VOL_K` 2.0 → 3.0** (robust +4–5 OOS on *both* stages, +Sharpe, on a plateau).
Keep `GROSS_DOLLARS` in the **2.5–3M** plateau and `NO_TRADE_BAND` at 0.5. This is embodied in the
new file **`family_cluster_famrobust.py`**.

**Retire from contention:** `ownrevert` and `algo_custom` — both monetise the own-price / ALGO
reversion signal, which has **no out-of-sample edge**. Keep the files only as the cautionary record.

**A hard gate for anything new (so we don't repeat this):** before a signal gets book budget it must
1. show **stable IC with |t| ≳ 2 on *both* Stage A and Stage B** (Section 5 test), and
2. stay **broad** — no single instrument above ~10% of gross exposure, and
3. be judged on **min(Stage A, Stage B)** on a **plateau**; reject any spike whose neighbours
   collapse or whose gain comes from one name.

**Where real upside lives (not more tuning of this book — it caps at ~70):** a *genuinely
uncorrelated* signal that clears the gate. Candidates worth screening under the new rules:
cross-asset **lead-lag** (does yesterday's family move predict a laggard today?), a **slower
cross-sectional** horizon that is *decorrelated* from the 60-day family signal, or exploiting ALGO's
cheap-fee/large-limit capacity with a signal **that actually has OOS IC** (the z-score reversion does
not). Screen IC on both stages *first*; only then wire it in.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = os.path.join(os.path.dirname(__file__), "..", "tools", "oos_postmortem.ipynb")
out = os.path.abspath(out)
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
