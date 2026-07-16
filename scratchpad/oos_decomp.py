#!/usr/bin/env python
"""Decompose WHY ownrevert failed OOS and WHY bigsize survived.

1. Split ownrevert into its two sleeves (family vs own-price) and score each alone,
   IS vs OOS -> which sleeve broke?
2. ALGO (asset 0) concentration: share of gross exposure and of PnL, per strategy.
3. Signal IC decay: family-reversion and own-price-reversion IC, IS window vs OOS window.
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.join(ROOT, 'strategy'))
os.chdir(ROOT)

from tools.research import backtest, loadPrices, score, featureIC
import strategy.family_cluster_ownrevert as OR

prc = loadPrices()
IS = prc[:, :500]


def sc(seg):
    mu, sd = float(np.mean(seg)), float(np.std(seg))
    return score(mu, sd)


def split_scores(pnl):
    return sc(pnl), sc(pnl[:125]), sc(pnl[125:])


# ---------- 1. SLEEVE DECOMPOSITION of ownrevert ----------
# Rebuild getMyPosition but keep only ONE sleeve. Each has its own held-position memory
# so the no-trade band behaves exactly like the live strategy.
def make_sleeve(which):
    state = {"prev": None, "nt": None}

    def hold(target):
        target = np.asarray(target, dtype=float)
        if state["prev"] is None or state["prev"].shape != target.shape:
            state["prev"] = target.copy()
        else:
            moved = np.abs(target - state["prev"]) > OR.NO_TRADE_BAND * np.maximum(np.abs(state["prev"]), 1.0)
            state["prev"] = np.where(moved, target, state["prev"])
        return state["prev"].astype(int)

    def fn(prcSoFar):
        nInst, nt = prcSoFar.shape
        if state["nt"] is None or nt != state["nt"] + 1:
            state["prev"] = None
        state["nt"] = nt
        if nt < OR.CLUSTER_WINDOW + 2:
            return hold(np.zeros(nInst))
        prices = prcSoFar[:, -1]
        dailyRet = np.diff(np.log(prcSoFar), axis=1)
        if which == "family":
            dollarPos = OR._family_dollars(prcSoFar, dailyRet)
        else:
            dollarPos = OR._own_dollars(prcSoFar, dailyRet)
        dollarPos = dollarPos * OR._vol_scale(prcSoFar)
        return hold(dollarPos / prices)
    return fn


print("=" * 78)
print("1. OWNREVERT SLEEVE DECOMPOSITION  (Score / H1 / H2)")
print("=" * 78)
print(f"{'sleeve':<16} | {'IS full':>8} {'IS H1':>7} {'IS H2':>7} | {'OOS full':>8} {'OOS H1':>7} {'OOS H2':>7}")
print("-" * 78)
for which in ("family", "own"):
    fn = make_sleeve(which)
    mi = backtest(fn, IS, 250, return_series=True)
    mo = backtest(fn, prc, 250, return_series=True)
    fi, h1i, h2i = split_scores(mi["pnl"])
    fo, h1o, h2o = split_scores(mo["pnl"])
    print(f"{which+' sleeve':<16} | {fi:>8.1f} {h1i:>7.1f} {h2i:>7.1f} | {fo:>8.1f} {h1o:>7.1f} {h2o:>7.1f}")
print("\n(family sleeve here uses FAMILY_GROSS=750k, the reduced budget ownrevert gave it,")
print(" NOT bigsize's 2.5M -- so its OOS is weaker than the standalone bigsize champion.)")

# ---------- 2. ALGO CONCENTRATION ----------
print("\n" + "=" * 78)
print("2. ALGO (asset 0) CONCENTRATION out-of-sample (days 501-750)")
print("=" * 78)
import importlib
strat_mods = {
    "bigsize":   "family_cluster_bigsize",
    "ownrevert": "family_cluster_ownrevert",
    "algo_custom": "family_cluster_algo_custom",
}
print(f"{'strategy':<12} | {'OOS totalPnL':>12} {'ALGO PnL':>10} {'ALGO %':>7} | "
      f"{'|gross| avg':>12} {'ALGO gross%':>11}")
print("-" * 78)
for tag, modname in strat_mods.items():
    mod = importlib.import_module(f"strategy.{modname}")
    mod._prevPos = None; mod._prevNt = None
    m = backtest(mod.getMyPosition, prc, 250, return_attribution=True)
    pbi = m["pnlByInst"]                 # (250, nInst)
    total = pbi.sum()
    algo = pbi[:, 0].sum()
    algo_pnl_pct = 100 * algo / total if total != 0 else float('nan')
    # gross exposure share: re-run collecting |dollar position| per day
    # approximate via |pnl| share is misleading; instead use share of abs-position dollars.
    # Recompute positions to measure gross $ share held in ALGO.
    mod._prevPos = None; mod._prevNt = None
    grossAlgo = 0.0; grossAll = 0.0
    for t in range(500, 750):
        pos = mod.getMyPosition(prc[:, :t])
        px = prc[:, t - 1]
        dollars = np.abs(pos * px)
        grossAll += dollars.sum(); grossAlgo += dollars[0]
    algo_gross_pct = 100 * grossAlgo / grossAll if grossAll else float('nan')
    print(f"{tag:<12} | {total:>12,.0f} {algo:>10,.0f} {algo_pnl_pct:>6.0f}% | "
          f"{grossAll/250:>12,.0f} {algo_gross_pct:>10.0f}%")

# ---------- 3. SIGNAL IC DECAY (IS vs OOS) ----------
print("\n" + "=" * 78)
print("3. RAW SIGNAL IC: in-sample (251-500) vs out-of-sample (501-750)")
print("   IC = daily cross-sectional Spearman(signal, next-day return), averaged.")
print("   POSITIVE IC = the bet pays; near-zero/negative = edge gone.")
print("=" * 78)

def feat_family(prcSoFar):
    """ownrevert family reversion signal (pre-vol-scaling), per asset."""
    dailyRet = np.diff(np.log(prcSoFar), axis=1)
    d = OR._family_dollars(prcSoFar, dailyRet)   # already sized; sign/rank is what matters
    return d

def feat_own(prcSoFar):
    """own-price reversion signal (pre-vol-scaling): -z of today vs prior OWN_WINDOW days."""
    nInst, nt = prcSoFar.shape
    logP = np.log(prcSoFar)
    out = np.zeros(nInst)
    if nt < OR.OWN_WINDOW + 2:
        return out
    for i in range(nInst):
        prior = logP[i, -OR.OWN_WINDOW - 1:-1]
        s = prior.std()
        if s < 1e-9:
            continue
        out[i] = -np.clip((logP[i, -1] - prior.mean()) / s, -OR.OWN_CLIP, OR.OWN_CLIP)
    return out

print(f"{'signal':<18} | {'IS meanIC':>10} {'IS t':>6} | {'OOS meanIC':>11} {'OOS t':>6}")
print("-" * 60)
for nm, fn in [("family reversion", feat_family), ("own-price reversion", feat_own)]:
    ri = featureIC(fn, IS, 250)
    ro = featureIC(fn, prc, 250)
    print(f"{nm:<18} | {ri['meanIC']:>+10.4f} {ri['tstat']:>+6.2f} | "
          f"{ro['meanIC']:>+11.4f} {ro['tstat']:>+6.2f}")

# own-price IC per regime-half of OOS
print("\nOwn-price reversion IC by OOS half (to see WHERE it broke):")
ro1 = featureIC(feat_own, prc[:, :625], 125)   # OOS H1 = days 501-625
ro2 = featureIC(feat_own, prc, 125)            # OOS H2 = days 626-750
print(f"  OOS H1 (501-625): meanIC {ro1['meanIC']:+.4f} (t {ro1['tstat']:+.2f})")
print(f"  OOS H2 (626-750): meanIC {ro2['meanIC']:+.4f} (t {ro2['tstat']:+.2f})")
