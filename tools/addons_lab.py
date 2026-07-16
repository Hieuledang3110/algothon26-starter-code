#!/usr/bin/env python
"""addons_lab.py -- reusable engine for exploring add-ons to the champion. NOT submitted.

WHAT IT'S FOR
    A place to try "what if we swapped ONE piece of the champion?" experiments and read a
    clean delta, without editing the strategy files. It rebuilds the champion
    (family_cluster_ownrevert) as a configurable, stateful closure so you can swap the
    clustering method or bolt on an extra sleeve and re-score in one line.

    The companion notebook `addon_experiments.ipynb` drives this and records the insights.
    Everything here is measured the way this project judges changes: the full-window Score
    AND the H1/H2 (weak/strong half) split, reusing research.backtest so the numbers match
    the official grader.

QUICK START (REPL or notebook)
    from addons_lab import make_strategy, evaluate, PRC
    evaluate("champion",        make_strategy(family_mode="hard"))
    evaluate("ew halflife=80",  make_strategy(family_mode="ew", ew_halflife=80))
    evaluate("gmm soft",        make_strategy(family_mode="gmm"))
    evaluate("champion + RSI",  make_strategy(rsi_budget=250_000))

    make_strategy() knobs:
      family_mode  "hard" (champion) | "ew" (recency-weighted corr) | "gmm" (soft membership)
      ew_halflife  recency half-life in days for family_mode="ew"
      rsi_budget   >0 adds an RSI reversion sleeve on top (any family_mode)
    Override any champion constant by passing it as a keyword, e.g. FAMILY_GROSS=1_000_000.
"""
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (root_dir, os.path.join(root_dir, "tools")):
    if p not in sys.path:
        sys.path.append(p)
if os.path.basename(os.getcwd()) == "tools":
    os.chdir("..")

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from research import loadPrices, score, backtest

PRC = loadPrices()
N_INST, N_T = PRC.shape

# champion defaults (mirror family_cluster_ownrevert.py)
CFG = dict(
    CLUSTER_WINDOW=120, N_CLUSTERS=6, REVERT_WINDOW=60, VOL_WINDOW=20,
    FAMILY_GROSS=750_000, OWN_WINDOW=3, OWN_BUDGET=2_000_000, OWN_CLIP=2.0,
    OWN_INST0_MULT=10.0, OWN_NEUTRALIZE=True, NO_TRADE_BAND=0.5,
    MKT_VOL_SHORT=20, MKT_VOL_LONG=100, VOL_FLOOR=0.2, VOL_K=1.0,
    GMM_DIMS=8, EW_DATA=350,
)


# ============================ clustering options ============================
def hard_labels(logRet, cfg):
    """Champion: hard hierarchical clusters from a flat-window correlation."""
    corr = np.nan_to_num(np.corrcoef(logRet), nan=0.0)
    dist = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    tree = linkage(squareform(dist, checks=False), method="average")
    return fcluster(tree, t=cfg["N_CLUSTERS"], criterion="maxclust")


def ew_labels(returns, halflife, cfg):
    """Recency-weighted correlation (newest day weight 1, half-life decay), then cluster."""
    T = returns.shape[1]
    w = 0.5 ** (np.arange(T - 1, -1, -1) / halflife)
    w = w / w.sum()
    mean = (returns * w).sum(axis=1, keepdims=True)
    dm = returns - mean
    cov = (dm * w) @ dm.T
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = np.nan_to_num(cov / np.outer(d, d), nan=0.0)
    dist = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    tree = linkage(squareform(dist, checks=False), method="average")
    return fcluster(tree, t=cfg["N_CLUSTERS"], criterion="maxclust")


def family_dollars_hard(dailyRet, labels, cfg):
    """The champion's cross-sectional family snap-back, given hard cluster labels."""
    n = dailyRet.shape[0]
    recentRet = dailyRet[:, -cfg["REVERT_WINDOW"]:]
    pullAway = np.zeros(n)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        familyRet = recentRet[idx].mean(axis=0)
        fc = familyRet - familyRet.mean()
        denom = np.dot(fc, fc)
        if denom <= 0:
            continue
        memberC = recentRet[idx] - recentRet[idx].mean(axis=1, keepdims=True)
        rideShare = (memberC @ fc) / denom
        leftover = recentRet[idx] - np.outer(rideShare, familyRet)
        pullAway[idx] = leftover.sum(axis=1)
    return _shape_signal(pullAway, dailyRet, cfg, cfg["FAMILY_GROSS"])


def family_dollars_gmm(dailyRet, cfg):
    """SOFT membership: PCA recent returns -> GMM soft-assign -> blend each asset's family move."""
    from sklearn.mixture import GaussianMixture
    n = dailyRet.shape[0]
    win = dailyRet[:, -cfg["CLUSTER_WINDOW"]:]
    X = win - win.mean(axis=1, keepdims=True)
    X = X / (X.std(axis=1, keepdims=True) + 1e-12)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    K = min(cfg["GMM_DIMS"], U.shape[1])
    feats = U[:, :K] * S[:K]
    gm = GaussianMixture(n_components=cfg["N_CLUSTERS"], covariance_type="full",
                         reg_covar=1e-3, random_state=0, n_init=1)
    gm.fit(feats)
    R = gm.predict_proba(feats)
    recentRet = dailyRet[:, -cfg["REVERT_WINDOW"]:]
    Rsum = R.sum(axis=0) + 1e-12
    clusterMean = (R.T @ recentRet) / Rsum[:, None]
    pullAway = np.zeros(n)
    for i in range(n):
        softFam = R[i] @ clusterMean
        fc = softFam - softFam.mean()
        denom = fc @ fc
        if denom <= 0:
            continue
        member = recentRet[i] - recentRet[i].mean()
        beta = (member @ fc) / denom
        leftover = recentRet[i] - beta * softFam
        pullAway[i] = leftover.sum()
    return _shape_signal(pullAway, dailyRet, cfg, cfg["FAMILY_GROSS"])


def _shape_signal(pullAway, dailyRet, cfg, budget):
    """Common tail: fade the pull-away, size down jumpy names, scale to a $ budget."""
    n = dailyRet.shape[0]
    signal = -(pullAway - pullAway.mean())
    vol = np.clip(np.std(dailyRet[:, -cfg["VOL_WINDOW"]:], axis=1), 1e-8, None)
    signal = signal / vol
    gross = np.sum(np.abs(signal))
    return np.zeros(n) if gross < 1e-12 else signal / gross * budget


# ============================ extra sleeves ============================
def own_dollars(prcSoFar, dailyRet, cfg):
    """The champion's own-price sleeve: each asset vs its own prior OWN_WINDOW-day mean."""
    n, ntt = prcSoFar.shape
    if ntt < max(cfg["OWN_WINDOW"], cfg["VOL_WINDOW"]) + 2:
        return np.zeros(n)
    logPrc = np.log(prcSoFar)
    raw = np.zeros(n)
    for i in range(n):
        prior = logPrc[i, -cfg["OWN_WINDOW"] - 1:-1]
        sd = prior.std()
        if sd < 1e-9:
            continue
        z = (logPrc[i, -1] - prior.mean()) / sd
        raw[i] = np.clip(-z, -cfg["OWN_CLIP"], cfg["OWN_CLIP"])
    vol = np.clip(np.std(dailyRet[:, -cfg["VOL_WINDOW"]:], axis=1), 1e-8, None)
    weight = raw / vol
    if cfg["OWN_NEUTRALIZE"]:
        weight = weight - weight.mean()
    weight[0] *= cfg["OWN_INST0_MULT"]
    gross = np.sum(np.abs(weight))
    return np.zeros(n) if gross < 1e-12 else weight / gross * cfg["OWN_BUDGET"]


def rsi_dollars(prcSoFar, dailyRet, cfg, n_rsi, budget):
    """RSI reversion sleeve (oversold -> long), same equal-risk / balanced sizing."""
    n = prcSoFar.shape[0]
    if budget <= 0 or prcSoFar.shape[1] < n_rsi + 2:
        return np.zeros(n)
    delta = np.diff(prcSoFar[:, -(n_rsi + 1):], axis=1)
    gain = np.clip(delta, 0, None).mean(axis=1)
    loss = np.clip(-delta, 0, None).mean(axis=1)
    rs = gain / (loss + 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    raw = np.clip((50.0 - rsi) / 50.0, -1.0, 1.0)
    vol = np.clip(np.std(dailyRet[:, -cfg["VOL_WINDOW"]:], axis=1), 1e-8, None)
    weight = (raw / vol)
    weight = weight - weight.mean()
    weight[0] *= cfg["OWN_INST0_MULT"]
    gross = np.sum(np.abs(weight))
    return np.zeros(n) if gross < 1e-12 else weight / gross * budget


def vol_scale(prcSoFar, cfg):
    mkt = np.diff(np.log(prcSoFar), axis=1).mean(axis=0)
    if len(mkt) < cfg["MKT_VOL_LONG"]:
        return 1.0
    recent = mkt[-cfg["MKT_VOL_SHORT"]:].std()
    normal = mkt[-cfg["MKT_VOL_LONG"]:].std()
    if recent <= 0:
        return 1.0
    return float(np.clip((normal / recent) ** cfg["VOL_K"], cfg["VOL_FLOOR"], 1.0))


# ============================ strategy builder ============================
def make_strategy(family_mode="hard", ew_halflife=80, rsi_budget=0.0, rsi_n=14, **overrides):
    """Return a stateful getMyPosition closure for one experiment configuration.

    family_mode: "hard" (champion) | "ew" (recency-weighted corr) | "gmm" (soft membership)
    ew_halflife: half-life (days) when family_mode="ew"
    rsi_budget:  >0 adds an RSI reversion sleeve on top of any family_mode
    **overrides: override any champion constant, e.g. FAMILY_GROSS=1_000_000
    """
    cfg = {**CFG, **overrides}
    state = {"prevPos": None, "prevNt": None}

    def _hold(target):
        target = np.asarray(target, dtype=float)
        pp = state["prevPos"]
        if pp is None or pp.shape != target.shape:
            state["prevPos"] = target.copy()
        else:
            moved = np.abs(target - pp) > cfg["NO_TRADE_BAND"] * np.maximum(np.abs(pp), 1.0)
            state["prevPos"] = np.where(moved, target, pp)
        return state["prevPos"].astype(int)

    def getMyPosition(prcSoFar):
        n, ntt = prcSoFar.shape
        if state["prevNt"] is None or ntt != state["prevNt"] + 1:
            state["prevPos"] = None
        state["prevNt"] = ntt
        if ntt < cfg["CLUSTER_WINDOW"] + 2:
            return _hold(np.zeros(n))

        prices = prcSoFar[:, -1]
        dailyRet = np.diff(np.log(prcSoFar), axis=1)

        if family_mode == "hard":
            labels = hard_labels(dailyRet[:, -cfg["CLUSTER_WINDOW"]:], cfg)
            fam = family_dollars_hard(dailyRet, labels, cfg)
        elif family_mode == "ew":
            dwin = min(cfg["EW_DATA"], dailyRet.shape[1])
            labels = ew_labels(dailyRet[:, -dwin:], ew_halflife, cfg)
            fam = family_dollars_hard(dailyRet, labels, cfg)
        elif family_mode == "gmm":
            fam = family_dollars_gmm(dailyRet, cfg)
        else:
            raise ValueError(f"unknown family_mode {family_mode!r}")

        dollarPos = fam + own_dollars(prcSoFar, dailyRet, cfg)
        if rsi_budget > 0:
            dollarPos = dollarPos + rsi_dollars(prcSoFar, dailyRet, cfg, rsi_n, rsi_budget)
        dollarPos = dollarPos * vol_scale(prcSoFar, cfg)
        return _hold(dollarPos / prices)

    return getMyPosition


def _split_scores(pnl, half):
    def sc(seg):
        mu, sd = float(np.mean(seg)), float(np.std(seg))
        sr = np.sqrt(250) * mu / sd if sd > 0 else 0.0
        return score(mu, sd), sr
    return sc(pnl), sc(pnl[:half]), sc(pnl[half:])


def evaluate(name, fn, days=250, prc=None, quiet=False):
    """Backtest fn and print/return full + H1/H2 Score (the project's judging metric)."""
    prc = PRC if prc is None else prc
    m = backtest(fn, prc, days, return_series=True)
    (fs, fsh), (h1, h1sh), (h2, h2sh) = _split_scores(m["pnl"], days // 2)
    res = dict(name=name, full=fs, full_sh=fsh, h1=h1, h2=h2,
               sharpe=m["annSharpe"], turn=m["avgDailyTurnover"], mean=m["meanPL"])
    if not quiet:
        print(f"  {name:34s} full {fs:6.1f} (Sh {fsh:4.2f}) | H1 {h1:6.1f} | H2 {h2:6.1f} | "
              f"turn ${m['avgDailyTurnover']:,.0f}")
    return res


if __name__ == "__main__":
    print(f"Loaded {N_INST} x {N_T}\n")
    evaluate("champion (hard)", make_strategy("hard"))
    for hl in (20, 80, 200, 320):
        evaluate(f"ew halflife={hl}", make_strategy("ew", ew_halflife=hl))
    evaluate("gmm soft", make_strategy("gmm"))
    for b in (250_000, 500_000, 1_000_000):
        evaluate(f"+ RSI14 budget={b:,}", make_strategy("hard", rsi_budget=b))
