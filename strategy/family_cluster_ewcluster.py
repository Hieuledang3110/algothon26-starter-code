import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# =============================================================================
# STRATEGY: family_cluster_ewcluster  (EXPERIMENT -- superseded by ownrevert)
#
#   THE IDEA WE TESTED (user's idea #2):
#   The champion decides "which assets are family" using ONLY the last 120 days and
#   ignores everything older. Question: does it help to ALSO use older data, but with
#   a smaller weight -- so recent days count most, but a full ~1-year business cycle
#   still informs the grouping? We swap the flat 120-day correlation for a
#   RECENCY-WEIGHTED correlation: today's weight is 1 and each older day's weight
#   decays by a half-life. Small half-life = even more recency-focused than the flat
#   window; large half-life = drags in the older ~1-year data the user wanted to use.
#
#   THE RESULT (250-day in-sample, judged on the H1/H2 weak/strong split):
#     champion (flat 120):     full 304.3 | H1 295.9 | H2 312.6   <- the bar
#     ew half-life = 20:       full 307.6 | H1 272.5 | H2 342.5   (more recency: helps H2, hurts H1)
#     ew half-life = 80:       full 302.7 | H1 305.3 | H2 300.2   (most balanced, but ties baseline)
#     ew half-life = 200:      full 286.1 | H1 250.0 | H2 321.8   (the "1-year" hypothesis)
#     ew half-life = 320:      full 293.4 | H1 266.0 | H2 320.6   (even more old data)
#
#   THE VERDICT: the user's specific hypothesis -- pull in more OLD data to capture the
#   business cycle -- is REFUTED. The long half-lives (200, 320) are the ones that use
#   the most old data, and they are exactly the ones that DROP the weak half (H1 296 ->
#   250-266). No half-life reliably beats the flat-120 champion. This re-confirms the
#   repo's standing lesson: the family engine wants FRESH, recent-only grouping -- the
#   day-to-day churn in the families is a feature, not noise to be smoothed away with
#   history. Kept on record as a clean negative result; default half-life below is the
#   most-balanced one (80), which merely ties the champion.
#
#   Everything else (the snap-back math, own-price sleeve, band, vol dial) is IDENTICAL
#   to family_cluster_ownrevert.py so this is a clean one-component swap.
# =============================================================================

# --- EW clustering knobs (the only real change) -------------------------------
CLUSTER_WINDOW = 120       # kept for the fresh-run guard; EW uses EW_DATA below instead
EW_HALFLIFE = 80           # recency half-life in days (small = more recent; large = more old data)
EW_DATA = 350              # how many days of history the weighted correlation may look back over
N_CLUSTERS = 6
REVERT_WINDOW = 60
VOL_WINDOW = 20
FAMILY_GROSS = 750_000

# --- OWN-PRICE sleeve (unchanged from the champion) ---------------------------
OWN_WINDOW = 3
OWN_BUDGET = 2_000_000
OWN_CLIP = 2.0
OWN_INST0_MULT = 10.0
OWN_NEUTRALIZE = True

# --- shared plumbing (unchanged) ----------------------------------------------
NO_TRADE_BAND = 0.5
MKT_VOL_SHORT = 20
MKT_VOL_LONG = 100
VOL_FLOOR = 0.2
VOL_K = 1.0

_prevPos = None
_prevNt = None


def _ew_corr(returns, halflife):
    """Recency-weighted correlation matrix. Newest day counts 1; each older day is
    down-weighted by the half-life (weight halves every `halflife` days back)."""
    T = returns.shape[1]
    w = 0.5 ** (np.arange(T - 1, -1, -1) / halflife)      # newest weight = 1
    w = w / w.sum()
    mean = (returns * w).sum(axis=1, keepdims=True)
    dm = returns - mean
    cov = (dm * w) @ dm.T                                  # weighted covariance
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(d, d)
    return np.nan_to_num(corr, nan=0.0)


def _family_labels(logRet):
    """Group assets by recency-weighted co-movement (the one change vs the champion)."""
    corr = _ew_corr(logRet, EW_HALFLIFE)
    dist = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    tree = linkage(squareform(dist, checks=False), method="average")
    return fcluster(tree, t=N_CLUSTERS, criterion="maxclust")


def _family_dollars(prcSoFar, dailyRet):
    """Sleeve A: cross-sectional family snap-back, returned as target dollars per asset."""
    nInst = prcSoFar.shape[0]
    dwin = min(EW_DATA, dailyRet.shape[1])
    labels = _family_labels(dailyRet[:, -dwin:])
    recentRet = dailyRet[:, -REVERT_WINDOW:]
    pullAway = np.zeros(nInst)
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
    signal = -(pullAway - pullAway.mean())
    vol = np.clip(np.std(dailyRet[:, -VOL_WINDOW:], axis=1), 1e-8, None)
    signal = signal / vol
    gross = np.sum(np.abs(signal))
    if gross < 1e-12:
        return np.zeros(nInst)
    return signal / gross * FAMILY_GROSS


def _own_dollars(prcSoFar, dailyRet):
    """Sleeve B: each asset vs its OWN recent price (unchanged from the champion)."""
    nInst, nt = prcSoFar.shape
    if nt < max(OWN_WINDOW, VOL_WINDOW) + 2:
        return np.zeros(nInst)
    logPrc = np.log(prcSoFar)
    raw = np.zeros(nInst)
    for i in range(nInst):
        prior = logPrc[i, -OWN_WINDOW - 1:-1]
        sd = prior.std()
        if sd < 1e-9:
            continue
        z = (logPrc[i, -1] - prior.mean()) / sd
        raw[i] = np.clip(-z, -OWN_CLIP, OWN_CLIP)
    vol = np.clip(np.std(dailyRet[:, -VOL_WINDOW:], axis=1), 1e-8, None)
    weight = raw / vol
    if OWN_NEUTRALIZE:
        weight = weight - weight.mean()
    weight[0] *= OWN_INST0_MULT
    gross = np.sum(np.abs(weight))
    if gross < 1e-12:
        return np.zeros(nInst)
    return weight / gross * OWN_BUDGET


def _vol_scale(prcSoFar):
    """Calm market -> ~1.0; choppy market -> shrink toward VOL_FLOOR. Past data only."""
    mkt = np.diff(np.log(prcSoFar), axis=1).mean(axis=0)
    if len(mkt) < MKT_VOL_LONG:
        return 1.0
    recent = mkt[-MKT_VOL_SHORT:].std()
    normal = mkt[-MKT_VOL_LONG:].std()
    if recent <= 0:
        return 1.0
    return float(np.clip((normal / recent) ** VOL_K, VOL_FLOOR, 1.0))


def _hold_or_trade(target):
    """Leave each position alone unless the new target moved a lot (NO_TRADE_BAND)."""
    global _prevPos
    target = np.asarray(target, dtype=float)
    if _prevPos is None or _prevPos.shape != target.shape:
        _prevPos = target.copy()
    else:
        moved = np.abs(target - _prevPos) > NO_TRADE_BAND * np.maximum(np.abs(_prevPos), 1.0)
        _prevPos = np.where(moved, target, _prevPos)
    return _prevPos.astype(int)


def getMyPosition(prcSoFar):
    global _prevPos, _prevNt
    nInst, nt = prcSoFar.shape
    if _prevNt is None or nt != _prevNt + 1:
        _prevPos = None
    _prevNt = nt

    if nt < CLUSTER_WINDOW + 2:
        return _hold_or_trade(np.zeros(nInst))

    prices = prcSoFar[:, -1]
    dailyRet = np.diff(np.log(prcSoFar), axis=1)
    dollarPos = _family_dollars(prcSoFar, dailyRet) + _own_dollars(prcSoFar, dailyRet)
    dollarPos = dollarPos * _vol_scale(prcSoFar)
    return _hold_or_trade(dollarPos / prices)
