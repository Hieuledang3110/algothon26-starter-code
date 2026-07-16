import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# =============================================================================
# STRATEGY: family_cluster_rsi  (EXPERIMENT -- superseded by ownrevert)
#
#   THE IDEA WE TESTED (user's idea #3):
#   Add a classic technical indicator -- RSI -- as a THIRD reversion sleeve on top of the
#   champion's family + own-price sleeves. RSI measures how one-sided an asset's recent
#   up/down days have been: a very "oversold" reading is a bet it bounces up, "overbought"
#   a bet it falls back. We add it as an equal-risk, dollar-balanced sleeve just like the
#   own-price one. (We also screened MACD -- see the notebook -- but MACD is a MOMENTUM
#   indicator, and in this reversion-heavy market its next-day edge is weakly NEGATIVE, so
#   we didn't build it.)
#
#   WHY RSI IS MOSTLY REDUNDANT (measured, not assumed):
#   RSI and the own-price sleeve are both "asset vs its own recent self" reversion bets, so
#   they overlap. Their daily cross-sectional signal correlation is +0.60 at a 5-day RSI and
#   +0.33 at a 14-day RSI -- i.e. short RSI is almost the same bet we already place, and even
#   14-day RSI is only partly new information.
#
#   THE RESULT (250-day in-sample, judged on the H1/H2 weak/strong split):
#     champion (no RSI):          full 304.3 | H1 295.9 | H2 312.6   <- the bar
#     + RSI14 budget   250k:      full 311.1 | H1 268.3 | H2 353.2
#     + RSI14 budget   500k:      full 299.4 | H1 232.1 | H2 365.5
#     + RSI14 budget 1,000k:      full 249.3 | H1 157.8 | H2 338.9
#     + RSI14 budget 2,000k:      full 171.8 | H1  43.5 | H2 305.5
#
#   THE VERDICT: a small RSI sleeve nudges the FULL-window score up (+7) -- but the wrong
#   way. Every RSI budget HURTS the weak half (H1 296 -> 268 and falling) while piling more
#   into the already-strong H2. This project judges by H1, so trading H1 away for H2 is a
#   losing swap, and RSI wrecks H1 the moment you size it up. RSI adds no regime-stable edge
#   the own-price sleeve doesn't already have. Kept on record; default budget below is the
#   small 250k (the only one that even helps the full window) so the file is a fair
#   representative -- but it is NOT an improvement on the champion.
#
#   Family + own-price + band + vol dial are identical to family_cluster_ownrevert.py.
# =============================================================================

# --- family + own-price knobs (identical to the champion) ---------------------
CLUSTER_WINDOW = 120
N_CLUSTERS = 6
REVERT_WINDOW = 60
VOL_WINDOW = 20
FAMILY_GROSS = 750_000
OWN_WINDOW = 3
OWN_BUDGET = 2_000_000
OWN_CLIP = 2.0
OWN_INST0_MULT = 10.0
OWN_NEUTRALIZE = True

# --- RSI sleeve (the new part) ------------------------------------------------
RSI_WINDOW = 14            # look-back for RSI (14 is the most distinct from own-price; 5 is nearly a duplicate)
RSI_BUDGET = 250_000       # target $ exposure for the RSI sleeve (kept small; larger crushes H1)

# --- shared plumbing (unchanged) ----------------------------------------------
NO_TRADE_BAND = 0.5
MKT_VOL_SHORT = 20
MKT_VOL_LONG = 100
VOL_FLOOR = 0.2
VOL_K = 1.0

_prevPos = None
_prevNt = None


def _family_labels(logRet):
    """Group assets by recent co-movement (identical to the champion)."""
    corr = np.nan_to_num(np.corrcoef(logRet), nan=0.0)
    dist = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    tree = linkage(squareform(dist, checks=False), method="average")
    return fcluster(tree, t=N_CLUSTERS, criterion="maxclust")


def _family_dollars(prcSoFar, dailyRet):
    """Sleeve A: cross-sectional family snap-back (identical to the champion)."""
    nInst = prcSoFar.shape[0]
    labels = _family_labels(dailyRet[:, -CLUSTER_WINDOW:])
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
    """Sleeve B: each asset vs its OWN recent price (identical to the champion)."""
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


def _rsi_dollars(prcSoFar, dailyRet):
    """Sleeve C (NEW): RSI reversion. Oversold (RSI < 50) -> lean long; overbought -> short.
    Same equal-risk sizing, dollar-balanced, asset-0 boost as the own-price sleeve."""
    nInst = prcSoFar.shape[0]
    if RSI_BUDGET <= 0 or prcSoFar.shape[1] < RSI_WINDOW + 2:
        return np.zeros(nInst)
    delta = np.diff(prcSoFar[:, -(RSI_WINDOW + 1):], axis=1)      # last RSI_WINDOW day-to-day moves
    gain = np.clip(delta, 0, None).mean(axis=1)
    loss = np.clip(-delta, 0, None).mean(axis=1)
    rs = gain / (loss + 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    raw = np.clip((50.0 - rsi) / 50.0, -1.0, 1.0)                 # 0..100 RSI -> ~[-1,1] reversion bet
    vol = np.clip(np.std(dailyRet[:, -VOL_WINDOW:], axis=1), 1e-8, None)
    weight = raw / vol
    weight = weight - weight.mean()                              # dollar-balanced
    weight[0] *= OWN_INST0_MULT
    gross = np.sum(np.abs(weight))
    if gross < 1e-12:
        return np.zeros(nInst)
    return weight / gross * RSI_BUDGET


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
    dollarPos = (_family_dollars(prcSoFar, dailyRet)
                 + _own_dollars(prcSoFar, dailyRet)
                 + _rsi_dollars(prcSoFar, dailyRet))
    dollarPos = dollarPos * _vol_scale(prcSoFar)
    return _hold_or_trade(dollarPos / prices)
