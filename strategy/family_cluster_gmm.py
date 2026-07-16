import numpy as np
from sklearn.mixture import GaussianMixture

# =============================================================================
# STRATEGY: family_cluster_gmm  (EXPERIMENT -- superseded by ownrevert)
#
#   THE IDEA WE TESTED (user's idea #1):
#   The champion puts each asset in exactly ONE family (a hard grouping). The user's
#   idea: let an asset belong PARTIALLY to several families at once (soft membership),
#   so the "family move" it snaps back toward is a blended, probability-weighted mix
#   rather than one hard group. We use a Gaussian Mixture Model for that:
#     1. take each asset's recent daily moves (120 days) and standardise them,
#     2. compress the 51 assets into a few directions of shared movement (PCA / SVD),
#     3. fit a GMM in that space, which hands back, for every asset, a probability of
#        belonging to each of the 6 clusters (its "soft membership"),
#     4. each asset's expected family move = the probability-weighted blend of all
#        clusters' moves; we then fade how far the asset has pulled away from THAT.
#
#   THE RESULT (250-day in-sample, judged on the H1/H2 weak/strong split):
#     champion (hard clusters):   full 304.3 | H1 295.9 | H2 312.6   <- the bar
#     gmm soft (6 comp, 8 dims):  full 281.1 | H1 279.4 | H2 282.7
#
#   THE VERDICT: soft membership is nicely REGIME-BALANCED (H1 ~= H2) but scores ~23
#   points BELOW the champion. The reason is the same lesson this repo keeps re-learning:
#   blending across clusters SMOOTHS each asset's "family move", and a smoother family
#   move gives a weaker, blurrier snap-back signal. The engine wants a SHARP, decisive
#   grouping, and the day-to-day churn of hard clusters is a feature. You could sharpen
#   the GMM (fewer PCA dims, a "temperature" that pushes memberships toward 0/1) to claw
#   the gap back -- but that just re-approximates hard clustering, which defeats the
#   point. Kept on record as a clean negative result.
#
#   NOTE ON PACKAGES: uses scikit-learn, which IS in the Algothon accepted set, so this
#   file is still submittable with no requirements.txt. Everything after the clustering
#   (own-price sleeve, band, vol dial) is identical to family_cluster_ownrevert.py.
# =============================================================================

# --- clustering knobs ---------------------------------------------------------
CLUSTER_WINDOW = 120
N_CLUSTERS = 6
GMM_DIMS = 8               # how many shared-movement directions to feed the GMM (PCA components)
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


def _family_dollars(prcSoFar, dailyRet):
    """Sleeve A, SOFT version: PCA the recent returns, GMM-soft-assign the assets, then
    fade each asset's pull-away from its probability-blended family move."""
    nInst = prcSoFar.shape[0]
    win = dailyRet[:, -CLUSTER_WINDOW:]
    X = win - win.mean(axis=1, keepdims=True)
    X = X / (X.std(axis=1, keepdims=True) + 1e-12)         # standardise each asset's series
    U, S, _ = np.linalg.svd(X, full_matrices=False)        # shared-movement directions
    K = min(GMM_DIMS, U.shape[1])
    feats = U[:, :K] * S[:K]                                # each asset as a point in that space
    gm = GaussianMixture(n_components=N_CLUSTERS, covariance_type="full",
                         reg_covar=1e-3, random_state=0, n_init=1)
    gm.fit(feats)
    R = gm.predict_proba(feats)                            # (nInst, N_CLUSTERS) soft membership

    recentRet = dailyRet[:, -REVERT_WINDOW:]
    Rsum = R.sum(axis=0) + 1e-12
    clusterMean = (R.T @ recentRet) / Rsum[:, None]        # each cluster's average move
    pullAway = np.zeros(nInst)
    for i in range(nInst):
        softFam = R[i] @ clusterMean                       # this asset's blended family move
        fc = softFam - softFam.mean()
        denom = fc @ fc
        if denom <= 0:
            continue
        member = recentRet[i] - recentRet[i].mean()
        beta = (member @ fc) / denom                       # how much it rides that blended move
        leftover = recentRet[i] - beta * softFam           # what's left = its own move
        pullAway[i] = leftover.sum()
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
