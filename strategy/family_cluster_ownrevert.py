import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# =============================================================================
# STRATEGY: family_cluster_ownrevert
#   Two independent reversion "sleeves" added together:
#     A) FAMILY sleeve  -- the champion's cross-sectional bet (asset vs its peer group).
#     B) OWN-PRICE sleeve -- a NEW, universe-wide short-horizon bet (asset vs its OWN
#        recent price). This is the big change and where most of the new profit comes from.
#
#   WHY THIS EXISTS (the discovery behind it):
#   Gemini's family_cluster_algo_custom.py carved out asset 0 (ALGO) and traded it with a
#   dedicated 5-day "too-far-from-its-own-recent-average -> snap back" signal, which lifted
#   the weak regime a lot. We screened EVERY instrument and found that same own-price
#   snap-back edge is NOT special to ALGO -- it's present across ~14 of the 51 names, and
#   (unlike the family signal) it works in BOTH regime halves H1 and H2. So instead of
#   applying it to one name, we apply it to the whole universe. Result on the 250-day
#   in-sample window: Score ~304 and annualised Sharpe ~2.88, with the weak half (H1) lifted
#   from ~7 (champion) to ~294 -- i.e. the two halves are now roughly EQUAL (H1 294 / H2 314),
#   the regime-stability the champion always lacked.
#
#   KEY DESIGN CHOICE -- the own-price sleeve is made "dollar-balanced" (equal $ long vs
#   short) on purpose. Left un-balanced it made ~30% MORE average profit, but that extra
#   came from an uncontrolled whole-market timing bet ("buy everything after the market
#   dips") -- a single, un-diversifiable wager that we don't trust to survive on new data.
#   Balancing it throws that piece away and keeps only the clean, diversified per-asset
#   snap-back. Flip OWN_NEUTRALIZE to False to take the higher-return / higher-risk version
#   (in-sample Score ~422, but leaning on that market-timing bet).
#
# --- The two ideas in plain words ---------------------------------------------
#   FAMILY (sleeve A): assets come in groups that move together; an asset that has run
#     ahead of its group tends to drift back. We fade that gap. (Regime-fragile: this edge
#     nearly vanishes in choppy stretches -- that's the champion's weakness.)
#   OWN-PRICE (sleeve B): separately, each asset tends to snap back toward its OWN price of
#     the last few days -- if it jumped above its recent 3-day average, lean short; if it
#     dropped below, lean long. This is a different, faster, and (importantly) regime-stable
#     source of profit, so it plugs the champion's weak-regime hole.
#   We size each asset's own-price bet smaller when that asset is jumpy (equal-risk), give
#   asset 0 a bigger share because its position limit is 10x larger and its fees 5x cheaper,
#   keep total buys ~= total sells, then leave positions alone unless the target moved a lot
#   (the "leave-it-alone" band) to avoid burning fees on noise.
# =============================================================================

# --- FAMILY sleeve (sleeve A) -- same knobs as the champion --------------------
CLUSTER_WINDOW = 120       # days of history used to decide which assets are "family"
N_CLUSTERS = 6             # how many families to split the assets into
REVERT_WINDOW = 60         # days over which the family snap-back plays out
VOL_WINDOW = 20            # days used to gauge how jumpy each ASSET is (per-asset bet sizing)
FAMILY_GROSS = 750_000     # target $ exposure for the family sleeve (was 1.0M in the champion;
                           #   smaller here because the own-price sleeve now does the heavy lifting)

# --- OWN-PRICE sleeve (sleeve B) -- the new part ------------------------------
OWN_WINDOW = 3             # look-back for each asset's "own recent average" (3-5 all work; 3 best)
OWN_BUDGET = 2_000_000     # target $ exposure for the own-price sleeve (eval still clips per name)
OWN_CLIP = 2.0             # cap each asset's z-score so one wild day can't dominate the book
OWN_INST0_MULT = 10.0      # give asset 0 ~10x weight: it has a 10x bigger $ limit and 5x cheaper fees
OWN_NEUTRALIZE = True      # True = dollar-balanced (safe, diversified). False = allow the market-
                           #   timing tilt (higher in-sample Score ~422, but a single risky bet).

# --- shared plumbing ----------------------------------------------------------
NO_TRADE_BAND = 0.5        # only re-trade an asset if its target moves >50% of the current size

# --- market "risk-off" dial (kept for safety; now nearly inert) ---------------
#   With the regime-stable own-price sleeve carrying the book, shrinking in choppy markets
#   barely changes anything (Score moves <1 pt across VOL_K 0..2). We keep a gentle dial as
#   cheap insurance in case a future stage is far more turbulent than this one.
MKT_VOL_SHORT = 20         # days to measure the market's *recent* jumpiness
MKT_VOL_LONG = 100         # days to measure the market's *normal* jumpiness (the baseline)
VOL_FLOOR = 0.2            # never shrink the whole book below 20% of full size
VOL_K = 1.0                # how hard we react to choppiness (1 = gentle)

_prevPos = None            # remembers yesterday's target so the leave-it-alone rule can compare
_prevNt = None             # how many days we saw last call -- used to spot a fresh backtest


def _family_labels(logRet):
    """Group assets by how alike their recent daily moves have been. One label per asset."""
    corr = np.nan_to_num(np.corrcoef(logRet), nan=0.0)   # how similarly assets move (1 = identical)
    dist = np.clip(1.0 - corr, 0.0, None)                # turn similarity into a distance
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2                            # make it exactly symmetric for scipy
    tree = linkage(squareform(dist, checks=False), method="average")
    return fcluster(tree, t=N_CLUSTERS, criterion="maxclust")


def _family_dollars(prcSoFar, dailyRet):
    """Sleeve A: cross-sectional family snap-back, returned as target dollars per asset."""
    nInst = prcSoFar.shape[0]
    labels = _family_labels(dailyRet[:, -CLUSTER_WINDOW:])
    recentRet = dailyRet[:, -REVERT_WINDOW:]
    pullAway = np.zeros(nInst)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue                                     # a lone asset has no family to compare to
        familyRet = recentRet[idx].mean(axis=0)          # the family's shared move
        fc = familyRet - familyRet.mean()
        denom = np.dot(fc, fc)
        if denom <= 0:
            continue
        # how much each member rides the shared family move, and what's left over (its own move)
        memberC = recentRet[idx] - recentRet[idx].mean(axis=1, keepdims=True)
        rideShare = (memberC @ fc) / denom
        leftover = recentRet[idx] - np.outer(rideShare, familyRet)
        pullAway[idx] = leftover.sum(axis=1)
    signal = -(pullAway - pullAway.mean())               # short what ran ahead, buy what lagged
    vol = np.clip(np.std(dailyRet[:, -VOL_WINDOW:], axis=1), 1e-8, None)
    signal = signal / vol                                # bet less on jumpier assets
    gross = np.sum(np.abs(signal))
    if gross < 1e-12:
        return np.zeros(nInst)
    return signal / gross * FAMILY_GROSS


def _own_dollars(prcSoFar, dailyRet):
    """Sleeve B: each asset vs its OWN recent price. Returns target dollars per asset.

    For each asset, measure how far today's (log) price sits above/below the average of the
    PRIOR OWN_WINDOW days (today excluded, so it's a clean 'how far did it jump' number), in
    units of that asset's own recent wiggle (a z-score). Bet on the snap-back: above -> short,
    below -> long. Divide by each asset's jumpiness so calm and jumpy names risk the same."""
    nInst, nt = prcSoFar.shape
    if nt < max(OWN_WINDOW, VOL_WINDOW) + 2:
        return np.zeros(nInst)
    logPrc = np.log(prcSoFar)
    raw = np.zeros(nInst)
    for i in range(nInst):
        prior = logPrc[i, -OWN_WINDOW - 1:-1]            # the OWN_WINDOW days BEFORE today
        sd = prior.std()
        if sd < 1e-9:
            continue
        z = (logPrc[i, -1] - prior.mean()) / sd          # how far today jumped from its recent trend
        raw[i] = np.clip(-z, -OWN_CLIP, OWN_CLIP)        # bet on the snap-back, capped
    vol = np.clip(np.std(dailyRet[:, -VOL_WINDOW:], axis=1), 1e-8, None)
    weight = raw / vol                                   # equal-risk sizing across names
    if OWN_NEUTRALIZE:
        weight = weight - weight.mean()                  # force $ long == $ short (drop the market tilt)
    weight[0] *= OWN_INST0_MULT                          # asset 0: bigger limit + cheaper fees -> size up
    gross = np.sum(np.abs(weight))
    if gross < 1e-12:
        return np.zeros(nInst)
    return weight / gross * OWN_BUDGET


def _vol_scale(prcSoFar):
    """Calm market -> ~1.0; choppy market -> shrink the whole book toward VOL_FLOOR. Past data only."""
    mkt = np.diff(np.log(prcSoFar), axis=1).mean(axis=0)   # the average asset's daily move
    if len(mkt) < MKT_VOL_LONG:
        return 1.0
    recent = mkt[-MKT_VOL_SHORT:].std()
    normal = mkt[-MKT_VOL_LONG:].std()
    if recent <= 0:
        return 1.0
    return float(np.clip((normal / recent) ** VOL_K, VOL_FLOOR, 1.0))


def _hold_or_trade(target):
    """Leave each position where it is unless the new target is a lot different (see NO_TRADE_BAND)."""
    global _prevPos
    target = np.asarray(target, dtype=float)
    if _prevPos is None or _prevPos.shape != target.shape:
        _prevPos = target.copy()
    else:
        moved = np.abs(target - _prevPos) > NO_TRADE_BAND * np.maximum(np.abs(_prevPos), 1.0)
        _prevPos = np.where(moved, target, _prevPos)     # only the big-enough changes go through
    return _prevPos.astype(int)


def getMyPosition(prcSoFar):
    global _prevPos, _prevNt
    nInst, nt = prcSoFar.shape
    # If the history didn't just grow by one day, we're in a fresh run (e.g. re-running a
    # backtest in the notebook), so forget the old positions and start the memory clean.
    if _prevNt is None or nt != _prevNt + 1:
        _prevPos = None
    _prevNt = nt

    if nt < CLUSTER_WINDOW + 2:
        return _hold_or_trade(np.zeros(nInst))

    prices = prcSoFar[:, -1]
    dailyRet = np.diff(np.log(prcSoFar), axis=1)         # day-to-day % moves (log), per asset

    # Add the two independent reversion sleeves together, then dial the whole book down if the
    # market is unusually choppy. eval clips each asset to its own $ limit afterwards.
    dollarPos = _family_dollars(prcSoFar, dailyRet) + _own_dollars(prcSoFar, dailyRet)
    dollarPos = dollarPos * _vol_scale(prcSoFar)

    return _hold_or_trade(dollarPos / prices)
