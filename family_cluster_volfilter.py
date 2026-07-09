import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# =============================================================================
# STRATEGY: family_cluster_volfilter
#   family_cluster_only  +  a market-volatility "risk-off" dial.
#   Everything about the rubber-band mean-reversion is identical to the baseline; the
#   ONLY addition is rule 6 below: when the whole market suddenly gets choppier than its
#   own normal, we shrink every bet down toward a floor, then let things recover as it
#   calms. Each file is self-contained so either one can be submitted on its own.
# =============================================================================
#
# --- Strategy: fade each asset's move relative to its "family" -----------------
#   1. Assets in this dataset come in families that tend to move together. We find those
#      families automatically by looking at which assets' daily moves have been most
#      correlated over the last 120 days, and grouping them into 6 clusters.
#   2. Within each family, prices behave like a rubber band: an asset that has recently
#      run ahead of its family tends to drift back toward it, and one that has fallen
#      behind tends to catch up (this snap-back plays out over ~60 days).
#   3. So for each asset we measure how far it has pulled away from its own family
#      (after removing the movement the family shares), then bet on it snapping back:
#      short the ones that ran ahead, buy the ones that lagged.
#   4. We keep total buys ~= total sells (market-balanced) and bet less on jumpier assets.
#   5. Once a bet is on, we leave it alone unless our new target for that asset is a lot
#      different from what we're already holding. This "leave-it-alone" rule stops us from
#      constantly nudging positions on day-to-day noise -- which just burns trading fees and
#      whipsaws us in and out without our actual view having changed.
#   6. When the WHOLE market suddenly gets choppier than usual, we shrink every bet (down to
#      a floor of 20% size), then let it recover as the market calms. Our snap-back edge tends
#      to disappear in turbulent stretches, so we lean back and risk less capital until things
#      settle. (This is the ONLY thing that differs from family_cluster_only. See
#      h1_analysis.ipynb for why: the strategy's steadiness and market volatility are
#      anti-correlated, so dialling back in choppy markets was a cheap win in every time-window
#      we tested -- full-window Score ~137 -> ~139, and it helped the weak stretch too.)
#
# Notes (see research.py / h1_analysis.ipynb): scores stay strong across cluster counts 3-8
# and exposure levels. Still in-sample numbers on 250 days -- expect the real (new-data) Score
# to be lower.
CLUSTER_WINDOW = 120       # days of history used to decide which assets are "family"
N_CLUSTERS = 6             # how many families to split the assets into
REVERT_WINDOW = 60         # days over which the snap-back plays out
VOL_WINDOW = 20            # days used to gauge how jumpy each ASSET is (for per-asset bet sizing)
GROSS_DOLLARS = 1_000_000  # target total $ exposure; eval clips each asset to its $ limit
NO_TRADE_BAND = 0.5        # only re-trade an asset if its target moves >50% of the current size

# --- the market "risk-off" dial (the one thing family_cluster_only doesn't have) ---
MKT_VOL_SHORT = 20         # days to measure the market's *recent* jumpiness
MKT_VOL_LONG = 100         # days to measure the market's *normal* jumpiness (the baseline)
VOL_FLOOR = 0.2            # never shrink the whole book below 20% of full size
VOL_K = 1.0               # how hard we react to choppiness (1 = gentle; higher = more aggressive)

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


def _vol_scale(prcSoFar):
    """How much of full size to hold, based on how choppy the market is *right now* vs its own
    normal. Returns ~1.0 in calm markets and shrinks toward VOL_FLOOR when recent swings are
    bigger than usual. Uses only past data, so it's safe to run live. Here "the market" = the
    average asset's daily move (an equal-weight basket of everything)."""
    mkt = np.diff(np.log(prcSoFar), axis=1).mean(axis=0)   # the average asset's daily move
    if len(mkt) < MKT_VOL_LONG:
        return 1.0                                         # not enough history yet -> full size
    recent = mkt[-MKT_VOL_SHORT:].std()                    # how jumpy the market has been lately
    normal = mkt[-MKT_VOL_LONG:].std()                     # how jumpy it usually is
    if recent <= 0:
        return 1.0
    # calm (recent <= normal) -> ~1.0 ; choppy (recent > normal) -> below 1, clipped at the floor
    return float(np.clip((normal / recent) ** VOL_K, VOL_FLOOR, 1.0))


def _hold_or_trade(target):
    """Leave each position where it is unless the new target is a lot different (see NO_TRADE_BAND).
    Remembers the result in _prevPos so tomorrow can compare against what we actually hold today."""
    global _prevPos
    target = np.asarray(target, dtype=float)
    if _prevPos is None or _prevPos.shape != target.shape:
        _prevPos = target.copy()
    else:
        moved = np.abs(target - _prevPos) > NO_TRADE_BAND * np.maximum(np.abs(_prevPos), 1.0)
        _prevPos = np.where(moved, target, _prevPos)   # only the big-enough changes go through
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

    # 1. split the assets into families using the last CLUSTER_WINDOW days
    labels = _family_labels(dailyRet[:, -CLUSTER_WINDOW:])

    # 2. for each asset, measure how far it pulled away from its own family over REVERT_WINDOW,
    #    after subtracting the movement the whole family shared
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

    # 3. bet on the snap-back: short what ran ahead, buy what lagged; keep buys ~= sells
    signal = -(pullAway - pullAway.mean())

    # 4. bet less on jumpier assets (divide by how volatile each one is)
    vol = np.clip(np.std(dailyRet[:, -VOL_WINDOW:], axis=1), 1e-8, None)
    signal = signal / vol

    # spread the dollar budget across the bets; eval enforces per-asset $ limits
    gross = np.sum(np.abs(signal))
    if gross < 1e-12:
        return _hold_or_trade(np.zeros(nInst))
    dollarPos = signal / gross * GROSS_DOLLARS

    # 6. dial the WHOLE book down when the market is unusually choppy (the only extra step)
    dollarPos = dollarPos * _vol_scale(prcSoFar)

    return _hold_or_trade(dollarPos / prices)
