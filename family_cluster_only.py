import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# =============================================================================
# STRATEGY: family_cluster_only  (baseline)
#   Mean-reversion within auto-detected families + a "leave-it-alone" no-trade band.
#   Sibling file family_cluster_volfilter.py is this exact strategy PLUS one extra rule
#   (shrink every bet when the whole market turns choppy). Each file is self-contained
#   so either one can be submitted on its own.
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
#
# Notes (see PLAN.md / research.py): scores stay strong across cluster counts 3-8 and across
# exposure levels, which is why we trust this over the single-window version. These are still
# in-sample numbers on 250 days -- expect the real (new-data) Score to be lower.
CLUSTER_WINDOW = 120       # days of history used to decide which assets are "family"
N_CLUSTERS = 6             # how many families to split the assets into
REVERT_WINDOW = 60         # days over which the snap-back plays out
VOL_WINDOW = 20            # days used to gauge how jumpy each asset is (for bet sizing)
GROSS_DOLLARS = 1_000_000  # target total $ exposure; eval clips each asset to its $ limit
NO_TRADE_BAND = 0.5        # only re-trade an asset if its target moves >50% of the current size

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

    return _hold_or_trade(dollarPos / prices)
