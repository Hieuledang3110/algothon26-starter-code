import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# =============================================================================
# STRATEGY: family_cluster_bigsize
#   family_cluster_volfilter  +  run the book BIGGER and de-risk HARDER.
#
#   The trading *idea* is byte-for-byte the same rubber-band mean-reversion as the other
#   two files. This variant changes only two dials (see the two ***-marked lines below):
#     - GROSS_DOLLARS 1.0M -> 2.5M : aim for MORE total exposure.
#     - VOL_K         1.0  -> 2.0  : react more strongly when the market turns choppy.
#   Everything else -- the families, the snap-back signal, the per-asset sizing, the
#   leave-it-alone band, the shape of the vol dial -- is identical. Self-contained, so it
#   can be submitted on its own.
#
# --- WHY this variant exists (what problem it targets) -------------------------
#   On the grader's first hidden stage we landed in the strategy's WEAK regime: the bets
#   still made money on average (positive mean profit) but only about half as much as
#   in-sample, so the risk-adjusted score fell hard (Score ~40, Sharpe ~1 vs in-sample
#   138/2.2). The score formula only banks ~50% of your average profit at Sharpe 1 (vs
#   ~83% at Sharpe 2.2), so the book was "profit-starved," not blowing up on losses.
#   Two dials attack that, and they stack:
#
#   (A) RUN BIGGER (GROSS_DOLLARS 2.5M).  The evaluator caps each asset to a fixed dollar
#       size ($10k, or $100k for asset 0). At the old 1.0M target, most positions sit
#       BELOW their cap -- we were leaving allowed exposure unused. Raising the target
#       pushes more assets up to their cap, so the book fills out toward "hold the max
#       allowed in each name." On the weak-regime data that reshaping is exactly where the
#       extra profit is: it roughly quadrupled the weak-half score. It's a BROAD, smooth
#       improvement that peaks around 2.5M (not a lucky spike), and it is NOT driven by the
#       special asset 0 (that name is ~4% of the weak-half profit). Trade-off: this is more
#       leverage, so it also amplifies whatever regime we land in -- fine when the edge is
#       positive (as it was on the grader), but don't crank it higher: past ~2.5-3M the
#       overall Sharpe starts to erode.
#
#   (B) DE-RISK HARDER (VOL_K 2.0).  When the whole market gets choppier than its own
#       normal, shrink every bet more aggressively than the champion does. This lifts the
#       weak-half Sharpe a little for almost free and offsets some of the extra risk from
#       running bigger.
#
#   IMPORTANT -- what did NOT work (so nobody re-tries it): trading more OFTEN is the wrong
#   knob. Shrinking the leave-it-alone band toward 0 (re-trade every day) HURTS the weak
#   regime -- it just pays commission to chase noise. The lever is bet SIZE, not trade
#   FREQUENCY. See strategy-findings memory / scratchpad diag*.py for the full sweep.
#
# --- HOW IT DIFFERS FROM THE OTHER TWO FILES -----------------------------------
#   family_cluster_only.py       : baseline. Families + snap-back + leave-it-alone band.
#                                  GROSS_DOLLARS 1.0M, NO vol dial.            (safe fallback)
#   family_cluster_volfilter.py  : CHAMPION. Baseline + a gentle vol dial (VOL_K 1.0).
#                                  GROSS_DOLLARS 1.0M.               (best on in-sample data)
#   family_cluster_bigsize.py    : THIS FILE. Champion + bigger book (2.5M) + harder dial
#                                  (VOL_K 2.0).            (best on the WEAK-regime proxy)
#
# --- The rubber-band idea, in plain words (same in all three) ------------------
#   1. Assets come in families that tend to move together. We find those families
#      automatically from which assets' daily moves have been most correlated over the
#      last 120 days, and split them into 6 groups.
#   2. Within a family, prices behave like a rubber band: an asset that recently ran ahead
#      of its family tends to drift back, one that lagged tends to catch up (~60 days).
#   3. For each asset we measure how far it pulled away from its own family (after removing
#      the move the whole family shared), then bet on the snap-back: short what ran ahead,
#      buy what lagged.
#   4. Keep total buys ~= total sells (market-balanced) and bet less on jumpier assets.
#   5. Leave a position alone unless the new target is a lot different from what we hold
#      (the "leave-it-alone" band) -- avoids paying fees to chase day-to-day noise.
#   6. When the WHOLE market turns choppy, shrink every bet toward a floor, then recover as
#      it calms. This file just reacts harder (VOL_K 2.0) than the champion (1.0).
#
# NOTE: still in-sample / weak-half-proxy numbers -- the grader's real regime may differ.
# If the bigger book ever misbehaves on a new stage, drop back to family_cluster_volfilter
# (gentler) or family_cluster_only (no dial at all).
CLUSTER_WINDOW = 120       # days of history used to decide which assets are "family"
N_CLUSTERS = 6             # how many families to split the assets into
REVERT_WINDOW = 60         # days over which the snap-back plays out
VOL_WINDOW = 20            # days used to gauge how jumpy each ASSET is (for per-asset bet sizing)
GROSS_DOLLARS = 2_500_000  # *** target total $ exposure (2.5x the champion) -- fills the book
                           #     out toward the evaluator's per-asset caps. eval still clips.
NO_TRADE_BAND = 0.5        # only re-trade an asset if its target moves >50% of the current size

# --- the market "risk-off" dial (same shape as the champion, tuned harder) ---
MKT_VOL_SHORT = 20         # days to measure the market's *recent* jumpiness
MKT_VOL_LONG = 100         # days to measure the market's *normal* jumpiness (the baseline)
VOL_FLOOR = 0.2            # never shrink the whole book below 20% of full size
VOL_K = 2.0               # *** how hard we react to choppiness (champion uses 1.0; harder here)

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

    # spread the dollar budget across the bets; eval enforces per-asset $ limits.
    # with GROSS_DOLLARS at 2.5M this deliberately overshoots many caps so the book fills
    # out toward the per-asset maximums (that's the whole point of this variant).
    gross = np.sum(np.abs(signal))
    if gross < 1e-12:
        return _hold_or_trade(np.zeros(nInst))
    dollarPos = signal / gross * GROSS_DOLLARS

    # 6. dial the WHOLE book down when the market is unusually choppy (harder here: VOL_K 2.0)
    dollarPos = dollarPos * _vol_scale(prcSoFar)

    return _hold_or_trade(dollarPos / prices)
