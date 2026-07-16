#!/usr/bin/env python
"""Algothon 2026 research harness (NOT submitted).

Two jobs:
  1. backtest(posFunc)  -> reproduce eval.py's Score/Sharpe/turnover for ANY position
     function, so we can compare strategies fast and match the official grader exactly.
  2. featureIC(featFunc) -> walk-forward Information Coefficient for a candidate feature,
     so we can screen signals BEFORE wiring them into a model.

Usage:
    python research.py                      # backtest the active strategy file
    python research.py --days 250           # choose test window
    python research.py --ic                 # also run a few example feature ICs

Nothing here is part of the submission; keep strategy logic in the family_cluster_*.py files.
"""
import os
import sys

# Ensure project root and strategy directories are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
if os.path.join(root_dir, 'strategy') not in sys.path:
    sys.path.append(os.path.join(root_dir, 'strategy'))

# If executed from within the tools directory, change CWD to project root
if os.path.basename(os.getcwd()) == 'tools':
    os.chdir('..')

import argparse
import numpy as np
import pandas as pd

# --- mechanics copied verbatim from eval.py (source of truth) ---
PRICES_FILE = "./prices.txt"
NUM_TEST_DAYS = 250
SCORE_PARAM = 1.0
DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002
DEFAULT_DLR_POS_LIMIT = 10_000
INST0_DLR_POS_LIMIT = 100_000


def loadPrices(fn=PRICES_FILE):
    """Load prices -> array shape (nInst, nt): one instrument per row, columns = days."""
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def score(mu, sigma, param=SCORE_PARAM):
    """Final score from daily-PnL mean & std (identical to eval.py)."""
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


def backtest(posFunc, prcHist, numTestDays=NUM_TEST_DAYS, verbose=False,
             return_series=False, return_attribution=False):
    """Reimplementation of eval.py::calcPL for an arbitrary position function.

    Kept line-for-line faithful (incl. the one-day-lagged commission accounting and
    integer/limit clipping) so the Score printed here equals the official grader's.
    Returns a metrics dict.
      - return_series=True     -> add "pnl": daily total-PnL array (for an equity curve).
      - return_attribution=True -> add "pnlByInst": shape (numTestDays, nInst) of each
        instrument's daily profit. These rows sum exactly to "pnl": one instrument's
        daily profit = (shares held into the day) * (today's price move) - its commission.
    """
    nInst, nt = prcHist.shape

    commRate = np.full(nInst, DEFAULT_COMM_RATE)
    commRate[0] = INST0_COMM_RATE
    dlrPosLimit = np.full(nInst, DEFAULT_DLR_POS_LIMIT)
    dlrPosLimit[0] = INST0_DLR_POS_LIMIT

    cash = 0.0
    curPos = np.zeros(nInst)
    totDVolume = 0.0
    value = 0.0
    comm = 0.0
    commVec = np.zeros(nInst)   # per-instrument commission, charged one day late (mirrors comm)
    prevPrices = None           # yesterday's prices, for the mark-to-market move
    todayPLL = []
    pnlByInst = []
    ret = 0.0

    startDay = nt - numTestDays
    for t in range(startDay, nt + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        if t < nt:
            newPosOrig = posFunc(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)

        # split today's total PnL back to each instrument, BEFORE we move positions/prices:
        # position held over the day * its price move, minus the commission being charged now.
        if t > startDay:
            pnlByInst.append(curPos * (curPrices - prevPrices) - commVec)

        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        totDVolume += np.sum(dvolumes)
        commVec = dvolumes * commRate
        comm = np.sum(commVec)

        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        prevPrices = curPrices

        if totDVolume > 0:
            ret = value / totDVolume

        if t > startDay:
            todayPLL.append(todayPL)
            if verbose:
                print(f"Day {t} value: {value:.2f} todayPL: ${todayPL:.2f} "
                      f"$-traded: {totDVolume:.0f} return: {ret:.5f}")

    pll = np.array(todayPLL)
    mu, sigma = float(np.mean(pll)), float(np.std(pll))
    sharpe = np.sqrt(250) * mu / sigma if sigma > 0 else 0.0
    # avg daily turnover in dollars over the scored window
    avgDailyTurnover = totDVolume / numTestDays
    result = {
        "meanPL": mu,
        "stdPL": sigma,
        "annSharpe": sharpe,
        "return": ret,
        "totDVolume": totDVolume,
        "avgDailyTurnover": avgDailyTurnover,
        "score": score(mu, sigma),
    }
    if return_series:
        result["pnl"] = pll
    if return_attribution:
        result["pnlByInst"] = np.array(pnlByInst)
    return result


def printReport(name, m):
    print("=====", name, "=====")
    print(f"mean(PL):       {m['meanPL']:.2f}")
    print(f"StdDev(PL):     {m['stdPL']:.2f}")
    print(f"annSharpe(PL):  {m['annSharpe']:.2f}")
    print(f"return:         {m['return']:.5f}")
    print(f"totDvolume:     {m['totDVolume']:.0f}")
    print(f"avgDayTurnover: {m['avgDailyTurnover']:.0f}")
    print(f"Score:          {m['score']:.2f}")


def featureIC(featFunc, prcHist, numTestDays=NUM_TEST_DAYS):
    """Walk-forward Information Coefficient of a feature vs next-day return.

    featFunc(prcSoFar) -> vector length nInst (higher = expected to outperform).
    IC = per-day Spearman rank-corr(feature_t, fwdRet_{t->t+1}), averaged over the
    test window. Reports mean IC and its t-stat (mean / SE). |mean IC| > ~0.03 with
    t-stat > ~2 is a signal worth keeping; sign tells you momentum(+) vs reversion(-).
    """
    nInst, nt = prcHist.shape
    startDay = nt - numTestDays
    ics = []
    for t in range(startDay, nt):   # for each day t in the test window
        prcSoFar = prcHist[:, :t]
        if prcSoFar.shape[1] < 2:
            continue
        feat = np.asarray(featFunc(prcSoFar), dtype=float)  # feature for day t, length nInst
        fwdRet = np.log(prcHist[:, t] / prcHist[:, t - 1])  # next-day return from t->t+1, length nInst
        s = pd.DataFrame({"f": feat, "r": fwdRet}).dropna()
        if s["f"].nunique() < 2 or s["r"].nunique() < 2:
            continue
        ics.append(s["f"].corr(s["r"], method="spearman"))  # Spearman for day t, length nInst
    ics = np.array(ics)                                    
    meanIC = float(np.mean(ics)) if len(ics) else 0.0       # Average IC of instruments over the test window
    seIC = float(np.std(ics) / np.sqrt(len(ics))) if len(ics) > 1 else np.nan # Standard error of the mean IC
    tstat = meanIC / seIC if seIC and seIC > 0 else 0.0
    return {"meanIC": meanIC, "tstat": tstat, "nDays": len(ics)}


# --- a couple of example features to demonstrate featureIC ---
def _feat_reversal_1d(prcSoFar):
    """Yesterday's return. Negative IC => cross-sectional mean reversion."""
    r = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    return r - np.mean(r)  # market-relative


def _feat_momentum_20d(prcSoFar):
    """20-day return, market-relative. Positive IC => momentum."""
    k = min(20, prcSoFar.shape[1] - 1)
    r = np.log(prcSoFar[:, -1] / prcSoFar[:, -1 - k])
    return r - np.mean(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=NUM_TEST_DAYS)
    ap.add_argument("--ic", action="store_true", help="run example feature IC screens")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prc = loadPrices()
    print(f"Loaded {prc.shape[0]} instruments for {prc.shape[1]} days\n")

    from strategy.family_cluster_ownrevert import getMyPosition   # active strategy -- switch to compare
    m = backtest(getMyPosition, prc, args.days, verbose=args.verbose)
    printReport("family_cluster_ownrevert.getMyPosition", m)

    if args.ic:
        print("\n----- feature IC (walk-forward) -----")
        for nm, fn in [("reversal_1d", _feat_reversal_1d),
                       ("momentum_20d", _feat_momentum_20d)]:
            r = featureIC(fn, prc, args.days)
            print(f"{nm:14s} meanIC={r['meanIC']:+.4f} t={r['tstat']:+.2f} "
                  f"(n={r['nDays']})")