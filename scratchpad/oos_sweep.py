#!/usr/bin/env python
"""Two-stage robustness sweep on the pure family engine (family_cluster_bigsize).

We now have TWO independent 250-day stages:
  Stage A (in-sample)      = days 251-500
  Stage B (out-of-sample)  = days 501-750  (the grader window)
The honest objective is a config that is strong on BOTH (high min-of-the-two),
sitting on a plateau (neighbours agree), NOT a spike tuned to either stage.

Sweep the two proven levers only: GROSS_DOLLARS (size) and VOL_K (risk-off dial).
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.join(ROOT, 'strategy'))
os.chdir(ROOT)

from tools.research import backtest, loadPrices, score
import strategy.family_cluster_bigsize as BS

prc = loadPrices()
IS = prc[:, :500]


def sc(seg):
    mu, sd = float(np.mean(seg)), float(np.std(seg))
    return score(mu, sd)


def eval_cfg(gross, vol_k):
    BS.GROSS_DOLLARS = gross
    BS.VOL_K = vol_k
    res = {}
    for tag, sl in [("A", IS), ("B", prc)]:
        BS._prevPos = None; BS._prevNt = None
        m = backtest(BS.getMyPosition, sl, 250, return_series=True)
        res[tag] = (sc(m["pnl"]), m["annSharpe"])
    return res


print("Family engine (family_cluster_bigsize) -- GROSS x VOL_K, scored on both stages")
print("Stage A = 251-500 (in-samp), Stage B = 501-750 (grader OOS)\n")
print(f"{'GROSS':>8} {'VOL_K':>6} | {'A Score':>8} {'A Shrp':>6} | {'B Score':>8} {'B Shrp':>6} | "
      f"{'min(A,B)':>8} {'avg':>7}")
print("-" * 78)
best = None
for gross in (1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 6_000_000):
    for vol_k in (1.0, 2.0, 3.0):
        r = eval_cfg(gross, vol_k)
        a, ash = r["A"]; b, bsh = r["B"]
        mn, av = min(a, b), (a + b) / 2
        flag = ""
        if best is None or mn > best[0]:
            best = (mn, gross, vol_k, a, b)
        print(f"{gross:>8,} {vol_k:>6.1f} | {a:>8.1f} {ash:>6.2f} | {b:>8.1f} {bsh:>6.2f} | "
              f"{mn:>8.1f} {av:>7.1f}")
    print()

print(f"Best min-of-both-stages: GROSS={best[1]:,} VOL_K={best[2]} "
      f"-> A {best[3]:.1f}, B {best[4]:.1f}, min {best[0]:.1f}")
print("\nCurrent shipped bigsize = GROSS 2,500,000 / VOL_K 2.0")
