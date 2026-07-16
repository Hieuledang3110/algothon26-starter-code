#!/usr/bin/env python
"""OOS post-mortem: days 501-750 are now available (the grader's hidden window).

Compares every main strategy in-sample (days 251-500, what we tuned on) vs
out-of-sample (days 501-750, the real grader window), with H1/H2 splits, and
confirms the numbers reproduce the leaderboard (bigsize +64, ownrevert -26).
"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.join(ROOT, 'strategy'))
os.chdir(ROOT)

from tools.research import backtest, loadPrices, score

prc = loadPrices()                       # (nInst, nt) = (51, 750)
nInst, nt = prc.shape
print(f"Loaded {nInst} instruments x {nt} days\n")

IS_END = 500                             # first-500-day file -> scores days 251-500
# OOS uses the full 750 -> scores days 501-750 (== grader hidden window)


def sc(seg):
    mu, sd = float(np.mean(seg)), float(np.std(seg))
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0.0
    return score(mu, sd), sr


def run(mod_name, fn_name="getMyPosition"):
    """Return IS and OOS metric dicts for a strategy module."""
    mod = __import__(f"strategy.{mod_name}", fromlist=[fn_name])
    out = {}
    for tag, price_slice in [("IS", prc[:, :IS_END]), ("OOS", prc)]:
        mod._prevPos = None
        mod._prevNt = None
        fn = getattr(mod, fn_name)
        m = backtest(fn, price_slice, 250, return_series=True, return_attribution=True)
        half = 125
        (fs, fsh) = sc(m["pnl"])
        (h1, h1sh) = sc(m["pnl"][:half])
        (h2, h2sh) = sc(m["pnl"][half:])
        out[tag] = {
            "score": fs, "sharpe": m["annSharpe"], "mean": m["meanPL"], "std": m["stdPL"],
            "turn": m["avgDailyTurnover"], "h1": h1, "h1_sh": h1sh, "h2": h2, "h2_sh": h2sh,
            "pnl": m["pnl"], "pnlByInst": m["pnlByInst"],
        }
    return out


STRATS = [
    ("family_cluster_bigsize",   "CHAMPION (submitted): family, bigger book + hard vol dial"),
    ("family_cluster_volfilter", "former champ: family + band + gentle vol dial"),
    ("family_cluster_only",      "baseline: family sleeve only, no vol dial"),
    ("family_cluster_ownrevert", "FAILED OOS: family + own-price sleeve (ALGO 10x)"),
    ("family_cluster_algo_custom","ALGO carve-out: family(1-50) + dedicated ALGO z-score"),
]

results = {}
print(f"{'strategy':<28} | {'window':<4} | {'Score':>8} {'Shrp':>5} {'mean':>7} {'std':>7} | "
      f"{'H1':>7} {'H2':>7} | {'turn/day':>9}")
print("-" * 108)
for name, desc in STRATS:
    r = run(name)
    results[name] = r
    for tag in ("IS", "OOS"):
        d = r[tag]
        print(f"{name:<28} | {tag:<4} | {d['score']:>8.1f} {d['sharpe']:>5.2f} "
              f"{d['mean']:>7.1f} {d['std']:>7.0f} | {d['h1']:>7.1f} {d['h2']:>7.1f} | "
              f"{d['turn']:>9,.0f}")
    print()

# ---- grader reconciliation ----
print("=" * 70)
print("GRADER RECONCILIATION (leaderboard vs local OOS days 501-750)")
print("  bigsize   leaderboard: Score 64.13, mean 107.99, std 1412.11, 11101 trades")
print("  ownrevert leaderboard: Score -26.11, mean -26.11, std 1833.57, 11871 trades")
for name in ("family_cluster_bigsize", "family_cluster_ownrevert"):
    d = results[name]["OOS"]
    print(f"  {name:<26} local OOS: Score {d['score']:.2f}, mean {d['mean']:.2f}, std {d['std']:.2f}")

# ---- persist for the notebook ----
dump = {}
for name, r in results.items():
    dump[name] = {}
    for tag in ("IS", "OOS"):
        d = r[tag]
        dump[name][tag] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in d.items()}
with open("scratchpad/oos_results.json", "w") as f:
    json.dump(dump, f)
print("\nsaved scratchpad/oos_results.json")
