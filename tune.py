#!/usr/bin/env python
"""tune.py -- self-service parameter explorer for the family_cluster_* strategies. NOT submitted.

WHAT IT'S FOR
    Sweep any strategy parameters yourself, without editing the strategy files. You pick which
    knobs to HOLD CONSTANT and which to INVESTIGATE (a list of values), and this runs the exact
    eval.py backtest for every combination and prints a ranked table -- with the full-window
    Score AND the H1/H2 weak/strong split, which is how this project judges every change.

    Scoring/trading mechanics come straight from research.backtest (verified to match eval.py),
    so the numbers here equal the official grader's.

HOW TO USE
    1. Edit the CONFIG block below:
         - a single value   -> HOLD THAT KNOB CONSTANT           e.g. "VOL_K": 2.0
         - a list/range     -> INVESTIGATE (sweep every value)   e.g. "GROSS_DOLLARS": [2_000_000, 2_500_000, 3_000_000]
         - None (or omit)   -> leave the strategy file's own default untouched
       Helpers for building lists:  irange(50, 70, 5) -> [50,55,60,65,70] ;  frange(0.1, 0.3, 0.05)
    2. Run it:
         python tune.py                       # sweep using the CONFIG block
         python tune.py --strategy family_cluster_volfilter
         python tune.py --sort h1 --top 10    # rank by weak-half Score, show best 10
         python tune.py --csv sweep.csv       # also save the full table to a CSV you can chart
         python tune.py --perturb             # OVERFIT TEST: nudge each constant knob +/- and
                                              #   see if the result cliffs (a fragile peak = overfit)

READING THE OUTPUT
    Every row shows: the swept knob value(s), then Score & Sharpe for the full window, H1 (the
    weak-regime proxy -- the first 125 test days) and H2 (the strong second 125). Judge changes
    by how they behave in H1, not just the full window, and REJECT any peak whose neighbours
    collapse (that's what --perturb is for). See CLAUDE.md "Our strategy" for the full playbook.
"""
import argparse
import importlib
import itertools

import numpy as np

from research import loadPrices, score, backtest

# ============================ EDIT THIS BLOCK ============================
STRATEGY  = "family_cluster_bigsize"   # which strategy file to tune (any family_cluster_* file)
TEST_DAYS = 250                        # scored window (eval.py uses 250)
SORT_BY   = "full"                     # rank rows by: "full" | "h1" | "h2" | "sharpe"
TOP       = 0                          # 0 = show every row; else show only the best N
SAVE_CSV  = ""                         # e.g. "sweep.csv" to also write a CSV; "" = don't

# Each knob: a single value = HOLD CONSTANT ; a list = INVESTIGATE (sweep) ; None = file default.
PARAMS = {
    # --- the mean-reversion core (present in ALL strategy files) ---
    "N_CLUSTERS":     6,               #   how many families to split the assets into
    "CLUSTER_WINDOW": 120,             #   days of history used to decide the families
    "REVERT_WINDOW":  60,              #   days over which the snap-back plays out  (known knife-edge!)
    "VOL_WINDOW":     20,              #   days used to gauge each asset's jumpiness
    "NO_TRADE_BAND":  0.5,             #   leave-it-alone band (0 = re-trade daily)
    "GROSS_DOLLARS":  [2_000_000, 2_500_000, 3_000_000],   # target total $ exposure  <-- example sweep

    # --- the market "risk-off" dial (family_cluster_volfilter / _bigsize only) ---
    "VOL_K":          2.0,             #   how hard to shrink the book when the market turns choppy
    "VOL_FLOOR":      0.2,             #   never shrink below this fraction of full size
    "MKT_VOL_SHORT":  20,              #   window for the market's *recent* jumpiness
    "MKT_VOL_LONG":   100,             #   window for the market's *normal* jumpiness
}
# ========================================================================

# every knob this tool knows how to set (union across the strategy files)
KNOWN = ["N_CLUSTERS", "CLUSTER_WINDOW", "REVERT_WINDOW", "VOL_WINDOW", "NO_TRADE_BAND",
         "GROSS_DOLLARS", "VOL_K", "VOL_FLOOR", "MKT_VOL_SHORT", "MKT_VOL_LONG"]


# ---------- little helpers you can use when building sweep lists ----------
def irange(start, stop, step=1):
    """Inclusive integer range: irange(50, 70, 5) -> [50, 55, 60, 65, 70]."""
    return list(range(start, stop + 1, step))


def frange(start, stop, step):
    """Inclusive float range: frange(0.1, 0.3, 0.05) -> [0.1, 0.15, 0.2, 0.25, 0.3]."""
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


def _fmt(v):
    """Compact value for the table (thousands separators for big ints)."""
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int) and abs(v) >= 10_000:
        return f"{v:,}"
    return str(v)


# ------------------------------ the engine ------------------------------
def _split_scores(pnl, half):
    """Score the full window and each half exactly like eval.py does on daily-PnL mean/std."""
    def sc(seg):
        mu, sd = float(np.mean(seg)), float(np.std(seg))
        sr = np.sqrt(250) * mu / sd if sd > 0 else 0.0
        return score(mu, sd), sr
    return sc(pnl), sc(pnl[:half]), sc(pnl[half:])


def run_one(mod, defaults, overrides, prc, test_days):
    """Reset the module to its defaults, apply this combo's overrides, backtest, return metrics."""
    for name, val in defaults.items():
        setattr(mod, name, val)                 # start every run from a clean slate
    for name, val in overrides.items():
        if hasattr(mod, name):
            setattr(mod, name, val)
    mod._prevPos = None                          # clear the strategy's held-position memory
    mod._prevNt = None
    m = backtest(mod.getMyPosition, prc, test_days, return_series=True)
    (fs, fsh), (h1, h1sh), (h2, h2sh) = _split_scores(m["pnl"], test_days // 2)
    return {"full": fs, "full_sh": fsh, "h1": h1, "h1_sh": h1sh, "h2": h2, "h2_sh": h2sh,
            "turn": m["avgDailyTurnover"], "mean": m["meanPL"], "sharpe": m["annSharpe"]}


def _print_table(rows, swept_names, sort_by, top):
    """rows: list of (overrides_dict, metrics_dict). Prints a ranked, aligned table."""
    key = {"full": "full", "h1": "h1", "h2": "h2", "sharpe": "sharpe"}.get(sort_by, "full")
    rows = sorted(rows, key=lambda r: r[1][key], reverse=True)
    if top and top > 0:
        rows = rows[:top]

    swept_cols = list(swept_names)
    widths = {c: max(len(c), max((len(_fmt(ov[c])) for ov, _ in rows), default=1)) for c in swept_cols}
    header = "  ".join(f"{c:>{widths[c]}}" for c in swept_cols)
    if header:
        header += "  | "
    header += f"{'full':>6} {'(Sh)':>6} | {'H1':>7} {'(Sh)':>6} | {'H2':>7} {'(Sh)':>6} | {'turn/day':>12}"
    print(header)
    print("-" * len(header))
    for ov, m in rows:
        line = "  ".join(f"{_fmt(ov[c]):>{widths[c]}}" for c in swept_cols)
        if swept_cols:
            line += "  | "
        line += (f"{m['full']:6.1f} {m['full_sh']:6.2f} | {m['h1']:7.1f} {m['h1_sh']:6.2f} | "
                 f"{m['h2']:7.1f} {m['h2_sh']:6.2f} | ${m['turn']:11,.0f}")
        print(line)
    return rows


def _save_csv(path, rows, swept_names):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(swept_names) + ["full", "full_sharpe", "H1", "H1_sharpe",
                                        "H2", "H2_sharpe", "turnover_day", "mean", "sharpe"])
        for ov, m in rows:
            w.writerow([ov[c] for c in swept_names] +
                       [round(m["full"], 3), round(m["full_sh"], 3), round(m["h1"], 3),
                        round(m["h1_sh"], 3), round(m["h2"], 3), round(m["h2_sh"], 3),
                        round(m["turn"], 0), round(m["mean"], 2), round(m["sharpe"], 3)])
    print(f"\nSaved {len(rows)} rows -> {path}")


def _neighbors(v, frac=0.15):
    """Perturbation neighbours of a value: for the overfit test. Ints step by >=1, floats by frac."""
    if isinstance(v, float) and not float(v).is_integer():
        return sorted({round(v * (1 - frac), 6), v, round(v * (1 + frac), 6)})
    iv = int(v)
    step = max(1, int(round(abs(iv) * frac)))
    return sorted({iv - step, iv, iv + step})


def main():
    ap = argparse.ArgumentParser(description="Self-service parameter explorer (edit CONFIG in the file).")
    ap.add_argument("--strategy", default=STRATEGY, help="strategy module to tune")
    ap.add_argument("--days", type=int, default=TEST_DAYS)
    ap.add_argument("--sort", default=SORT_BY, choices=["full", "h1", "h2", "sharpe"])
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--csv", default=SAVE_CSV)
    ap.add_argument("--perturb", action="store_true",
                    help="overfit test: hold the CONFIG's constants as the centre and nudge each +/-")
    args = ap.parse_args()

    prc = loadPrices()
    mod = importlib.import_module(args.strategy)
    defaults = {n: getattr(mod, n) for n in KNOWN if hasattr(mod, n)}  # snapshot the file's own values

    # warn about config keys the chosen strategy doesn't have (e.g. VOL_K on family_cluster_only)
    missing = [k for k, v in PARAMS.items() if v is not None and not hasattr(mod, k)]
    if missing:
        print(f"NOTE: {args.strategy} has no {missing} -- those entries are ignored.\n")

    print(f"Loaded {prc.shape[0]} instruments x {prc.shape[1]} days | strategy={args.strategy} | "
          f"test_days={args.days}")

    # split the CONFIG into fixed knobs and swept knobs
    fixed = {k: v for k, v in PARAMS.items()
             if v is not None and not isinstance(v, (list, tuple)) and hasattr(mod, k)}
    swept = {k: list(v) for k, v in PARAMS.items()
             if isinstance(v, (list, tuple)) and hasattr(mod, k)}

    # ---------- OVERFIT TEST MODE ----------
    if args.perturb:
        centre = {**{k: getattr(mod, k) for k in defaults}, **fixed}
        print("\nOVERFIT TEST -- nudging each knob one at a time around your constants "
              "(watch for a neighbour that cliffs):\n")
        base = run_one(mod, defaults, centre, prc, args.days)
        print(f"centre config: full {base['full']:.1f} (Sh {base['full_sh']:.2f}) | "
              f"H1 {base['h1']:.1f} | H2 {base['h2']:.1f}\n")
        for name in KNOWN:
            if name not in centre:
                continue
            vals = _neighbors(centre[name])
            if len(vals) < 2:
                continue
            print(f"  {name} (centre {_fmt(centre[name])}):")
            for val in vals:
                ov = {**centre, name: val}
                m = run_one(mod, defaults, ov, prc, args.days)
                flag = "  <-- centre" if val == centre[name] else ""
                print(f"      {_fmt(val):>12}  full {m['full']:6.1f} (Sh {m['full_sh']:4.2f}) | "
                      f"H1 {m['h1']:6.1f} | H2 {m['h2']:6.1f}{flag}")
            print()
        return

    # ---------- SWEEP MODE ----------
    n_runs = 1
    for vals in swept.values():
        n_runs *= len(vals)
    if fixed:
        print("holding constant: " + ", ".join(f"{k}={_fmt(v)}" for k, v in fixed.items()))
    if swept:
        print("investigating:    " + ", ".join(f"{k}={[_fmt(x) for x in v]}" for k, v in swept.items()))
    print(f"total runs: {n_runs}\n")
    if n_runs > 500:
        print(f"WARNING: {n_runs} combinations -- this may take a while. Ctrl-C to abort.\n")

    names = list(swept.keys())
    rows = []
    for combo in itertools.product(*swept.values()) if swept else [()]:
        overrides = {**fixed, **dict(zip(names, combo))}
        m = run_one(mod, defaults, overrides, prc, args.days)
        rows.append((overrides, m))

    ranked = _print_table(rows, names, args.sort, args.top)
    if ranked:
        best_ov, best = ranked[0][0], ranked[0][1]
        bestdesc = ", ".join(f"{k}={_fmt(best_ov[k])}" for k in names) or "(single config)"
        print(f"\nbest by {args.sort}: {bestdesc}  ->  full {best['full']:.1f}, "
              f"H1 {best['h1']:.1f}, H2 {best['h2']:.1f}, Sharpe {best['full_sh']:.2f}")
    if args.csv:
        _save_csv(args.csv, rows, names)


if __name__ == "__main__":
    main()
