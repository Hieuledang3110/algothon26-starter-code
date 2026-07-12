# Algothon 2026 Starter Code

Starter code for the Susquehanna x UNSW FinTech Society Algothon 2026 - the seventh year of Australia's first student-led algorithmic trading hackathon.

Full rules, scoring, schedule, and submission details live on the **[Algothon 2026 Wiki](https://wiki.algothon.au/)** - this README only covers what's in this repo and how to run it. If anything here ever seems to disagree with the wiki, the wiki is correct.

> **Where things live:** `CLAUDE.md` holds the *explanations* - what each strategy is, the scoring mechanics, and the research findings. **This README is the *command reference*** - how to run each file and what it does. Read `CLAUDE.md` for *why*; read here for *how to run*.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows (this repo's platform)
source .venv/bin/activate   # macOS/Linux
pip install -r requirements-dev.txt
```

## Repo file map

There's **one file per strategy** (a small library) - each is a self-contained, independently-submittable `getMyPosition`. There is no `teamName.py`; at submission time you copy your chosen strategy file to `<YourTeamName>.py`. See `CLAUDE.md` for what each strategy actually does.

| File | Type | One-liner (details in `CLAUDE.md`) |
| :--- | :--- | :--- |
| `family_cluster_volfilter.py` | strategy | **Champion.** Family mean-reversion + no-trade band + vol dial. |
| `family_cluster_bigsize.py` | strategy | Weak-regime challenger: champion + bigger book + harder dial. |
| `family_cluster_only.py` | strategy | Baseline / fallback: champion minus the vol dial. |
| `eval.py` | helper (run) | Official backtest/scorer. |
| `research.py` | helper (run) | Backtest any function + feature-IC screen. |
| `tune.py` | helper (run) | Self-service parameter sweep + overfit test. |
| `helper.ipynb`, `h1_analysis.ipynb`, `clustering_analysis.ipynb` | notebook | Analysis dashboards. |
| `prices.txt` | data | Current stage's prices. |
| `requirements-dev.txt` | env | Matches the grading sandbox. **Never submit.** |

---

# Commands

## `eval.py` — official backtest

Scores the last 250 days of `prices.txt` for the **active** strategy.

```bash
python eval.py
```

- **Switch strategy:** edit the import near line 10, e.g. `from family_cluster_bigsize import getMyPosition as getPosition`. Don't change anything else in `eval.py`.
- **Output:** `mean(PL)`, `return`, `StdDev(PL)`, `annSharpe(PL)`, `totDvolume`, `Score`.

## `research.py` — research harness (NOT submitted)

Reproduces `eval.py`'s numbers for any position function, plus a walk-forward signal screen.

```bash
python research.py                 # backtest the active strategy (import near line 200)
python research.py --days 250      # choose the scored window
python research.py --ic            # also print example feature-IC screens
python research.py --verbose       # print per-day PnL lines
```

- In a REPL/notebook: `from research import loadPrices, backtest; backtest(fn, loadPrices())` scores any `fn` **without editing files** and returns a metrics dict (`score`, `annSharpe`, `avgDailyTurnover`, ...). Pass `return_series=True` for the daily-PnL array or `return_attribution=True` for per-instrument PnL.
- `featureIC(featFunc, prc)` returns `meanIC` / `tstat` for a candidate signal (sign = momentum(+) vs reversion(-)).

## `tune.py` — parameter explorer (NOT submitted)

Sweep strategy parameters yourself: each knob is either a **single value (hold constant)** or a **list (investigate)**. Prints a ranked table with the full-window Score **and** the H1/H2 (weak/strong-half) split, using the same mechanics as `eval.py`.

**1. Edit the CONFIG block** at the top of `tune.py`:

```python
PARAMS = {
    "VOL_K":         2.0,                                  # single value  -> hold constant
    "GROSS_DOLLARS": [2_000_000, 2_500_000, 3_000_000],   # a list        -> investigate (sweep)
    "REVERT_WINDOW": None,                                 # None / omit   -> leave the file default
    ...
}
```
List-building helpers (already imported in the file): `irange(50, 70, 5)` → `[50,55,60,65,70]`; `frange(0.1, 0.3, 0.05)` → `[0.1,0.15,0.2,0.25,0.3]`. Put lists on two knobs to sweep the grid of both.

**2. Run it:**

```bash
python tune.py                                   # sweep using the CONFIG block
python tune.py --strategy family_cluster_volfilter   # tune a different strategy file
python tune.py --days 250                         # scored window
python tune.py --sort h1 --top 10                 # rank by weak-half Score, show best 10
python tune.py --sort full|h1|h2|sharpe           # ranking metric
python tune.py --csv sweep.csv                    # also save the full table to CSV
python tune.py --perturb                          # OVERFIT TEST: nudge each constant +/- and
                                                  #   watch for a neighbour that cliffs
```

- **Reading the table:** each row shows the swept value(s), then Score & Sharpe for `full`, `H1` (weak first 125 days) and `H2` (strong last 125), plus turnover/day. Judge changes by **H1**, and reject any peak whose neighbours collapse (that's what `--perturb` is for).
- Knobs a strategy file doesn't have (e.g. `VOL_K` on `family_cluster_only`) are skipped with a note.
- CLI flags override the CONFIG block for a one-off run.

## Notebooks

```bash
jupyter notebook          # then open any of the .ipynb files
```

- `helper.ipynb` — equity curve + drawdown, daily-profit profile, today's bets, per-instrument profit attribution, signal lab.
- `h1_analysis.ipynb` — regime diagnosis (why the weak half is weak) and the vol-dial derivation.
- `clustering_analysis.ipynb` — family/cluster exploration.

Each notebook aliases a strategy as `teamName`; re-point that one import to analyse a different file.

---

## Submitting

When you're ready: copy your chosen strategy file (e.g. `family_cluster_bigsize.py`) to `<YourTeamName>.py` (matching your registered team name) and zip **only** that file at the archive root. Do **not** include `eval.py`, `prices.txt`, `research.py`, `tune.py`, or `requirements-dev.txt`. Add a `requirements.txt` **only** if you used packages beyond the accepted set (numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib are pre-accepted - don't redeclare them). See the [Submission Guide](https://wiki.algothon.au/submission/) and submit through the [live leaderboard](https://www.algothon.au/leaderboard).

## Questions

Post in the questions forum on our Discord - moderators are there to help.
