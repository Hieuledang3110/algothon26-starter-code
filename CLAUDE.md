# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Starter code for the **Algothon 2026** — the Susquehanna x UNSW FinTech Society
algorithmic-trading hackathon (7th year). You build a systematic trading strategy
by implementing a single function, `getMyPosition(prcSoFar)`, which is backtested
against historical price data and scored on a risk-adjusted PnL metric.

Full rules, scoring, schedule, and submission details live on the
**[Algothon 2026 Wiki](https://wiki.algothon.au/)**. If anything here disagrees
with the wiki, **the wiki wins**. Other key links:
- Submission Guide: https://wiki.algothon.au/submission/
- Live leaderboard / submission portal: https://www.algothon.au/leaderboard

## Repo layout

We keep **one file per strategy** (a small library), so we can revisit/compare and always fall
back. Each strategy file is a self-contained, independently-submittable `getMyPosition`. There is
**no `teamName.py`** anymore — at submission time, copy the chosen strategy file to `<YourTeamName>.py`.

| File | Purpose |
| :--- | :--- |
| `family_cluster_volfilter.py` | **CHAMPION — the current submission file.** Family mean-reversion + no-trade band + a market-volatility "risk-off" dial. In-sample Score **138.63**, Sharpe 2.21. |
| `family_cluster_bigsize.py` | **Weak-regime challenger.** The champion's exact strategy run *bigger* (more exposure) and with a *harder* risk-off dial. Built after the first grader run landed in the weak regime (Score 40, mean +81, Sharpe ~1): in-sample Score **151.71**, Sharpe 2.21, and it lifts the weak half of the window from ~7 to ~33. A deliberate leverage increase, so the two files above stay as fallbacks. |
| `family_cluster_only.py` | **Baseline / fallback.** Same strategy minus the vol dial. Score 136.79, Sharpe 2.19. Use if the dial ever misbehaves on a new stage's data. |
| `eval.py` | Official evaluation/backtest script. Authoritative source for scoring and trading mechanics. Imports the **active** strategy on line ~9 (currently `family_cluster_volfilter`) — flip that one line to score a different file. **Don't edit anything else.** |
| `research.py` | Local research harness (**NOT submitted**). `backtest()` reproduces eval.py's Score/Sharpe/turnover exactly for any position function; `featureIC()` walk-forward-screens a candidate signal's predictive power. |
| `tune.py` | Self-service parameter explorer (**NOT submitted**). Sweep any strategy's knobs (each held constant or investigated over a list) and rank the results by the full-window and weak/strong-half Score, with a built-in `--perturb` overfit check. **Commands: see `README.md`.** |
| `helper.ipynb` | Analysis dashboard: equity curve + drawdown, daily-profit profile, today's bets, **per-instrument profit attribution**, and a signal lab. Aliases the champion as `teamName`. |
| `h1_analysis.ipynb` | Regime diagnosis — *why* the weak half of the window is weak — and the derivation of the vol dial. Aliases the baseline as `teamName`. |
| `prices.txt` | Current stage's price data — whitespace-separated, one column per instrument, one row per day, with a header row of tickers. |
| `requirements-dev.txt` | Pins the grading sandbox's package versions for a matching local env. **Never submit this file.** |

## The contract: `getMyPosition(prcSoFar)`

- **Input** `prcSoFar`: a NumPy array of shape `(nInst, nt)` — one row per instrument,
  columns are days in chronological order, `[:, -1]` is the most recent day.
  (Note: `prices.txt` is stored day-per-row, but `loadPrices` transposes it, so
  the function receives instrument-per-row.)
- **Output**: an integer array of length `nInst` — the **desired absolute position**
  (share count, not a delta) per instrument. Positive = long, negative = short.
- Called once per test day with all history up to and including that day.
- Return `np.zeros(nInst)` on the earliest days when there isn't enough history.

Current data: **51 instruments, 500 days**. Don't hard-code these — read them from
`prcSoFar.shape`. Our strategy files already derive `nInst`/`nt` from the shape (the grader may
run a different count on hidden data).

## Scoring & trading mechanics (from `eval.py` — the source of truth)

- Backtest scores the **last `numTestDays = 250` days** of whatever `prices.txt` provides.
- **Position limits (dollar-notional, per instrument):** default **$10,000**; instrument 0
  is special at **$100,000**. Positions are clipped to `int(dlrPosLimit / price)` shares
  each side, then floored to integers. Requesting more is silently clipped — not rejected.
- **Commission:** charged on traded dollar volume `price * |deltaPos|`. Default rate
  **0.0001 (1 bp)**; instrument 0 is special at **0.00002 (0.2 bp)**.
- **PnL** each day = change in portfolio value (cash + mark-to-market positions), net of commission.
- **Score** = `mu * sr^2 / (sr^2 + 1)` where `sr = sqrt(250) * mu / sigma` is the annualised
  Sharpe, `mu`/`sigma` are the daily-PnL mean/std, and `param = 1.0`. If `mu <= 0` the score
  is just `mu`. This rewards mean PnL but heavily penalises low Sharpe — high average PnL
  with high variance scores poorly.

Key takeaways for strategy design: turnover costs real money (mind commission and the clip),
instrument 0 has both a looser limit and cheaper trading, and the objective favours steady,
high-Sharpe PnL over volatile gains.

## Our strategy — what it is, what works, what doesn't

**What it does (plain language).** The champion `family_cluster_volfilter.py` is a *cross-sectional
mean-reversion* engine:
1. **Find families.** Group the 51 instruments into 6 "families" by which ones' daily moves have
   been most alike over the last **120 days** (re-done every day). Think of families as loose
   industry groups the data discovers on its own.
2. **Measure the stretch.** Within each family, an asset behaves like a rubber band tied to its
   family. Over the last **60 days** we measure how far each asset has pulled away from its family
   *after removing the move the whole family shared*.
3. **Bet on the snap-back.** Short the assets that ran ahead of their family, buy the ones that
   lagged; keep total buys ≈ total sells (so we're not betting on the market's overall direction).
4. **Size by calmness.** Bet less on jumpier assets (divide each bet by its own 20-day volatility).
5. **Leave-it-alone (no-trade band = 0.5).** Only re-trade an asset if its new target differs from
   what we hold by >50% — stops us paying commission to chase day-to-day noise.
6. **Risk-off dial (vol filter, k=1).** When the *whole market* gets choppier than its own 100-day
   normal, shrink every bet toward a 20% floor; ease back as it calms. Uses only past data.

**Strengths.**
- High in-sample risk-adjusted score (Score 138.63, annualised Sharpe 2.21).
- Roughly market-neutral (balanced longs/shorts) — doesn't rely on the market going up.
- **Well-tuned and *not fragile*.** Score stays strong across cluster counts 3–8 and across
  exposure levels; and (see below) it survived every "improve it" attempt we threw at it. That
  robustness is itself reassuring before a submission.
- Cheap built-in defences: the band cuts turnover ~8%; the vol dial is a gentle, Pareto-positive
  risk-off (helps H1, H2, and full).

**Weaknesses / fallbacks (the honest limits — read before trusting the headline).**
- **The edge is regime-dependent.** Split the 250-day scored window in half and it's two different
  worlds: a **dead** first half (H1: Score ~7, Sharpe ~0.5) and a **spectacular** second half
  (H2: Score ~277, Sharpe ~4). The headline 138 is just their average. **On new/hidden data we
  could land in either regime — expect the real Score to swing, possibly far below 138.** Don't
  over-trust the headline; judge changes by how they behave in the *weak* half, not just the full window.
- **The edge genuinely vanishes in some regimes** (H1 = higher-volatility market; our signal's
  next-day predictive power there is ~0 vs ~0.035 in H2). No trick manufactures edge that isn't
  there — the best available response is to *de-risk* (what the vol dial does), not trade harder.
- **Recency-dependent.** The signal needs *fresh* families and a short snap-back window; it does
  **not** benefit from more history or smoothing.
- **Fallback:** if the vol dial ever hurts on a given stage, `family_cluster_only.py` (band only,
  Score 136.79) is the safe drop-back.

**Ideas we TESTED and REJECTED (don't re-run these without a genuinely new angle).** Each was
measured by full backtest across the H1/H2/full split:
- **EWMA position smoothing** — hurts at every setting. A fast snap-back signal must actually trade;
  smoothing makes it hold stale bets.
- **No-trade band ≥ ~1.0** — collapses to *negative* Score (freezing a reversion book kills it). We
  took band = 0.5 (safe), *not* the noisy Score spike at 0.8 (a peak next to a cliff = overfit).
- **Edge/performance throttle** (scale exposure by our *own* recent realized IC or PnL) — actively
  harmful; drove H1 negative. A reversion book *loses right before it wins* (the spread stretching
  against us is the biggest coming snap-back), so cutting size after losses misses the bounce.
  **Lesson: throttle on the *environment* (market vol), never on our own P&L.**
- **"More robust" family detection** (Ledoit-Wolf shrinkage and/or longer 180–250-day correlation
  window) — hurts. Steadier families score *worse*; the ~22% day-to-day family churn is a *feature*.
- **Horizon averaging** (ensemble the snap-back over several windows) — the apparent big win (a config
  scoring 164) was **overfit**: it failed a ±5-day perturbation test (scattered 79→172), and the
  robust *dense-range* version scored *below* baseline. The single 60-day horizon is a real sweet spot.

**Meta-lesson.** This is a **fast, recency-driven cross-sectional reversion engine with a single
sharp sweet spot.** Gains do *not* come from smoothing / slowing / robustifying / using more data —
those all fight its nature. The only unexhausted direction is adding a genuinely *different,
uncorrelated* signal (cross-signal diversification for a higher Sharpe) — and even horizon
diversification already failed, so the bar is high. When testing any change: judge by **Score across
the H1/H2 split**, and **reject peaks that don't survive a small perturbation** (that's overfitting).

## Running the backtest

**Commands (setup, `eval.py`, `research.py`, `tune.py`, notebooks) live in `README.md`** — that's
the command reference; this file is for explanation. In short: `eval.py` scores the **active**
strategy (currently `family_cluster_volfilter`); flip its one import line to score a different file.
`research.py` mirrors it via the harness, and `research.backtest(fn, prc, ...)` scores any function
without editing files. `tune.py` sweeps parameters and runs the overfit check.

## Submission rules (see the Submission Guide before submitting)

- Zip **only** your algorithm file `<YourTeamName>.py` (files at the archive root, no nested
  folder), plus `requirements.txt` **only if** you use packages beyond the accepted set.
- **Do not** submit `eval.py`, `prices.txt`, or `requirements-dev.txt`.
- **Accepted without declaration:** numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib.
  Do **not** list these in `requirements.txt` — redeclaring them causes rejection. List only
  extra packages (e.g. `xgboost`), one per line; version pinning optional.
- **Sandbox constraints on `getMyPosition`:** no network access, no downloading/scraping,
  no reading local files beyond the `prcSoFar` argument. Rely only on accepted libraries.

## Communication & comment style

The user has a **data-science / ML background, not a finance one**. In code comments,
explanations, and summaries, explain finance concepts in **plain everyday language and
analogies**, not jargon. ML/stats terms are fine; finance shorthand is not.

- Instead of: *"strip out each instrument's beta to the market and fade the idiosyncratic
  cross-sectional move."*
- Write: *"ignore the general movement of the whole market and trade only the price swings
  unique to each asset."*

Keep the real numbers/metrics — just translate the finance idea the first time it appears.

## Conventions for changes here

- **One file per strategy.** Each strategy file is self-contained (its own `getMyPosition` +
  module-level helpers/state) and independently submittable. A new variant = a new descriptively-named
  file (copy the closest one), not an edit-in-place that loses the old version.
- The **champion** (currently `family_cluster_volfilter.py`) is the intended submission. `eval.py`
  and `research.py` pick the active strategy via a single switchable import line; the notebooks
  alias it as `teamName`, so re-pointing them at another strategy file is a one-line change.
- Preserve the `(nInst, nt)` input orientation and integer-position output contract; derive
  `nInst`/`nt` from `prcSoFar.shape` (don't hard-code 51/500).
- Don't alter `eval.py`'s mechanics while developing (only its import line) — the grader uses its
  own evaluator; local `eval.py` only mirrors it.
- When measuring a change, judge it by the printed **Score** (risk-adjusted) across the **H1/H2
  split**, not raw mean PnL and not just the full window — and reject peaks that fail a small
  perturbation (overfitting). See "Our strategy" above for the full record of what's been tried.
- Strategy `getMyPosition` must stay self-contained: only accepted libraries (the champion uses
  just numpy + scipy → no `requirements.txt` needed), no file/network access beyond `prcSoFar`.
