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
| `family_cluster_ownrevert.py` | **CHAMPION — the current submission file.** Adds a NEW, universe-wide *own-price* reversion sleeve on top of the family sleeve. This is the big leap: in-sample Score **304**, Sharpe **2.88**, and — the point — it is **regime-balanced** (H1 296 ≈ H2 313), fixing the champion's fatal weak-half hole (was ~7). Default sleeve is dollar-balanced for generalization; a `OWN_NEUTRALIZE=False` flag takes a higher-return/higher-risk variant (Score ~432). See "The own-price reversion discovery" below. |
| `family_cluster_algo_custom.py` | **Gemini's ALGO-carve-out (superseded).** First to spot the own-price reversion edge, but applied it to *only* asset 0 (ALGO) via a dedicated 5-day z-score. In-sample Score **250**, Sharpe 3.06, H1 147. Correct insight, ~1/14 of the harvest — `ownrevert` generalises it to the whole universe. Kept as the record of where the idea came from. |
| `family_cluster_volfilter.py` | **Former champion / fallback.** Family mean-reversion + no-trade band + a market-volatility "risk-off" dial. In-sample Score **138.63**, Sharpe 2.21. Regime-fragile (H1 ~7). Safe drop-back if the own-price sleeve ever misbehaves on a new stage. |
| `family_cluster_bigsize.py` | **Weak-regime challenger (last submitted).** The volfilter strategy run *bigger* + a *harder* risk-off dial. Grader result: Score 64, mean +108, Sharpe ~1 (up from volfilter's Score 40, mean +81). In-sample Score **151.71**, Sharpe 2.21, H1 ~33. |
| `family_cluster_only.py` | **Baseline.** Family strategy minus the vol dial. Score 136.79, Sharpe 2.19. |
| `eval.py` | Official evaluation/backtest script. Authoritative source for scoring and trading mechanics. Imports the **active** strategy on line ~10 (currently `family_cluster_ownrevert`) — flip that one line to score a different file. **Don't edit anything else.** |
| `research.py` | Local research harness (**NOT submitted**, lives in `tools/`). `backtest()` reproduces eval.py's Score/Sharpe/turnover exactly for any position function; `featureIC()` walk-forward-screens a candidate signal's predictive power. |
| `tune.py` | Self-service parameter explorer (**NOT submitted**, in `tools/`). Sweep any strategy's knobs (each held constant or investigated over a list) and rank the results by the full-window and weak/strong-half Score, with a built-in `--perturb` overfit check. **Commands: see `README.md`.** |
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

**Meta-lesson (about the FAMILY sleeve).** The family engine is a **fast, recency-driven
cross-sectional reversion engine with a single sharp sweet spot.** Gains do *not* come from
smoothing / slowing / robustifying / using more data — those all fight its nature. The one
unexhausted direction was adding a genuinely *different, uncorrelated* signal for a higher Sharpe —
**and that is exactly what the own-price sleeve below turned out to be.** When testing any change:
judge by **Score across the H1/H2 split**, and **reject peaks that don't survive a small
perturbation** (that's overfitting).

## The own-price reversion discovery (the current champion's engine)

**The one-line story.** Gemini's `family_cluster_algo_custom.py` carved out asset 0 (ALGO) and
traded it with its *own* short-horizon reversion signal ("if today jumped above its own last-few-day
average, bet it snaps back"), which healed the weak regime. We asked: *is that edge special to ALGO?*
We screened all 51 instruments and found **it isn't** — the same own-price snap-back is present in
~14 names, and (unlike the family signal) it works in **both** regime halves. So we applied it to the
**whole universe**, not just ALGO. That is `family_cluster_ownrevert.py` (Score 304 vs the champion's
138), and it's the answer to "what other avenues": *harvest the broad own-price reversion, not just
asset 0's.*

**Two facts that matter most.**
- **The data reverts at two independent levels.** (1) *Cross-sectional* — asset vs its family (the old
  family sleeve; strong but **regime-fragile**, dead in H1). (2) *Own-price* — asset vs its **own**
  recent price over ~3 days (the new sleeve; weaker per-bet but **regime-stable**, works in H1 *and*
  H2). Screening every instrument, own-price reversion↔next-day-return correlation is ~+0.15–0.20 and
  **stable across both halves** (e.g. window-5: H1 +0.169, H2 +0.181). That regime-stability is the
  whole point — it plugs the hole the family signal can't.
- **ALGO is not special for the *signal*, only for the *sizing*.** Its reversion edge (+0.148) is
  middling — BLBT (inst 41) is stronger (+0.201). ALGO is worth over-weighting only because its
  position limit is 10× larger and its fees 5× cheaper (`OWN_INST0_MULT`), so you can run more size
  on it cheaply. That's the correct, generalisable reading of "hidden rule asymmetries": exploit the
  *limits/fees* of special names, don't assume the *signal* lives only there.

**The champion's design (`family_cluster_ownrevert.py`).** Two independent reversion sleeves added
together, then the usual leave-it-alone band + (now near-inert) vol dial:
- **Family sleeve** — the old cross-sectional bet, but at a smaller budget (`FAMILY_GROSS` 0.75M,
  down from 1.0M) since the own-price sleeve now does the heavy lifting.
- **Own-price sleeve** — each asset's 3-day z-score reversion (today excluded from its own mean, so
  no look-ahead), equal-risk sized (÷ each asset's 20-day vol), `OWN_BUDGET` 2.0M, asset 0 boosted 10×.

**The one honest risk knob — `OWN_NEUTRALIZE` (default True).** The own-price sleeve, left as-is, ran a
big *swinging* net market exposure (mean +9% of budget but **std 56%**). That un-balanced version made
~30% more average profit — but the extra came from an **uncontrolled whole-market timing bet** ("buy
everything after the market dips"), a single un-diversifiable wager. We **default to dollar-balanced**
(`OWN_NEUTRALIZE=True`, Score 304, Sharpe 2.88, H1 296 ≈ H2 313) because that keeps only the clean,
diversified per-asset alpha — the generalisation-safe choice. Set it `False` for the higher-return,
higher-risk variant (Score ~432, but leaning on that market-timing tilt).

**Robustness (why we trust it more than the old headline).** Judged on the H1/H2 split and perturbed:
- **Regime-balanced** (H1 296 ≈ H2 313). The exact property the family-only champion lacked — the
  thing that torched us on the grader (in-sample 138 → grader 40/64).
- **Smooth, not a knife-edge:** own-window 3–5 and family/own budgets and the band are all flat/monotone
  (no cliffs). The vol dial `k` now barely matters (<1 pt across k 0→2) — it was a crutch for the
  regime-fragile family signal; the regime-stable sleeve doesn't need it.
- **Broad, not a few lucky names:** drop the single best reversion name (BLBT) → basically unchanged;
  drop the top 6 of the 14 edge names → still beats the old champion in the weak half.

**Still in-sample.** 304 is a 250-day in-sample number; expect regime shrinkage on hidden data (the
family sleeve went ~2× lower on the grader). But the own-price sleeve's *regime-stability* is the
reason to expect it to hold up **better** than the family-only book did.

**Gemini's other two avenues — measured, low-value (don't spend time here):**
- *Cross-asset lead-lag for ALGO* (yesterday's market return vs ALGO, corr ~−0.06): tiny next to the
  own-price +0.17, and ALGO-specific. The universe-wide own sleeve already captures ALGO plus 13 others.
- *Momentum / trend-following hedge*: the 20-day momentum screen is mostly **negative** (reversion even
  at 20 days) and **not** regime-stable — momentum is a weak diversifier here. This is a reversion
  market at every horizon; a momentum overlay is unlikely to add uncorrelated value (consistent with
  the horizon-averaging and P&L-throttle rejections above).

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
