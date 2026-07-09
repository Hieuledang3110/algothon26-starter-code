# Algothon 2026 — Strategy Plan

Working design doc. Not part of the submission (only `<YourTeamName>.py` is submitted).

## The problem in one line

Each day we see all close prices up to today (`prcSoFar`, shape `(nInst, nt)`) and must
return **integer target positions** per instrument. We are scored on the **last 250 days**.

## What the scoring actually rewards

Score = `mu · sr²/(sr²+1)`, where `mu` = mean **daily PnL in dollars** and
`sr` = annualised Sharpe (`sqrt(250)·mu/sigma`). See [eval.py](eval.py).

- Below ~Sharpe 1 the fraction is < 0.5 → heavy penalty. Above ~Sharpe 3 it saturates near 1.
- **Implication:** get Sharpe comfortably high **via diversification**, then **maximise deployed
  capital / mean PnL**. Don't over-optimise Sharpe past ~3 while leaving position limits unused.
- High Sharpe comes from **many small, weakly-correlated bets**, not a few big ones. With 51
  instruments × $10k limits (~$500k gross, +$100k on instrument 0), a broad, roughly
  **market-neutral** book is the natural high-Sharpe structure.

### Mechanics that shape the strategy (all from `eval.py`)
- **Position limits (dollar):** $10,000 default; **instrument 0 = $100,000**. Positions are
  clipped to `int(limit/price)` shares each side — over-requests are silently clipped, not rejected.
- **Commission:** 1 bp (`0.0001`) on traded dollar volume `price·|Δpos|`; **instrument 0 = 0.2 bp**.
  Turnover is a real cost and the main silent Sharpe-killer.
- Positions are absolute targets (not deltas); only `int` share counts survive.

## Core thesis

At daily frequency across a correlated universe, the dominant, most robust edge is usually
**cross-sectional mean reversion / statistical arbitrage**: instruments that have recently
*out*performed their peers tend to give some of it back, and vice versa. This is naturally
diversified and market-neutral → exactly the high-Sharpe structure the score rewards.

Clustering feeds this directly: an instrument's **deviation from its cluster** (the residual/
spread) is a cleaner reversion signal than its deviation from the whole market.

## Decisions to lock down first (matter more than the model)

1. **Prediction target:** next-day (or 2–5 day) forward log return per instrument. Everything —
   features, clusters — serves this. Longer horizon → lower turnover.
2. **Validation:** **walk-forward** (train on days ≤ t, test forward, roll). Screen every feature
   by out-of-sample **Information Coefficient** (daily cross-sectional rank-corr of feature vs
   forward return) *before* it enters the model. The grader runs on **new/extended data across
   stages**, so anything tuned to the current 250 days will not generalise.
3. **Costs/turnover budget:** no-trade band + position smoothing (or explicit turnover penalty)
   from day one, not bolted on later.

## Feature menu (grouped by the edge they capture)

Only **daily close** is available — no volume, no OHLC — so microstructure/intraday-vol features
are out. Every feature must use data only up to day `t` (rolling/expanding, no look-ahead).

- **Trend / momentum:** multi-horizon returns (5/10/20/60d); price vs MA; MA crossover;
  rolling trend slope + its R².
- **Mean reversion:** z-score of price vs rolling mean (workhorse); 1-day reversal; RSI;
  distance from Bollinger bands; lag-1 return autocorrelation (reverts vs trends).
- **Cross-sectional / cluster-relative (the clustering payoff):** return − cluster-mean return
  (spread) and its z-score; cross-sectional rank of momentum; beta to equal-weight "market"
  and the idiosyncratic residual.
- **Volatility / risk:** rolling realised vol (2 windows); short/long vol ratio (regime).
  Vol also doubles as the **position-sizing denominator** (size ∝ signal/vol).
- **Statistical character:** Hurst exponent (trend vs revert); spectral dominant-cycle period;
  up-day ratio (directional bias).

## Pipeline (each stage swappable)

```
prices → leak-free rolling features → rolling clustering (on returns, corr-distance)
       → per-cluster / pooled regression → forward-return forecast
       → vol-scaled, dollar-neutral sizing → turnover smoothing → integer positions
```

## Roadmap

1. **Research harness** (`research.py`, not submitted): walk-forward backtest that reproduces
   `eval.py`'s Score, plus a per-feature IC report. Lets us screen ideas without eyeballing.
2. **Baseline** (`teamName.py`): pure cross-sectional mean reversion (long peer-underperformers,
   short peer-outperformers, vol-scaled, dollar-neutral). ~30 lines, hard to overfit, naturally
   high-Sharpe. This is the number every fancier idea must beat **after costs**.
3. **Add clustering** → replace market-relative with cluster-relative residuals.
4. **Regression** → combine the vetted features into a forecast; keep it regularised (ridge /
   lightGBM). Only keep it if it beats the baseline net of turnover.
5. **Sizing & cost control** → tune gross exposure to push `mu` up while Sharpe stays > ~2–3;
   add no-trade band / smoothing to cut commission.

## Guardrails
- Leak-free features (rolling stats + expanding standardisation only).
- Judge everything by the printed **Score**, not raw mean PnL.
- Re-cluster on a rolling window; don't let cluster churn create phantom turnover.
- Start linear/simple; prove the pipeline before adding model complexity.
