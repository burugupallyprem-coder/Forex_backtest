# Gold ORB-momentum (strategy #11) - 2026-07-22 12:14 UTC

RESEARCH ONLY - nothing deploys. Distinct family: NY opening-range breakout, big R targets; judged on UNTOUCHED holdout.
data 2018-01-01 -> 2026-07-21 - select 2024-07-01..2025-12-31 - holdout <= 2024-06-30

## Selection grid (spent window)

- or_minutes=15, target_r=2.0, trend_filter=False: 384 trades, 0.044R, PF 1.072
- or_minutes=15, target_r=2.0, trend_filter=True: 255 trades, 0.001R, PF 1.038
- or_minutes=15, target_r=3.0, trend_filter=False: 384 trades, 0.049R, PF 1.07
- or_minutes=15, target_r=3.0, trend_filter=True: 255 trades, 0.049R, PF 1.126
- or_minutes=30, target_r=2.0, trend_filter=False: 381 trades, 0.087R, PF 1.123
- or_minutes=30, target_r=2.0, trend_filter=True: 233 trades, 0.043R, PF 1.11
- or_minutes=30, target_r=3.0, trend_filter=False: 381 trades, 0.062R, PF 1.077
- or_minutes=30, target_r=3.0, trend_filter=True: 233 trades, 0.054R, PF 1.133

## Verdict: FAIL
- winner: or_minutes=30, target_r=2.0, trend_filter=False
- selection: 381 trades, 0.087R, PF 1.123
- HOLDOUT (untouched judge): 1576 trades, win 42.3%, -0.033R ($-0.48/trade), PF 0.985, 10/26 quarters+, maxDD $4,622.61
- holdout GROSS (0 spread): 1576 trades, 0.04R (spread eats +0.073R)
- bootstrap holdout 90% CI: [-0.077R, +0.010R], P(>0)=11.1% -> CI includes 0
- walk-forward: 1/6 folds+ (per-fold R: -0.158, -0.021, -0.076, -0.056, +0.121, -0.011)
- 2026 reference (CONTAMINATED, not a gate): 139 trades, 0.052R, PF 1.03
- gate: expectancy -0.033R < 0.05R; PF 0.985 < 1.15; only 10/26 quarters positive
