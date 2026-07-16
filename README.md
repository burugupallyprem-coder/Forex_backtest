# Forex_backtest

**RESEARCH ONLY.** Nothing in this repo places trades, paper or real. It
backtests Peter's own strategy (from his 2026-07-16 video, rules confirmed
by him before implementation) on forex with honest costs.

## The strategy under test

Previous-day high/low levels -> 15m bounce entry (bar touches the level,
closes back inside) -> enter next bar -> stop behind the level
(0.1-0.2x prev-day range) -> target at the next fib (0.382/0.5) ->
London + New York entries only -> max 1-2 trades/day -> flat 21:00 UTC ->
0.5% risk per trade. Long-short symmetric.

## Why forex (and not crypto) is the fair test

The crypto twin (Crypto_trading_Bot repo, run 2026-07-16) measured the
signal at ~0.00R gross - the entire net loss there was the 0.20% cost
toll. FX spreads are 20-50x smaller relative to price, and prev-day
levels have structural meaning in FX. This repo answers whether the
strategy has real edge when costs stop drowning the signal.

## Setup

1. Push this folder to a **private** GitHub repo (e.g. `Forex_backtest`).
2. Add 3 repo secrets (Settings -> Secrets and variables -> Actions):
   `OANDA_API_KEY` (practice API token), `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`.
3. Actions -> `fx-prevday` -> Run workflow (~10-25 min).
4. Read the [FX-PREVDAY] Slack verdict; full report commits to `reports/`.

## Method (same discipline as the other repos)

Pre-declared grid (8 combos) -> winner picked on TRAIN (2024-07 ->
2025-12) -> judged ONCE on untouched 2026 validation -> gate: 100 trades,
0.05R, PF 1.15, 60% quarters+ -> WEAK PASS labeling -> spread x2
sensitivity. Verdicts are informational; deployment would need its own
roadmap, paper trial, legal clearance (F-1), and owner approval.

## Local tests

```bash
pip install -r requirements.txt
python tests/test_prevday_fx.py
```
