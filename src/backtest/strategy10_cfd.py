"""Strategy #10 (3-step boring scalp) rerun on GOLD + S&P 500 CFDs - RESEARCH ONLY.

PRE-REGISTERED (2026-07-22). SAME engine and rules as strategy #10 on FX
(daily 9/21 EMA direction, PDH/PDL zones + 5-minute NY opening range,
1-minute break-and-retest, swing-trailing stop, flat 21:00 UTC, 0.5% risk).
Only three things change, all declared:

  1. Instruments are OANDA CFDs quoted in USD - XAU_USD (gold) and
     SPX500_USD (S&P 500 index). Peter chose the index CFD as the equity
     proxy; single stocks would need the Alpaca repo and are out of scope.
  2. Costs are modeled in PRICE units (spread_price / 2 per side) instead of
     FX pips, because these are not pip-quoted pairs.
  3. Each instrument is judged as its OWN independent study - its own train
     winner, its own once-only validation, its own gate verdict - so "how
     did it do on gold" and "how did it do on the S&P" get separate answers
     rather than a blended pool.

The core simulator, sizing, EMA/level helpers and honesty rules are IMPORTED
unchanged from strategy10 (single source of truth; already unit-tested).
usd_pnl_factor is 1.0 for both (USD quote) and unit notional = price, which
the shared helpers already produce. NOTE (declared): under the repo-wide 20%
notional cap, a ~5000-point index unit sizes down hard, so SPX net $ and
drawdown are understated - but expectancy_R and PF are size-independent, so
the PASS/FAIL verdict is unaffected. US cash open modeled at 13:30 UTC (EDT);
DST drift accepted as everywhere else. 2026 validation carries the usual
multiple-testing debt; any PASS is provisional.

Run: python -m src.backtest.strategy10_cfd
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import oanda_data, slackbot
from src.backtest import metrics
from src.backtest.strategy10_boring_scalp import (expand_grid, simulate_instrument,
                                                  split_trades)

ROOT = Path(__file__).resolve().parent.parent.parent


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def study_one(df, inst, fx, cfg_sim, half_spread):
    """Full train/pick/validate/gate for a SINGLE instrument. Returns report lines + slack line."""
    gate = fx["gate"]
    train_floor = fx.get("min_train_expectancy_r", 0.02)
    train_end = pd.to_datetime(fx["train_end"]).date()
    val_start = pd.to_datetime(fx["val_start"]).date()
    combos = expand_grid(fx["grids"])

    results = []
    for combo in combos:
        tr, va = split_trades(simulate_instrument(df, inst, combo, cfg_sim), train_end, val_start)
        results.append((combo, metrics.summarize(tr), va))

    lines = [f"### {inst}", "", "Train grid:"]
    for combo, m, _ in results:
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        lines.append(f"- {cs}: {m.get('trades', 0)} trades, "
                     f"{m.get('expectancy_r', 0)}R, PF {m.get('profit_factor', 0)}")

    eligible = [r for r in results if r[1].get("trades", 0) >= fx["min_train_trades"]]
    if not eligible:
        v = f"SKIP - no combo reached {fx['min_train_trades']} train trades"
        lines += ["", f"Verdict: {v}", ""]
        return lines, f"{inst}: {v}"

    eligible.sort(key=lambda r: r[1]["expectancy_r"], reverse=True)
    best_combo, best_train, best_va = eligible[0]
    vm = metrics.summarize(best_va)
    combo_str = ", ".join(f"{k}={v}" for k, v in sorted(best_combo.items()))
    if vm.get("trades", 0) == 0:
        v = f"FAIL - winner {combo_str} produced 0 validation trades"
        lines += ["", f"Verdict: {v}", ""]
        return lines, f"{inst}: {v}"

    verdict, why = metrics.gate_verdict(vm, gate)
    weak = verdict == "PASS" and best_train["expectancy_r"] < train_floor
    label = "WEAK PASS" if weak else verdict
    cfg2 = {**cfg_sim, "half_spread": {k: v * 2 for k, v in half_spread.items()}}
    _, va2 = split_trades(simulate_instrument(df, inst, best_combo, cfg2), train_end, val_start)
    vm2 = metrics.summarize(va2)
    lines += [
        "",
        f"Verdict: {label}",
        f"- winner: {combo_str}",
        f"- train: {best_train['trades']} trades, {best_train['expectancy_r']}R, "
        f"PF {best_train['profit_factor']}",
        f"- validation: {vm['trades']} trades, win {vm['win_rate']}%, {vm['expectancy_r']}R "
        f"(${vm['expectancy_usd']}/trade), PF {vm['profit_factor']}, "
        f"{vm['quarters_positive']}/{vm['quarters_total']} quarters+",
        f"- gate (informational): {why}",
        f"- spread x2 validation: {vm2.get('trades', 0)} trades, "
        f"{vm2.get('expectancy_r', 0)}R, PF {vm2.get('profit_factor', 0)}", ""]
    slack = (f"{inst}: train {best_train['expectancy_r']:+}R ({best_train['trades']}t) -> "
             f"val {vm['expectancy_r']:+}R (PF {vm['profit_factor']}, {vm['trades']}t) -> *{label}* "
             f"[x2 {vm2.get('expectancy_r', 0):+}R]")
    return lines, slack


def run():
    cfg = load_config()
    fx = cfg["fx_strategy10_cfd"]
    val_end = fx.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    instruments = fx["instruments"]

    half_spread = {inst: float(fx["spread_price"][inst]) / 2.0 for inst in instruments}
    cfg_sim = {"risk": cfg["risk"], "half_spread": half_spread,
               "min_stop_cost_mult": fx.get("min_stop_cost_mult", 2.0),
               "ny_open_min": fx["ny_open_min"], "or_minutes": fx["or_minutes"],
               "flat_min": fx["flat_min"], "ema_fast": fx["ema_fast"],
               "ema_slow": fx["ema_slow"], "zone_frac": fx["zone_frac"],
               "stop_buf_frac": fx["stop_buf_frac"]}

    report = [f"# 3-step boring scalp (strategy #10) on GOLD + S&P 500 CFDs - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Same engine/rules as fx_strategy10; "
              "CFD costs in price units; each instrument judged as its own study.",
              f"instruments {instruments} - spreads (price): {fx['spread_price']} - "
              "1m break-and-retest, 9/21 EMA dir + PDH/PDL + 5m OR, swing-trail, 0.5% risk",
              f"train {fx['train_start']} -> {fx['train_end']} - "
              f"validation {fx['val_start']} -> {val_end}", ""]
    slack_lines = []
    for inst in instruments:
        print(f"downloading {inst} M1, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end,
                                      granularity=fx.get("granularity", "M1"))
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if df.empty:
            report += [f"### {inst}", "", "Verdict: SKIP - no candles returned.", ""]
            slack_lines.append(f"{inst}: SKIP - no candles")
            continue
        lines, slack = study_one(df, inst, fx, cfg_sim, half_spread)
        report += lines
        slack_lines.append(slack)
        print(f"  {slack}", flush=True)

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy10cfd_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy10cfd_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY10-CFD]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Strategy #10 on GOLD + S&P 500 ({instruments}), 1m break-and-retest, "
              "9/21 EMA dir + PDH/PDL + 5m OR, swing-trail, 0.5% risk")
    footer = f"Full detail: reports/fx_strategy10cfd_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_lines + [footer]))


if __name__ == "__main__":
    run()
