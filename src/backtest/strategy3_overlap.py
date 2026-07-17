"""Overlap opening-range breakout on FOREX - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED (2026-07-17, before any run). Hypothesis: during the London/NY
overlap - the day's highest-volume, tightest-spread window - an opening-range
breakout captures directional expansion. This is the idea explored on GOLD in the
gold repo (2026-07): there it measured ~0.00R GROSS edge and lost after costs. FX
is the fair retest because spreads are 20-50x smaller relative to price. The 2026
FX validation window already carries multiple-testing debt (this is the 3rd
hypothesis judged on it, after fx_prevday and fx_strategy2) - declared honestly;
any PASS is provisional until re-confirmed on data that does not exist yet.

Rules (declared): overlap window 13:00-17:00 UTC (fixed; DST drift accepted, as in
the rest of this repo). Opening range = first or_minutes of the window. Enter the
first 15m CLOSE that breaks the opening-range high (long) or low (short); fill at
the NEXT bar's open +/- half-spread (entry bar gets no free pass). Stop = opposite
side of the opening range (R = range width). Target = or_high + target_r*R (long) /
or_low - target_r*R (short). Optional daily-trend filter: only trade breaks aligned
with prev-day close vs its sma_days SMA. Max 1 trade/day. Flat by 17:00 UTC.
Long-short symmetric. Stop before target in the same bar; cost floor skips setups
whose stop < min_stop_cost_mult x round trip. 0.5% risk, 20% notional cap, no
compounding. JPY-quote PnL converted to USD at entry (declared approximation).

Grid (8 combos, declared): or_minutes [30, 60], target_r [1.0, 1.5],
trend_filter [false, true]. Train 2024-07 -> 2025-12, winner on train (>=100
trades), judged ONCE on 2026 validation. Gate: 100 trades, 0.05R, PF 1.15, 60%
quarters+. WEAK PASS labeling. Spread x2 sensitivity.

Run: python -m src.backtest.strategy3_overlap
"""

import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src import oanda_data, slackbot
from src.backtest import metrics

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Trade:
    symbol: str
    strategy: str
    date: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    shares: float
    stop: float
    target: float
    pnl: float
    r_multiple: float
    exit_reason: str
    signal_reason: str


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        import yaml
        return yaml.safe_load(fh)


def expand_grid(grid):
    keys = sorted(grid.keys())
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[k] for k in keys))]


def pip_size(instrument):
    return 0.01 if instrument.endswith("JPY") else 0.0001


def usd_pnl_factor(instrument, price):
    return (1.0 / price) if instrument.endswith("JPY") else 1.0


def unit_notional_usd(instrument, price):
    return 1.0 if instrument.startswith("USD") else price


def size_units(entry, stop, instrument, cfg):
    r = cfg["risk"]
    risk_dollars = r["equity"] * r["risk_pct"] / 100.0
    risk_ps_usd = abs(entry - stop) * usd_pnl_factor(instrument, entry)
    if risk_ps_usd <= 0:
        return 0.0
    max_units = (r["equity"] * r["max_position_pct"] / 100.0) / unit_notional_usd(instrument, entry)
    return round(max(min(risk_dollars / risk_ps_usd, max_units), 0.0), 2)


def daily_trend_map(df, sma_days):
    """trend_dir per date in {+1,-1,0}, known at that day's open (prior days only)."""
    d = df["ts"].dt.strftime("%Y-%m-%d")
    daily = df.groupby(d)["close"].last()
    sma = daily.rolling(sma_days).mean()
    prev_c, prev_s = daily.shift(1), sma.shift(1)
    out = {}
    for date in daily.index:
        pc, ps = prev_c[date], prev_s[date]
        if pd.isna(pc) or pd.isna(ps):
            out[date] = 0
        else:
            out[date] = 1 if pc > ps else (-1 if pc < ps else 0)
    return out


def split_trades(trades, train_end, val_start):
    tr = [t for t in trades if pd.to_datetime(t.date).date() <= train_end]
    va = [t for t in trades if pd.to_datetime(t.date).date() >= val_start]
    return tr, va


def simulate_instrument(df, instrument, params, cfg):
    """15m overlap opening-range breakout for one FX instrument. Costs = half-spread/side."""
    or_min = int(params["or_minutes"])
    tgt_r = float(params["target_r"])
    use_trend = bool(params["trend_filter"])
    min_mult = float(cfg.get("min_stop_cost_mult", 2.0))
    hs = float(cfg["half_spread"][instrument])
    ov_start = int(cfg["overlap_start_min"])
    ov_end = int(cfg["overlap_end_min"])
    or_end = ov_start + or_min
    trend = daily_trend_map(df, int(cfg.get("sma_days", 50))) if use_trend else None

    trades = []
    pos = None
    pending = None
    cur_date = None
    or_hi = or_lo = None
    took_today = False
    n = len(df)

    def close_trade(i, exit_px, reason):
        row = df.iloc[i]
        f = usd_pnl_factor(instrument, pos["entry"])
        gross = (exit_px - pos["entry"]) * pos["shares"] * pos["side"] * f
        r = (exit_px - pos["entry"]) * pos["side"] / pos["risk_ps"]
        trades.append(Trade(
            symbol=instrument, strategy="strategy3_overlap",
            date=pos["entry_date"], entry_time=pos["entry_time"],
            exit_time=str(row["ts"]), entry=round(pos["entry"], 6),
            exit=round(exit_px, 6), shares=pos["shares"],
            stop=round(pos["stop"], 6), target=round(pos["target"], 6),
            pnl=round(gross, 2), r_multiple=round(r, 3),
            exit_reason=reason, signal_reason=pos["reason"]))

    for i in range(n):
        row = df.iloc[i]
        date = row["ts"].strftime("%Y-%m-%d")
        minute = row["ts"].hour * 60 + row["ts"].minute
        if date != cur_date:
            cur_date = date
            or_hi = or_lo = None
            took_today = False
            pending = None

        # 1) flat at overlap end
        if pos is not None and minute >= ov_end:
            px = float(row["open"]) - pos["side"] * hs
            close_trade(i, px, "flat_overlap_end")
            pos = None

        # 2) pending fills at THIS bar open (inside the window)
        if pos is None and pending is not None and ov_start <= minute < ov_end:
            side = pending["side"]
            entry_px = float(row["open"]) + side * hs
            stop, target = pending["stop"], pending["target"]
            risk_ps = (entry_px - stop) * side
            good = (risk_ps > 0 and (target - entry_px) * side > 0
                    and risk_ps >= min_mult * 2 * hs)
            if good:
                units = size_units(entry_px, stop, instrument, cfg)
                if units > 0:
                    pos = {"side": side, "entry": entry_px, "stop": stop,
                           "target": target, "shares": units,
                           "entry_date": date, "entry_time": str(row["ts"]),
                           "risk_ps": risk_ps, "reason": pending["reason"]}
                    took_today = True
        pending = None

        # 3) manage position - entry bar included, stop before target
        if pos is not None:
            if pos["side"] == 1 and float(row["low"]) <= pos["stop"]:
                close_trade(i, min(float(row["open"]), pos["stop"]) - hs, "stop"); pos = None
            elif pos["side"] == 1 and float(row["high"]) >= pos["target"]:
                close_trade(i, max(float(row["open"]), pos["target"]) - hs, "target"); pos = None
            elif pos["side"] == -1 and float(row["high"]) >= pos["stop"]:
                close_trade(i, max(float(row["open"]), pos["stop"]) + hs, "stop"); pos = None
            elif pos["side"] == -1 and float(row["low"]) <= pos["target"]:
                close_trade(i, min(float(row["open"]), pos["target"]) + hs, "target"); pos = None

        # 4) build the opening range over [ov_start, or_end)
        if ov_start <= minute < or_end:
            h, l = float(row["high"]), float(row["low"])
            or_hi = h if or_hi is None else max(or_hi, h)
            or_lo = l if or_lo is None else min(or_lo, l)

        # 5) breakout signal on THIS close, fills next bar
        if (pos is None and not took_today and pending is None
                and or_hi is not None and or_end <= minute < ov_end and i < n - 1):
            rng = or_hi - or_lo
            if rng > 0:
                c = float(row["close"])
                td = trend.get(date, 0) if use_trend else None
                if c > or_hi and (not use_trend or td > 0):
                    pending = {"side": 1, "stop": or_lo,
                               "target": or_hi + tgt_r * rng,
                               "reason": "overlap_break_long"}
                elif c < or_lo and (not use_trend or td < 0):
                    pending = {"side": -1, "stop": or_hi,
                               "target": or_lo - tgt_r * rng,
                               "reason": "overlap_break_short"}
    return trades


def run():
    cfg = load_config()
    fx = cfg["fx_strategy3"]
    val_end = fx.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    instruments = fx["instruments"]
    gate = fx["gate"]
    train_floor = fx.get("min_train_expectancy_r", 0.02)
    train_end = pd.to_datetime(fx["train_end"]).date()
    val_start = pd.to_datetime(fx["val_start"]).date()

    half_spread = {inst: float(fx["spread_pips"][inst]) * pip_size(inst) / 2.0
                   for inst in instruments}
    cfg_sim = {"risk": cfg["risk"], "half_spread": half_spread,
               "min_stop_cost_mult": fx.get("min_stop_cost_mult", 2.0),
               "overlap_start_min": fx["overlap_start_min"],
               "overlap_end_min": fx["overlap_end_min"],
               "sma_days": fx.get("sma_days", 50)}

    frames = {}
    for inst in instruments:
        print(f"downloading {inst} M15, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end, granularity="M15")
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = df
    if not frames:
        slackbot.post(f"[FX-STRATEGY3] {ts} - FAILED: no candles returned.")
        return

    combos = expand_grid(fx["grids"])
    results = []
    for idx, combo in enumerate(combos, 1):
        all_trades = []
        for inst, df in frames.items():
            all_trades.extend(simulate_instrument(df, inst, combo, cfg_sim))
        tr, va = split_trades(all_trades, train_end, val_start)
        m = metrics.summarize(tr)
        print(f"  [s3 {idx}/{len(combos)}] {combo} -> train {m.get('trades', 0)} trades, "
              f"{m.get('expectancy_r', 0)}R", flush=True)
        results.append((combo, m, va))

    report = [f"# Overlap opening-range breakout on FOREX - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Pre-registered 2026-07-17.",
              f"instruments {instruments} - spreads (pips): {fx['spread_pips']} - "
              "overlap 13:00-17:00 UTC - 0.5% risk/trade - flat 17:00 UTC",
              f"train {fx['train_start']} -> {fx['train_end']} - "
              f"validation {fx['val_start']} -> {val_end}", "",
              "## Train grid (all combos)", ""]
    for combo, m, _ in results:
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        report.append(f"- {cs}: {m.get('trades', 0)} trades, "
                      f"{m.get('expectancy_r', 0)}R, PF {m.get('profit_factor', 0)}")
    report.append("")

    eligible = [r for r in results if r[1].get("trades", 0) >= fx["min_train_trades"]]
    if not eligible:
        verdict_line = f"SKIP - no combo reached {fx['min_train_trades']} train trades"
        report += [f"## Verdict: {verdict_line}", ""]
        slack_body = [verdict_line]
    else:
        eligible.sort(key=lambda r: r[1]["expectancy_r"], reverse=True)
        best_combo, best_train, best_va = eligible[0]
        vm = metrics.summarize(best_va)
        combo_str = ", ".join(f"{k}={v}" for k, v in sorted(best_combo.items()))
        if vm.get("trades", 0) == 0:
            verdict_line = f"FAIL - winner {combo_str} produced 0 validation trades"
            report += [f"## Verdict: {verdict_line}", ""]
            slack_body = [verdict_line]
        else:
            verdict, why = metrics.gate_verdict(vm, gate)
            weak = verdict == "PASS" and best_train["expectancy_r"] < train_floor
            label = "WEAK PASS" if weak else verdict
            cfg2 = {**cfg_sim, "half_spread": {k: v * 2 for k, v in half_spread.items()}}
            t2 = []
            for inst, df in frames.items():
                t2.extend(simulate_instrument(df, inst, best_combo, cfg2))
            _, va2 = split_trades(t2, train_end, val_start)
            vm2 = metrics.summarize(va2)
            report += [
                f"## Verdict: {label}",
                f"- winner: {combo_str}",
                f"- train: {best_train['trades']} trades, {best_train['expectancy_r']}R, "
                f"PF {best_train['profit_factor']}",
                f"- validation: {vm['trades']} trades, win {vm['win_rate']}%, "
                f"{vm['expectancy_r']}R (${vm['expectancy_usd']}/trade), PF {vm['profit_factor']}, "
                f"{vm['quarters_positive']}/{vm['quarters_total']} quarters+",
                f"- gate (informational): {why}",
                f"- spread x2 validation: {vm2.get('trades', 0)} trades, "
                f"{vm2.get('expectancy_r', 0)}R, PF {vm2.get('profit_factor', 0)}", ""]
            slack_body = [
                f"winner {combo_str}",
                f"train {best_train['expectancy_r']:+}R ({best_train['trades']}t) -> "
                f"val {vm['expectancy_r']:+}R (PF {vm['profit_factor']}, {vm['trades']}t) -> *{label}*",
                f"spread x2 val: {vm2.get('expectancy_r', 0):+}R"]

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy3_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy3_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY3]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Overlap opening-range breakout, {instruments}, 15m, "
              "London/NY overlap 13:00-17:00 UTC, 0.5% risk, spreads modeled")
    footer = f"Full detail: reports/fx_strategy3_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
