"""Peter's prev-day fib bounce on FOREX - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED (2026-07-16, rules confirmed by Peter from his own video,
written before any run): previous-day high/low levels; BOUNCE entry (bar
touches the prev-day low/high and the 15m candle closes back inside);
enter next bar; stop behind the level by stop_buf_frac x prev-day range;
target at target_fib of the prev-day range from the touched boundary;
London (08:00-13:30 UTC) + New York (13:30-21:00 UTC) entries only;
max 1-2 trades/day; flat by 21:00 UTC. Long-short symmetric.

WHY FOREX IS THE REAL TEST (declared): costs. USD/JPY round trip is
~1.6 pips (~0.011%) vs crypto's 0.20%. The crypto twin (run 2026-07-16)
measured the signal at ~0.00R GROSS - all crypto loss was toll. FX levels
have structural meaning (daily close, session flows) and the toll is
20-50x smaller. Verdict unknown; the 2026 FX validation window has never
been used - no multiple-testing debt on this data.

Honesty rules: fills at next bar open +/- half-spread; entry bar gets no
free pass; stop before target in the same bar; gaps fill at the open on
the bad side; cost floor - skip setups whose stop < min_stop_cost_mult x
round-trip cost; fixed $100k equity, 0.5% risk sizing, 20% notional cap,
no compounding. JPY-quote PnL converted to USD at entry price (declared
approximation). Sessions use fixed UTC windows (DST drift accepted and
declared). Weekend gaps: Monday's levels come from the previous TRADING
day present in the data (Friday).

Grid (8 combos, declared): target_fib [0.382, 0.5],
stop_buf_frac [0.1, 0.2], max_trades_day [1, 2]. Train 2024-07 ->
2025-12, validate 2026 -> now, winner on train (>=100 trades), judged
ONCE on validation. Gate: 100 trades, 0.05R, PF 1.15, 60% quarters+.
WEAK PASS labeling. Spread x2 sensitivity.

Run: python -m src.backtest.prevday_fib_fx
"""

import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import oanda_data, slackbot
from src.backtest import metrics

ROOT = Path(__file__).resolve().parent.parent.parent

LONDON = (8 * 60, 13 * 60 + 30)
NEWYORK = (13 * 60 + 30, 21 * 60)
FLAT_MIN = 21 * 60


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
        return yaml.safe_load(fh)


def expand_grid(grid):
    keys = sorted(grid.keys())
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[k] for k in keys))]


def in_session(minute):
    return (LONDON[0] <= minute < LONDON[1]) or (NEWYORK[0] <= minute < NEWYORK[1])


def pip_size(instrument):
    return 0.01 if instrument.endswith("JPY") else 0.0001


def usd_pnl_factor(instrument, price):
    """USD value of a 1-price-unit move per 1 unit of base currency."""
    return (1.0 / price) if instrument.endswith("JPY") else 1.0


def unit_notional_usd(instrument, price):
    """Approx USD notional of one unit of base currency."""
    return 1.0 if instrument.startswith("USD") else price


def size_units(entry, stop, instrument, cfg):
    r = cfg["risk"]
    risk_dollars = r["equity"] * r["risk_pct"] / 100.0
    risk_ps_usd = abs(entry - stop) * usd_pnl_factor(instrument, entry)
    if risk_ps_usd <= 0:
        return 0.0
    max_units = (r["equity"] * r["max_position_pct"] / 100.0) / unit_notional_usd(instrument, entry)
    return round(max(min(risk_dollars / risk_ps_usd, max_units), 0.0), 2)


def prev_day_levels(df):
    d = df["ts"].dt.strftime("%Y-%m-%d")
    g = df.groupby(d).agg(hi=("high", "max"), lo=("low", "min"))
    g["prev_hi"] = g["hi"].shift(1)
    g["prev_lo"] = g["lo"].shift(1)
    return {idx: (row["prev_hi"], row["prev_lo"])
            for idx, row in g.iterrows()
            if row["prev_hi"] == row["prev_hi"]}


def simulate_instrument(df, instrument, params, cfg):
    """15m bounce sim for one FX instrument. Costs = half-spread per side."""
    tgt_f = float(params["target_fib"])
    buf = float(params["stop_buf_frac"])
    max_td = int(params["max_trades_day"])
    min_mult = float(cfg.get("min_stop_cost_mult", 2.0))
    hs = float(cfg["half_spread"][instrument])
    levels = prev_day_levels(df)

    trades = []
    pos = None
    pending = None
    cur_date, trades_today = None, 0
    n = len(df)

    def close_trade(i, exit_px, reason):
        row = df.iloc[i]
        f = usd_pnl_factor(instrument, pos["entry"])
        gross = (exit_px - pos["entry"]) * pos["shares"] * pos["side"] * f
        r = (exit_px - pos["entry"]) * pos["side"] / pos["risk_ps"]
        trades.append(Trade(
            symbol=instrument, strategy="prevday_fib_fx",
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
            cur_date, trades_today = date, 0
            pending = None

        # 1) flat cutoff
        if pos is not None and minute >= FLAT_MIN:
            px = float(row["open"]) - pos["side"] * hs
            close_trade(i, px, "flat_2100")
            pos = None

        # 2) pending entry fills at THIS bar open
        if pos is None and pending is not None and minute < FLAT_MIN and in_session(minute):
            side = pending["side"]
            entry_px = float(row["open"]) + side * hs
            stop, target = pending["stop"], pending["target"]
            risk_ps = (entry_px - stop) * side
            good = (risk_ps > 0
                    and (target - entry_px) * side > 0
                    and risk_ps >= min_mult * 2 * hs)
            if good:
                units = size_units(entry_px, stop, instrument, cfg)
                if units > 0:
                    pos = {"side": side, "entry": entry_px, "stop": stop,
                           "target": target, "shares": units,
                           "entry_date": date, "entry_time": str(row["ts"]),
                           "risk_ps": risk_ps, "reason": pending["reason"]}
                    trades_today += 1
        pending = None

        # 3) manage position - including the entry bar (no free pass).
        #    Stop checked before target, conservative.
        if pos is not None:
            if pos["side"] == 1 and float(row["low"]) <= pos["stop"]:
                px = min(float(row["open"]), pos["stop"]) - hs
                close_trade(i, px, "stop")
                pos = None
            elif pos["side"] == 1 and float(row["high"]) >= pos["target"]:
                px = max(float(row["open"]), pos["target"]) - hs
                close_trade(i, px, "target")
                pos = None
            elif pos["side"] == -1 and float(row["high"]) >= pos["stop"]:
                px = max(float(row["open"]), pos["stop"]) + hs
                close_trade(i, px, "stop")
                pos = None
            elif pos["side"] == -1 and float(row["low"]) <= pos["target"]:
                px = min(float(row["open"]), pos["target"]) + hs
                close_trade(i, px, "target")
                pos = None

        # 4) signal on THIS close for next bar
        if (pos is None and date in levels and trades_today < max_td
                and in_session(minute) and i < n - 1):
            prev_hi, prev_lo = levels[date]
            rng = prev_hi - prev_lo
            if rng > 0:
                lo_bar, hi_bar, c = float(row["low"]), float(row["high"]), float(row["close"])
                if lo_bar <= prev_lo and c > prev_lo:
                    pending = {"side": 1, "stop": prev_lo - buf * rng,
                               "target": prev_lo + tgt_f * rng,
                               "reason": "pd_low_bounce_long"}
                elif hi_bar >= prev_hi and c < prev_hi:
                    pending = {"side": -1, "stop": prev_hi + buf * rng,
                               "target": prev_hi - tgt_f * rng,
                               "reason": "pd_high_bounce_short"}

    if pos is not None and n > 0:
        row = df.iloc[n - 1]
        px = float(row["close"]) - pos["side"] * hs
        close_trade(n - 1, px, "data_end")
    return trades


def split_trades(trades, train_end, val_start):
    tr = [t for t in trades if pd.to_datetime(t.date).date() <= train_end]
    va = [t for t in trades if pd.to_datetime(t.date).date() >= val_start]
    return tr, va


def run():
    cfg = load_config()
    fx = cfg["fx_prevday"]
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
               "min_stop_cost_mult": fx.get("min_stop_cost_mult", 2.0)}

    frames = {}
    for inst in instruments:
        print(f"downloading {inst} M15, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end, granularity="M15")
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = df
    if not frames:
        slackbot.post(f"[FX-PREVDAY] {ts} - FAILED: no candles returned.")
        return

    combos = expand_grid(fx["grids"])
    results = []
    for idx, combo in enumerate(combos, 1):
        all_trades = []
        for inst, df in frames.items():
            all_trades.extend(simulate_instrument(df, inst, combo, cfg_sim))
        tr, va = split_trades(all_trades, train_end, val_start)
        m = metrics.summarize(tr)
        print(f"  [fx {idx}/{len(combos)}] {combo} -> train {m.get('trades', 0)} trades, "
              f"{m.get('expectancy_r', 0)}R", flush=True)
        results.append((combo, m, va))

    report = [f"# Prev-day fib bounce (Peter's strategy) on FOREX - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Rules confirmed by Peter 2026-07-16.",
              f"instruments {instruments} - spreads (pips): {fx['spread_pips']} - "
              "0.5% risk/trade - London+NY entries - flat 21:00 UTC",
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
            cfg2 = {**cfg_sim,
                    "half_spread": {k: v * 2 for k, v in half_spread.items()}}
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
    (out_dir / f"fx_prevday_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_prevday_{stamp}.md", flush=True)

    header = (f"*[FX-PREVDAY]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Peter's prev-day fib bounce, {instruments}, 15m, "
              "London+NY only, 0.5% risk, spreads modeled")
    footer = f"Full detail: reports/fx_prevday_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
