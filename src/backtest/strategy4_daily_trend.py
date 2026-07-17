"""Daily-timeframe trend on FX majors - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED (2026-07-17, before any run). WHY THIS IS DIFFERENT: the five
intraday hypotheses tested so far (prevday bounce, strategy2's four families,
overlap breakout) all measured ~0R GROSS - simple intraday chart patterns on
liquid FX carry no edge, and the spread then buries them. This changes BOTH
levers at once: (1) DAILY bars with trades that last weeks, so a 1-2 pip spread
is a rounding error against multi-hundred-pip trend legs; (2) ~20 years of daily
history, so the honest validation is NOT confined to the thin, now-reused 2026
intraday window - far less multiple-testing debt.

Hypothesis (declared): FX majors show persistent time-series momentum (the
documented trend-following / CTA premium). A plain moving-average-cross,
stop-and-reverse system should capture it if it exists. This is the same CLASS
as Peter's one validated edge - the daily gold macro-trend champion - which works
precisely because it is slow, not fast.

Rules (declared): DAILY candles. Direction = sign(SMA_fast - SMA_slow) using
values through the PRIOR close; act at the NEXT day's open +/- half-spread (no
lookahead). Always in the market when a direction exists: flip (exit + reverse)
when the cross flips. Protective stop = entry -/+ atr_k x ATR(atr_len) from the
prior day; stop checked before the flip, filled conservatively on gaps. 0.5%
risk per trade sized off that stop, 20% notional cap, no compounding. JPY-quote
PnL converted to USD at entry (declared approximation).

Grid (4 combos, declared): fast [20, 50], slow [100, 200] (fast < slow always).
atr_len 20, atr_k 3.0 fixed (not tuned). Train 2005-01 -> 2019-12, winner on
train (>=100 trades), judged ONCE on 2020 -> now. Gate: 100 trades, 0.05R, PF
1.15, 60% quarters+. WEAK PASS labeling. Spread x2 sensitivity.

HONEST GATE CAVEAT (declared): the 60%-positive-quarters criterion is built for
higher-frequency strategies. Trend-following is inherently lumpy - a few big
winners carry many small losers - so a genuinely positive trend edge can still
miss that one sub-gate. Read expectancy_R and profit factor as the primary
evidence; treat the quarters check as informational.

Run: python -m src.backtest.strategy4_daily_trend
"""

import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
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


def indicators(df, fast, slow, atr_len):
    """Return (dir_prev, atr_prev) aligned to df rows, both known at that row's OPEN."""
    close = df["close"]
    sma_f = close.rolling(fast).mean()
    sma_s = close.rolling(slow).mean()
    dir_prev = np.sign(sma_f.shift(1) - sma_s.shift(1))       # prior-close cross
    prev_close = close.shift(1)
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr_prev = tr.rolling(atr_len).mean().shift(1)            # prior-day ATR
    return dir_prev, atr_prev


def split_trades(trades, train_end, val_start):
    tr = [t for t in trades if pd.to_datetime(t.date).date() <= train_end]
    va = [t for t in trades if pd.to_datetime(t.date).date() >= val_start]
    return tr, va


def simulate_instrument(df, instrument, params, cfg):
    """Daily MA-cross stop-and-reverse for one FX instrument. Cost = half-spread/side."""
    fast, slow = int(params["fast"]), int(params["slow"])
    atr_k = float(cfg.get("atr_k", 3.0))
    atr_len = int(cfg.get("atr_len", 20))
    min_mult = float(cfg.get("min_stop_cost_mult", 2.0))
    hs = float(cfg["half_spread"][instrument])
    df = df.reset_index(drop=True)
    dir_prev, atr_prev = indicators(df, fast, slow, atr_len)
    n = len(df)
    trades = []
    pos = None

    def close_trade(i, exit_px, reason):
        row = df.iloc[i]
        f = usd_pnl_factor(instrument, pos["entry"])
        gross = (exit_px - pos["entry"]) * pos["shares"] * pos["side"] * f
        r = (exit_px - pos["entry"]) * pos["side"] / pos["risk_ps"]
        trades.append(Trade(
            symbol=instrument, strategy="strategy4_daily_trend",
            date=pos["entry_date"], entry_time=pos["entry_time"],
            exit_time=str(row["ts"]), entry=round(pos["entry"], 6),
            exit=round(exit_px, 6), shares=pos["shares"],
            stop=round(pos["stop"], 6), target=0.0,
            pnl=round(gross, 2), r_multiple=round(r, 3),
            exit_reason=reason, signal_reason=pos["reason"]))

    for i in range(n):
        row = df.iloc[i]
        dv = dir_prev.iloc[i]
        d = 0 if pd.isna(dv) else int(dv)
        a = atr_prev.iloc[i]

        # 1) protective stop (uses today's range), conservative on gaps
        if pos is not None:
            if pos["side"] == 1 and float(row["low"]) <= pos["stop"]:
                close_trade(i, min(float(row["open"]), pos["stop"]) - hs, "stop"); pos = None
            elif pos["side"] == -1 and float(row["high"]) >= pos["stop"]:
                close_trade(i, max(float(row["open"]), pos["stop"]) + hs, "stop"); pos = None

        # 2) flip: exit at today's open when the cross reverses
        if pos is not None and d != 0 and d != pos["side"]:
            close_trade(i, float(row["open"]) - pos["side"] * hs, "flip"); pos = None

        # 3) entry: flat + direction defined + ATR available
        if pos is None and d != 0 and not pd.isna(a) and a > 0:
            side = d
            entry_px = float(row["open"]) + side * hs
            stop = entry_px - side * atr_k * a
            risk_ps = (entry_px - stop) * side
            if risk_ps >= min_mult * 2 * hs:
                units = size_units(entry_px, stop, instrument, cfg)
                if units > 0:
                    pos = {"side": side, "entry": entry_px, "stop": stop, "shares": units,
                           "entry_date": row["ts"].strftime("%Y-%m-%d"),
                           "entry_time": str(row["ts"]), "risk_ps": risk_ps,
                           "reason": f"ma{fast}x{slow}_{'long' if side == 1 else 'short'}"}

    if pos is not None and n > 0:
        row = df.iloc[n - 1]
        close_trade(n - 1, float(row["close"]) - pos["side"] * hs, "data_end")
    return trades


def run():
    cfg = load_config()
    fx = cfg["fx_strategy4"]
    val_end = fx.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    instruments = fx["instruments"]
    gate = fx["gate"]
    train_floor = fx.get("min_train_expectancy_r", 0.02)
    train_end = pd.to_datetime(fx["train_end"]).date()
    val_start = pd.to_datetime(fx["val_start"]).date()
    gran = fx.get("granularity", "D")

    half_spread = {inst: float(fx["spread_pips"][inst]) * pip_size(inst) / 2.0
                   for inst in instruments}
    cfg_sim = {"risk": cfg["risk"], "half_spread": half_spread,
               "min_stop_cost_mult": fx.get("min_stop_cost_mult", 2.0),
               "atr_len": fx.get("atr_len", 20), "atr_k": fx.get("atr_k", 3.0)}

    frames = {}
    for inst in instruments:
        print(f"downloading {inst} {gran}, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end, granularity=gran)
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = df
    if not frames:
        slackbot.post(f"[FX-STRATEGY4] {ts} - FAILED: no candles returned.")
        return

    combos = expand_grid(fx["grids"])
    results = []
    for idx, combo in enumerate(combos, 1):
        all_trades = []
        for inst, df in frames.items():
            all_trades.extend(simulate_instrument(df, inst, combo, cfg_sim))
        tr, va = split_trades(all_trades, train_end, val_start)
        m = metrics.summarize(tr)
        print(f"  [s4 {idx}/{len(combos)}] {combo} -> train {m.get('trades', 0)} trades, "
              f"{m.get('expectancy_r', 0)}R", flush=True)
        results.append((combo, m, va))

    report = [f"# Daily-timeframe FX trend (MA cross) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Pre-registered 2026-07-17.",
              f"instruments {instruments} - spreads (pips): {fx['spread_pips']} - "
              f"daily bars - 0.5% risk/trade - stop-and-reverse - ATR({fx.get('atr_len', 20)}) x {fx.get('atr_k', 3.0)} stop",
              f"train {fx['train_start']} -> {fx['train_end']} - "
              f"validation {fx['val_start']} -> {val_end}",
              "NOTE: 60%-quarters gate is ill-suited to lumpy trend-following; "
              "read expectancy_R and PF as the primary evidence.", "",
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
    (out_dir / f"fx_strategy4_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy4_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY4]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Daily-timeframe FX trend (MA cross), {instruments}, daily bars, "
              "stop-and-reverse, 0.5% risk, spreads modeled")
    footer = f"Full detail: reports/fx_strategy4_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
