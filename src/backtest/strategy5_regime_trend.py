"""Regime-conditional daily FX trend - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED (2026-07-17, before any run). Strategy #4 showed the daily trend
had a REAL in-sample edge (+0.135R) that decayed out-of-sample (-0.103R) - a
regime effect, not a cost problem (spread x2 barely moved it). Hypothesis here:
the trend edge is only harvestable when price is actually TRENDING, so gate entries
on a trend-vs-range regime filter and skip the chop. This mirrors Peter's one
validated edge - the gold macro-trend champion, which trades only when its
real-yield filter permits.

Design (declared): base trend FIXED at SMA 20/200 (the trend rule that showed the
strongest in-sample edge in strategy #4) - fixed on purpose so the ONLY thing the
grid searches is the regime gate, minimising new degrees of freedom on the
already-reused 2020 validation window. Regime filter = Kaufman Efficiency Ratio
over er_len days: ER = |close[t]-close[t-n]| / sum|daily change| over n. ER near 1
= clean trend, near 0 = chop. Enter only when prior-day ER >= er_min. Everything
else identical to strategy #4: daily bars, stop-and-reverse, ATR(atr_len) x atr_k
stop, honest next-open +/- half-spread fills, stop before flip, 0.5% risk, 20% cap.

Grid (3 combos, declared): er_min [0.25, 0.35, 0.45]. Train 2005-2019, judged ONCE
on 2020 -> now. Gate 100 trades / 0.05R / PF 1.15 / 60% quarters+. WEAK PASS. Spread x2.

HONEST CAVEATS (declared): (1) the regime gate cuts trade count by design - if it
drops the validation sample well below 100, a positive result is low-confidence, not
a win. (2) the 60%-quarters gate is ill-suited to lumpy trend-following; read
expectancy_R and PF first. (3) this is another hypothesis on the same 2020 window -
multiple-testing debt; any PASS is provisional until fresh data.

Run: python -m src.backtest.strategy5_regime_trend
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


def efficiency_ratio(close, n):
    change = close.diff(n).abs()
    vol = close.diff().abs().rolling(n).sum()
    return change / vol.replace(0.0, np.nan)


def indicators(df, fast, slow, atr_len, er_len):
    """All returned series are known at the row's OPEN (shifted by 1, no lookahead)."""
    close = df["close"]
    dir_prev = np.sign(close.rolling(fast).mean().shift(1) - close.rolling(slow).mean().shift(1))
    prev_close = close.shift(1)
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr_prev = tr.rolling(atr_len).mean().shift(1)
    er_prev = efficiency_ratio(close, er_len).shift(1)
    return dir_prev, atr_prev, er_prev


def split_trades(trades, train_end, val_start):
    tr = [t for t in trades if pd.to_datetime(t.date).date() <= train_end]
    va = [t for t in trades if pd.to_datetime(t.date).date() >= val_start]
    return tr, va


def simulate_instrument(df, instrument, params, cfg):
    """Daily 20/200 trend, entries gated by Efficiency Ratio >= er_min."""
    fast, slow = int(cfg["fast"]), int(cfg["slow"])
    er_len = int(cfg["er_len"])
    er_min = float(params["er_min"])
    atr_k = float(cfg.get("atr_k", 3.0))
    atr_len = int(cfg.get("atr_len", 20))
    min_mult = float(cfg.get("min_stop_cost_mult", 2.0))
    hs = float(cfg["half_spread"][instrument])
    df = df.reset_index(drop=True)
    dir_prev, atr_prev, er_prev = indicators(df, fast, slow, atr_len, er_len)
    n = len(df)
    trades = []
    pos = None

    def close_trade(i, exit_px, reason):
        row = df.iloc[i]
        f = usd_pnl_factor(instrument, pos["entry"])
        gross = (exit_px - pos["entry"]) * pos["shares"] * pos["side"] * f
        r = (exit_px - pos["entry"]) * pos["side"] / pos["risk_ps"]
        trades.append(Trade(
            symbol=instrument, strategy="strategy5_regime_trend",
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
        er = er_prev.iloc[i]

        if pos is not None:
            if pos["side"] == 1 and float(row["low"]) <= pos["stop"]:
                close_trade(i, min(float(row["open"]), pos["stop"]) - hs, "stop"); pos = None
            elif pos["side"] == -1 and float(row["high"]) >= pos["stop"]:
                close_trade(i, max(float(row["open"]), pos["stop"]) + hs, "stop"); pos = None

        if pos is not None and d != 0 and d != pos["side"]:
            close_trade(i, float(row["open"]) - pos["side"] * hs, "flip"); pos = None

        # entry: flat + direction + ATR + REGIME GATE (ER >= er_min)
        if (pos is None and d != 0 and not pd.isna(a) and a > 0
                and not pd.isna(er) and er >= er_min):
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
                           "reason": f"ma{fast}x{slow}_er_{'long' if side == 1 else 'short'}"}

    if pos is not None and n > 0:
        row = df.iloc[n - 1]
        close_trade(n - 1, float(row["close"]) - pos["side"] * hs, "data_end")
    return trades


def run():
    cfg = load_config()
    fx = cfg["fx_strategy5"]
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
               "atr_len": fx.get("atr_len", 20), "atr_k": fx.get("atr_k", 3.0),
               "fast": fx["fast"], "slow": fx["slow"], "er_len": fx.get("er_len", 20)}

    frames = {}
    for inst in instruments:
        print(f"downloading {inst} {gran}, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end, granularity=gran)
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = df
    if not frames:
        slackbot.post(f"[FX-STRATEGY5] {ts} - FAILED: no candles returned.")
        return

    combos = expand_grid(fx["grids"])
    results = []
    for idx, combo in enumerate(combos, 1):
        all_trades = []
        for inst, df in frames.items():
            all_trades.extend(simulate_instrument(df, inst, combo, cfg_sim))
        tr, va = split_trades(all_trades, train_end, val_start)
        m = metrics.summarize(tr)
        print(f"  [s5 {idx}/{len(combos)}] {combo} -> train {m.get('trades', 0)} trades, "
              f"{m.get('expectancy_r', 0)}R", flush=True)
        results.append((combo, m, va))

    report = [f"# Regime-conditional daily FX trend (ER-gated 20/200) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Pre-registered 2026-07-17.",
              f"instruments {instruments} - spreads (pips): {fx['spread_pips']} - "
              f"daily bars - base trend 20/200 fixed - regime gate = Efficiency Ratio({fx.get('er_len', 20)})",
              f"train {fx['train_start']} -> {fx['train_end']} - "
              f"validation {fx['val_start']} -> {val_end}",
              "NOTE: regime gate cuts trade count by design; low validation N = low "
              "confidence. 60%-quarters gate ill-suited to trend-following.", "",
              "## Train grid (all combos)", ""]
    for combo, m, _ in results:
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        report.append(f"- {cs}: {m.get('trades', 0)} trades, "
                      f"{m.get('expectancy_r', 0)}R, PF {m.get('profit_factor', 0)}")
    report.append("")

    eligible = [r for r in results if r[1].get("trades", 0) >= fx["min_train_trades"]]
    if not eligible:
        verdict_line = f"SKIP - no combo reached {fx['min_train_trades']} train trades (regime gate too strict)"
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
    (out_dir / f"fx_strategy5_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy5_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY5]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Regime-conditional daily FX trend (ER-gated 20/200), {instruments}, "
              "daily bars, 0.5% risk, spreads modeled")
    footer = f"Full detail: reports/fx_strategy5_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
