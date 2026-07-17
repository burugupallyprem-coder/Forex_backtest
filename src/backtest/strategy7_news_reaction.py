"""News-reaction on FX majors - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED (2026-07-17, before any run). Peter's idea: when a major news
event hits, trade the affected pair in the direction the news pushed it, across
all majors. Two honest constraints shape this build:

  * No economic calendar is available here, so events are detected the way the
    market itself reveals them: a simultaneous VOLATILITY + VOLUME spike on M15
    (true range > spike_k x ATR AND volume > vol_mult x average) is the footprint
    of a major release. Direction = the spike bar's own direction (the "news
    direction"). A true calendar version (event time + forecast vs actual) would
    replace this detector and needs an external calendar file - noted, not built.

  * NEWS-TIME EXECUTION is the whole game. Spreads blow out 5-20x exactly when you
    would enter. Entry pays a WIDENED spread (news_spread_mult x normal); exit pays
    normal. A concept test on gold M15 (2019-2026, 1242 spikes) found the raw
    continuation edge is ~+0.01 to +0.05R (a coin flip) and goes NEGATIVE even at
    a normal spread, catastrophic (-0.5 to -1R) at 5-10x. Expect this to FAIL; the
    value is measuring it honestly on real FX rather than asserting.

Rules (declared): M15 bars. On a spike bar, enter the NEXT bar's open in the spike
direction, paying news_spread_mult x half-spread. Protective stop = entry -/+
stop_atr x ATR. Exit on stop or after hold_bars (whichever first); no overlapping
trades. 0.5% risk sized off the stop, 20% notional cap. Grid (declared, 4 combos):
spike_k [2.5, 3.5], hold_bars [4, 8]. vol_mult 3.0, stop_atr 1.5 fixed. Train
2018-2022, judged once on 2023 -> now. Gate 100 trades / 0.05R / PF 1.15 / 60%
quarters+. WEAK PASS labeling. Reported at news spread, normal-only, and 10x.

Run: python -m src.backtest.strategy7_news_reaction
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
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


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


def split_trades(trades, train_end, val_start):
    tr = [t for t in trades if pd.to_datetime(t.date).date() <= train_end]
    va = [t for t in trades if pd.to_datetime(t.date).date() >= val_start]
    return tr, va


def simulate_instrument(df, instrument, params, cfg):
    """News-spike continuation on M15 with news-widened entry spread."""
    spike_k = float(params["spike_k"])
    hold = int(params["hold_bars"])
    atr_len = int(cfg.get("atr_len", 20))
    vol_len = int(cfg.get("vol_len", 20))
    vol_mult = float(cfg.get("vol_mult", 3.0))
    stop_atr = float(cfg.get("stop_atr", 1.5))
    hs = float(cfg["half_spread"][instrument])
    news_hs = hs * float(cfg.get("news_spread_mult", 6.0))
    df = df.reset_index(drop=True)
    o = df["open"].values.astype(float); h = df["high"].values.astype(float)
    l = df["low"].values.astype(float); c = df["close"].values.astype(float)
    vv = df["volume"].values.astype(float)
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1).values
    atr = pd.Series(tr).rolling(atr_len).mean().shift(1).values
    avgv = df["volume"].rolling(vol_len).mean().shift(1).values
    n = len(df)
    trades = []
    i = max(atr_len, vol_len) + 1
    while i < n - hold - 1:
        spike = (np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(avgv[i]) and avgv[i] > 0
                 and tr[i] > spike_k * atr[i] and vv[i] > vol_mult * avgv[i] and (c[i] - o[i]) != 0)
        if not spike:
            i += 1
            continue
        d = 1 if c[i] > o[i] else -1
        e = i + 1
        entry = o[e] + d * news_hs
        stop = entry - d * stop_atr * atr[i]
        risk_ps = stop_atr * atr[i]
        if risk_ps <= 0:
            i += 1
            continue
        units = size_units(entry, stop, instrument, cfg)
        if units <= 0:
            i += 1
            continue
        xe = e + hold
        exit_px, reason = None, "timeout"
        for j in range(e, xe + 1):
            if d == 1 and l[j] <= stop:
                exit_px, reason = min(o[j], stop) - hs, "stop"; break
            if d == -1 and h[j] >= stop:
                exit_px, reason = max(o[j], stop) + hs, "stop"; break
        if exit_px is None:
            exit_px = c[xe] - d * hs
        net = (exit_px - entry) * d
        f = usd_pnl_factor(instrument, entry)
        trades.append(Trade(
            symbol=instrument, strategy="strategy7_news_reaction",
            date=str(df["ts"].iloc[e])[:10], entry_time=str(df["ts"].iloc[e]),
            exit_time=str(df["ts"].iloc[min(xe, n - 1)]), entry=round(entry, 6),
            exit=round(exit_px, 6), shares=units, stop=round(stop, 6), target=0.0,
            pnl=round(net * units * f, 2), r_multiple=round(net / risk_ps, 3),
            exit_reason=reason, signal_reason=f"news_{'up' if d == 1 else 'down'}"))
        i = xe  # no overlapping trades
    return trades


def _sim_all(frames, combo, cfg_sim):
    out = []
    for inst, df in frames.items():
        out.extend(simulate_instrument(df, inst, combo, cfg_sim))
    return out


def run():
    cfg = load_config()
    fx = cfg["fx_strategy7"]
    val_end = fx.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    instruments = fx["instruments"]
    gate = fx["gate"]
    train_floor = fx.get("min_train_expectancy_r", 0.02)
    train_end = pd.to_datetime(fx["train_end"]).date()
    val_start = pd.to_datetime(fx["val_start"]).date()
    gran = fx.get("granularity", "M15")

    half_spread = {inst: float(fx["spread_pips"][inst]) * pip_size(inst) / 2.0 for inst in instruments}
    base = {"risk": cfg["risk"], "half_spread": half_spread,
            "atr_len": fx.get("atr_len", 20), "vol_len": fx.get("vol_len", 20),
            "vol_mult": fx.get("vol_mult", 3.0), "stop_atr": fx.get("stop_atr", 1.5)}
    news_mult = float(fx.get("news_spread_mult", 6.0))
    cfg_sim = {**base, "news_spread_mult": news_mult}

    frames = {}
    for inst in instruments:
        print(f"downloading {inst} {gran}, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end, granularity=gran)
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = df
    if not frames:
        slackbot.post(f"[FX-STRATEGY7] {ts} - FAILED: no candles returned.")
        return

    combos = expand_grid(fx["grids"])
    results = []
    for idx, combo in enumerate(combos, 1):
        tr, va = split_trades(_sim_all(frames, combo, cfg_sim), train_end, val_start)
        m = metrics.summarize(tr)
        print(f"  [s7 {idx}/{len(combos)}] {combo} -> train {m.get('trades', 0)} trades, "
              f"{m.get('expectancy_r', 0)}R", flush=True)
        results.append((combo, m, va))

    report = [f"# News-reaction on FX majors (spike proxy) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Pre-registered 2026-07-17.",
              f"instruments {instruments}", f"spreads (pips): {fx['spread_pips']}",
              f"event = TR>spike_k*ATR AND vol>{fx.get('vol_mult', 3.0)}x avg - trade continuation - "
              f"entry pays {news_mult}x spread (news blowout) - M15",
              f"train {fx['train_start']} -> {fx['train_end']} - validation {fx['val_start']} -> {val_end}",
              "CAVEAT: spike proxy (no calendar); entry-at-next-open is GENEROUS vs real mid-spike fills.", "",
              "## Train grid (all combos)", ""]
    for combo, m, _ in results:
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        report.append(f"- {cs}: {m.get('trades', 0)} trades, {m.get('expectancy_r', 0)}R, PF {m.get('profit_factor', 0)}")
    report.append("")

    eligible = [r for r in results if r[1].get("trades", 0) >= fx["min_train_trades"]]
    if not eligible:
        line = f"SKIP - no combo reached {fx['min_train_trades']} train trades"
        report += [f"## Verdict: {line}", ""]
        slack_body = [line]
    else:
        eligible.sort(key=lambda r: r[1]["expectancy_r"], reverse=True)
        best_combo, best_train, best_va = eligible[0]
        vm = metrics.summarize(best_va)
        combo_str = ", ".join(f"{k}={v}" for k, v in sorted(best_combo.items()))
        if vm.get("trades", 0) == 0:
            line = f"FAIL - winner {combo_str} produced 0 validation trades"
            report += [f"## Verdict: {line}", ""]
            slack_body = [line]
        else:
            verdict, why = metrics.gate_verdict(vm, gate)
            weak = verdict == "PASS" and best_train["expectancy_r"] < train_floor
            label = "WEAK PASS" if weak else verdict
            _, va_norm = split_trades(_sim_all(frames, best_combo, {**base, "news_spread_mult": 1.0}), train_end, val_start)
            vm_norm = metrics.summarize(va_norm)
            _, va_10 = split_trades(_sim_all(frames, best_combo, {**base, "news_spread_mult": 10.0}), train_end, val_start)
            vm_10 = metrics.summarize(va_10)
            report += [
                f"## Verdict: {label}",
                f"- winner: {combo_str}",
                f"- train: {best_train['trades']} trades, {best_train['expectancy_r']}R, PF {best_train['profit_factor']}",
                f"- validation (news {news_mult}x spread): {vm['trades']} trades, win {vm['win_rate']}%, "
                f"{vm['expectancy_r']}R (${vm['expectancy_usd']}/trade), PF {vm['profit_factor']}, "
                f"{vm['quarters_positive']}/{vm['quarters_total']} quarters+",
                f"- validation NORMAL spread only: {vm_norm.get('expectancy_r', 0)}R, PF {vm_norm.get('profit_factor', 0)}",
                f"- validation 10x spread: {vm_10.get('expectancy_r', 0)}R, PF {vm_10.get('profit_factor', 0)}",
                f"- gate (informational): {why}", ""]
            slack_body = [
                f"winner {combo_str}",
                f"train {best_train['expectancy_r']:+}R ({best_train['trades']}t) -> "
                f"val {vm['expectancy_r']:+}R (PF {vm['profit_factor']}, {vm['trades']}t) -> *{label}*",
                f"normal-spread val {vm_norm.get('expectancy_r', 0):+}R | 10x {vm_10.get('expectancy_r', 0):+}R"]

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy7_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy7_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY7]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"News-reaction (spike proxy), {len(instruments)} majors, M15, "
              f"continuation, entry pays {news_mult}x spread")
    footer = f"Full detail: reports/fx_strategy7_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
