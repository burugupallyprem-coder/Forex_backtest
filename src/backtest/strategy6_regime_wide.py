"""Strategy #5 rule on a WIDER basket + overnight swap cost - RESEARCH ONLY, nothing trades.

NOT A NEW HYPOTHESIS. This re-tests the pre-registered Strategy #5 rule EXACTLY
(daily SMA 20/200 stop-and-reverse, gated by Efficiency Ratio >= er_min, same
grid er_min [0.25,0.35,0.45], same 2005-2019 train / 2020-> validation split).
Two things change, both to make the evidence honest, not to fish:

  1. WIDER BASKET (7 majors: USD_JPY, EUR_USD, GBP_USD, AUD_USD, USD_CHF,
     USD_CAD, NZD_USD). Strategy #5 was positive in BOTH periods (val +0.161R,
     PF 1.19) but on only 57 validation trades - statistically indistinguishable
     from zero. More pairs -> more independent trades in the same window, to see
     if the edge survives at higher statistical power. CAVEAT (declared): these
     pairs are correlated (AUD/NZD/CAD risk-on; EUR/GBP/CHF Europe), so the
     effective independent sample is smaller than the raw trade count.

  2. OVERNIGHT SWAP. S5 charged spread but NOT financing, and this strategy holds
     for weeks. swap_annual_pct is a CONSERVATIVE, symmetric per-day drag on
     notional (declared stand-in). Real swap needs interest-rate-differential data
     we don't have here and can be POSITIVE (carry) or negative - so treat this as
     a cost FLOOR, not a precise model. Reported alongside a no-swap number.

Everything else identical to strategy #5. Honest fills, stop before flip, 0.5%
risk, WEAK PASS labeling, spread x2 sensitivity. Gate 100 trades / 0.05R / PF 1.15
/ 60% quarters+. Multiple-testing debt on the 2020 window still applies.

Run: python -m src.backtest.strategy6_regime_wide
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
    """S5 regime-conditional trend + overnight swap drag (swap_annual_pct)."""
    fast, slow = int(cfg["fast"]), int(cfg["slow"])
    er_len = int(cfg["er_len"])
    er_min = float(params["er_min"])
    atr_k = float(cfg.get("atr_k", 3.0))
    atr_len = int(cfg.get("atr_len", 20))
    min_mult = float(cfg.get("min_stop_cost_mult", 2.0))
    swap_annual = float(cfg.get("swap_annual_pct", 0.0))
    hs = float(cfg["half_spread"][instrument])
    df = df.reset_index(drop=True)
    dir_prev, atr_prev, er_prev = indicators(df, fast, slow, atr_len, er_len)
    n = len(df)
    trades = []
    pos = None

    def close_trade(i, exit_px, reason):
        row = df.iloc[i]
        f = usd_pnl_factor(instrument, pos["entry"])
        gross_move = (exit_px - pos["entry"]) * pos["side"]
        d0 = pd.to_datetime(pos["entry_date"]).date()
        d1 = pd.to_datetime(str(row["ts"])).date()
        days_held = max((d1 - d0).days, 0)
        swap = days_held * pos["entry"] * (swap_annual / 100.0) / 365.0   # adverse, price units
        net_move = gross_move - swap
        trades.append(Trade(
            symbol=instrument, strategy="strategy6_regime_wide",
            date=pos["entry_date"], entry_time=pos["entry_time"],
            exit_time=str(row["ts"]), entry=round(pos["entry"], 6),
            exit=round(exit_px, 6), shares=pos["shares"],
            stop=round(pos["stop"], 6), target=0.0,
            pnl=round(net_move * pos["shares"] * f, 2),
            r_multiple=round(net_move / pos["risk_ps"], 3),
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


def _sim_all(frames, combo, cfg_sim):
    out = []
    for inst, df in frames.items():
        out.extend(simulate_instrument(df, inst, combo, cfg_sim))
    return out


def run():
    cfg = load_config()
    fx = cfg["fx_strategy6"]
    val_end = fx.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    instruments = fx["instruments"]
    gate = fx["gate"]
    train_floor = fx.get("min_train_expectancy_r", 0.02)
    train_end = pd.to_datetime(fx["train_end"]).date()
    val_start = pd.to_datetime(fx["val_start"]).date()
    gran = fx.get("granularity", "D")

    half_spread = {inst: float(fx["spread_pips"][inst]) * pip_size(inst) / 2.0 for inst in instruments}
    base = {"risk": cfg["risk"], "half_spread": half_spread,
            "min_stop_cost_mult": fx.get("min_stop_cost_mult", 2.0),
            "atr_len": fx.get("atr_len", 20), "atr_k": fx.get("atr_k", 3.0),
            "fast": fx["fast"], "slow": fx["slow"], "er_len": fx.get("er_len", 20)}
    swap = float(fx.get("swap_annual_pct", 0.0))
    cfg_sim = {**base, "swap_annual_pct": swap}          # realistic: spread + swap

    frames = {}
    for inst in instruments:
        print(f"downloading {inst} {gran}, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end, granularity=gran)
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = df
    if not frames:
        slackbot.post(f"[FX-STRATEGY6] {ts} - FAILED: no candles returned.")
        return

    combos = expand_grid(fx["grids"])
    results = []
    for idx, combo in enumerate(combos, 1):
        tr, va = split_trades(_sim_all(frames, combo, cfg_sim), train_end, val_start)
        m = metrics.summarize(tr)
        print(f"  [s6 {idx}/{len(combos)}] {combo} -> train {m.get('trades', 0)} trades, "
              f"{m.get('expectancy_r', 0)}R", flush=True)
        results.append((combo, m, va))

    report = [f"# Strategy #5 rule on WIDER basket + swap - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Re-test of pre-registered strategy #5 (unchanged rule).",
              f"instruments {instruments}", f"spreads (pips): {fx['spread_pips']}",
              f"swap drag: {swap}%/yr on notional (conservative stand-in) - "
              "daily bars - base trend 20/200 - ER-gated - 0.5% risk",
              f"train {fx['train_start']} -> {fx['train_end']} - validation {fx['val_start']} -> {val_end}",
              "CAVEATS: pairs are correlated (effective N < trade count); swap is a cost floor, "
              "real financing can be +/-; 60%-quarters gate ill-suited to trend-following.", "",
              "## Train grid (all combos)", ""]
    for combo, m, _ in results:
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        report.append(f"- {cs}: {m.get('trades', 0)} trades, {m.get('expectancy_r', 0)}R, "
                      f"PF {m.get('profit_factor', 0)}")
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
            # sensitivities on the winner
            _, va_ns = split_trades(_sim_all(frames, best_combo, {**base, "swap_annual_pct": 0.0}), train_end, val_start)
            vm_ns = metrics.summarize(va_ns)
            _, va_2x = split_trades(_sim_all(frames, best_combo,
                                             {**base, "swap_annual_pct": swap,
                                              "half_spread": {k: v * 2 for k, v in half_spread.items()}}),
                                    train_end, val_start)
            vm_2x = metrics.summarize(va_2x)
            report += [
                f"## Verdict: {label}",
                f"- winner: {combo_str}",
                f"- train: {best_train['trades']} trades, {best_train['expectancy_r']}R, PF {best_train['profit_factor']}",
                f"- validation (spread+swap): {vm['trades']} trades, win {vm['win_rate']}%, "
                f"{vm['expectancy_r']}R (${vm['expectancy_usd']}/trade), PF {vm['profit_factor']}, "
                f"{vm['quarters_positive']}/{vm['quarters_total']} quarters+",
                f"- validation NO swap (spread only): {vm_ns.get('expectancy_r', 0)}R, PF {vm_ns.get('profit_factor', 0)}",
                f"- validation spread x2 (+swap): {vm_2x.get('expectancy_r', 0)}R, PF {vm_2x.get('profit_factor', 0)}",
                f"- gate (informational): {why}", ""]
            slack_body = [
                f"winner {combo_str}",
                f"train {best_train['expectancy_r']:+}R ({best_train['trades']}t) -> "
                f"val {vm['expectancy_r']:+}R (PF {vm['profit_factor']}, {vm['trades']}t) -> *{label}*",
                f"no-swap val {vm_ns.get('expectancy_r', 0):+}R | spread x2 {vm_2x.get('expectancy_r', 0):+}R"]

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy6_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy6_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY6]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"S5 rule on wider basket + swap, {len(instruments)} majors, daily, 0.5% risk")
    footer = f"Full detail: reports/fx_strategy6_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
