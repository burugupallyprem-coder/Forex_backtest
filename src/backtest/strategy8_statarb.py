"""Intraday statistical arbitrage on FX pairs - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED (2026-07-17, before any run). The one genuinely different class we
had not tried: MARKET-NEUTRAL relative value, not a directional bet. Take two
correlated legs (AUD_USD/NZD_USD, EUR_USD/GBP_USD, EUR_USD/USD_CHF), form the
log-spread s = log(A) - log(B), z-score it over a rolling window, and when the
spread stretches beyond z_enter, bet it reverts: short the rich leg, long the cheap
leg. Exit when it reverts to z_exit (~mean), or a stop_z blow-out, or a max_hold cap.

Honest catch (declared): a spread trade pays the spread on BOTH legs, entry AND exit
- four half-spreads per round trip. That double toll is the most likely killer, the
same wall as everything else, doubled. Reported GROSS vs NET so the cost is explicit.

Rules (declared): H1 bars (intraday granularity; positions typically revert within a
day, hard cap max_hold). No lookahead: rolling mean/std use the PRIOR window (shift).
Risk unit R = spread std at entry; size so a 1-std adverse move = 0.5% equity.
Grid (4 combos): z_enter [2.0, 2.5], lookback [50, 100]. z_exit 0, stop_z 4, fixed.
Train 2018-2022, judged once on 2023 -> now. Gate 100 trades / 0.05R / PF 1.15 /
60% quarters+. WEAK PASS labeling. Multiple-testing debt on the window applies.

Run: python -m src.backtest.strategy8_statarb
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


def split_trades(trades, train_end, val_start):
    tr = [t for t in trades if pd.to_datetime(t.date).date() <= train_end]
    va = [t for t in trades if pd.to_datetime(t.date).date() >= val_start]
    return tr, va


def simulate_pair(dfA, dfB, legA, legB, params, cfg):
    """Log-spread z-score mean-reversion between two legs. Costs on BOTH legs."""
    z_enter = float(params["z_enter"]); lookback = int(params["lookback"])
    z_exit = float(cfg.get("z_exit", 0.0)); stop_z = float(cfg.get("stop_z", 4.0))
    max_hold = int(cfg.get("max_hold", 48)); cost_off = bool(cfg.get("cost_off", False))
    a = dfA[["ts", "close"]].rename(columns={"close": "a"})
    b = dfB[["ts", "close"]].rename(columns={"close": "b"})
    m = pd.merge(a, b, on="ts").sort_values("ts").reset_index(drop=True)
    if len(m) < lookback + 5:
        return []
    av = m["a"].values.astype(float); bv = m["b"].values.astype(float)
    ts = m["ts"].astype(str).values
    s = np.log(av) - np.log(bv)
    ss = pd.Series(s)
    mean = ss.rolling(lookback).mean().shift(1).values
    std = ss.rolling(lookback).std().shift(1).values
    z = (s - mean) / std
    hsA = cfg["spread_pips"][legA] * pip_size(legA) / 2.0
    hsB = cfg["spread_pips"][legB] * pip_size(legB) / 2.0
    equity_risk = cfg["risk"]["equity"] * cfg["risk"]["risk_pct"] / 100.0
    trades = []; pos = None; n = len(m)
    for i in range(n):
        zi = z[i]
        if not np.isfinite(zi):
            continue
        if pos is None:
            if abs(zi) >= z_enter and np.isfinite(std[i]) and std[i] > 0:
                side = -1 if zi > 0 else 1                     # profit if spread reverts toward mean
                costfrac = 0.0 if cost_off else 2.0 * ((hsA / av[i]) + (hsB / bv[i]))
                pos = {"i": i, "s0": s[i], "side": side, "std": std[i], "cost": costfrac}
        else:
            held = i - pos["i"]
            revert = (pos["side"] == -1 and zi <= z_exit) or (pos["side"] == 1 and zi >= z_exit)
            blow = abs(zi) >= stop_z
            if revert or blow or held >= max_hold:
                net = pos["side"] * (s[i] - pos["s0"]) - pos["cost"]
                notional = equity_risk / pos["std"] if pos["std"] > 0 else 0.0
                trades.append(Trade(
                    symbol=f"{legA}/{legB}", strategy="strategy8_statarb",
                    date=ts[pos["i"]][:10], entry_time=ts[pos["i"]], exit_time=ts[i],
                    entry=round(pos["s0"], 6), exit=round(s[i], 6), shares=round(notional, 2),
                    stop=0.0, target=0.0, pnl=round(net * notional, 2),
                    r_multiple=round(net / pos["std"], 3) if pos["std"] > 0 else 0.0,
                    exit_reason=("stop" if blow else ("revert" if revert else "timeout")),
                    signal_reason=f"z{'hi' if pos['side'] == -1 else 'lo'}_{z_enter}"))
                pos = None
    return trades


def _sim_all(frames, leg_pairs, combo, cfg_sim):
    out = []
    for legA, legB in leg_pairs:
        if legA in frames and legB in frames:
            out.extend(simulate_pair(frames[legA], frames[legB], legA, legB, combo, cfg_sim))
    return out


def run():
    cfg = load_config()
    fx = cfg["fx_strategy8"]
    val_end = fx.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    leg_pairs = [tuple(p) for p in fx["leg_pairs"]]
    gate = fx["gate"]
    train_floor = fx.get("min_train_expectancy_r", 0.02)
    train_end = pd.to_datetime(fx["train_end"]).date()
    val_start = pd.to_datetime(fx["val_start"]).date()
    gran = fx.get("granularity", "H1")

    base = {"risk": cfg["risk"], "spread_pips": fx["spread_pips"],
            "z_exit": fx.get("z_exit", 0.0), "stop_z": fx.get("stop_z", 4.0),
            "max_hold": fx.get("max_hold", 48)}
    instruments = sorted({x for p in leg_pairs for x in p})
    frames = {}
    for inst in instruments:
        print(f"downloading {inst} {gran}, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end, granularity=gran)
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = df
    if len(frames) < 2:
        slackbot.post(f"[FX-STRATEGY8] {ts} - FAILED: not enough candles returned.")
        return

    combos = expand_grid(fx["grids"])
    results = []
    for idx, combo in enumerate(combos, 1):
        tr, va = split_trades(_sim_all(frames, leg_pairs, combo, base), train_end, val_start)
        m = metrics.summarize(tr)
        print(f"  [s8 {idx}/{len(combos)}] {combo} -> train {m.get('trades', 0)} trades, "
              f"{m.get('expectancy_r', 0)}R", flush=True)
        results.append((combo, m, va))

    report = [f"# Intraday statistical arbitrage (pairs mean-reversion) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Pre-registered 2026-07-17.",
              f"leg pairs {leg_pairs} - {gran} - market-neutral - spread charged on BOTH legs",
              f"train {fx['train_start']} -> {fx['train_end']} - validation {fx['val_start']} -> {val_end}", "",
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
            _, va_g = split_trades(_sim_all(frames, leg_pairs, best_combo, {**base, "cost_off": True}), train_end, val_start)
            vm_g = metrics.summarize(va_g)
            report += [
                f"## Verdict: {label}",
                f"- winner: {combo_str}",
                f"- train: {best_train['trades']} trades, {best_train['expectancy_r']}R, PF {best_train['profit_factor']}",
                f"- validation NET (both-leg costs): {vm['trades']} trades, win {vm['win_rate']}%, "
                f"{vm['expectancy_r']}R, PF {vm['profit_factor']}, {vm['quarters_positive']}/{vm['quarters_total']} quarters+",
                f"- validation GROSS (no costs): {vm_g.get('expectancy_r', 0)}R, PF {vm_g.get('profit_factor', 0)}",
                f"- gate (informational): {why}", ""]
            slack_body = [
                f"winner {combo_str}",
                f"train {best_train['expectancy_r']:+}R ({best_train['trades']}t) -> "
                f"val NET {vm['expectancy_r']:+}R (PF {vm['profit_factor']}, {vm['trades']}t) -> *{label}*",
                f"val GROSS {vm_g.get('expectancy_r', 0):+}R (cost is the gap)"]

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy8_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy8_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY8]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Intraday stat-arb (pairs mean-reversion), {gran}, market-neutral, both-leg costs")
    footer = f"Full detail: reports/fx_strategy8_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
