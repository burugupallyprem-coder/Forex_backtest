"""Peter's strategy #2 (from his 2026-07-16 video) - FOUR trigger families,
because the owner couldn't name which one his chart shows. RESEARCH ONLY.

PRE-REGISTERED (2026-07-16, before any run). Confirmed by Peter: both
directions; London+NY entries only; max 1-2 trades/day; RSI must confirm
(declared thresholds: long needs RSI14 <= 40, short needs RSI14 >= 60);
flat 21:00 UTC; 0.5% risk. "Figure out the best stop/target" is implemented
the only honest way: target_mode lives in the DECLARED grid, selected on
train, judged once on validation.

The four families (each judged separately, like the crypto intraday sweep):
1. sweep    - liquidity-sweep reversal: bar pierces the prev-day low/high
              by >= sweep_frac x prev-day range, then CLOSES back inside ->
              reverse next bar. Stop: sweep extreme -/+ 0.25 x ATR.
              structure target: 0.382 fib of prev-day range.
2. triangle - compression breakout: 10-bar range < contraction_frac x
              20-bar range -> close breaks the 10-bar high/low -> follow
              next bar. Stop: other side of the 10-bar range.
              structure target: measured move (compression height).
3. tline    - trendline break + retest: line through the last two CONFIRMED
              swing highs (descending) breaks, price retests it within
              20 bars and holds -> long next bar (mirror for shorts).
              Stop: retest bar extreme -/+ 0.5 x ATR. structure target:
              the second swing's extreme.
4. zone     - supply/demand tap: a confirmed swing that impulsed away by
              >= impulse_atr x ATR leaves a zone; first return into the
              zone that closes back out -> trade away next bar.
              Stop: far side of zone +/- 0.25 x ATR. structure target:
              2 x zone height from entry side.

Anti-lookahead: swings confirm 3 bars late and generators only use
confirmed ones; all entries fill at the NEXT bar's open +/- half-spread;
entry bar gets no free pass; stop before target; cost floor: stop >=
min_stop_cost_mult x round trip.

DECLARED HONESTY DEBT: these are hypothesis families #2-5 judged against
the same FX 2026 validation window (after fx_prevday). Any PASS here is
provisional until re-confirmed on data that does not exist yet (Q4 2026).
A crypto twin is deliberately NOT built: the sessions diagnostic (2026-07-16)
already prices intraday crypto stops at a 0.4-0.8R toll - running it would
burn evidence to confirm arithmetic.

Grids (declared): common max_trades_day [1, 2] and target_mode
["rr2", "structure"]; per family: sweep_frac [0.05, 0.15] /
contraction_frac [0.5, 0.65] / retest_tol_atr [0.25, 0.5] /
impulse_atr [1.5, 2.5]. 8 combos per family, 32 total. Train 2024-07 ->
2025-12, winner per family needs >= 100 train trades, judged ONCE on 2026
validation. Gate: 100 trades, 0.05R, PF 1.15, 60% quarters+. WEAK PASS
labeling. Spread x2 sensitivity for any family that passes.

Run: python -m src.backtest.strategy2_sweep
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src import oanda_data, slackbot
from src.backtest import metrics
from src.backtest.prevday_fib_fx import (FLAT_MIN, Trade, expand_grid, in_session,
                                         load_config, pip_size, prev_day_levels,
                                         size_units, split_trades)

ROOT = Path(__file__).resolve().parent.parent.parent

RSI_LONG_MAX = 40.0
RSI_SHORT_MIN = 60.0


# ---------------------------------------------------------------- indicators

def prep(df):
    df = df.reset_index(drop=True).copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["rsi"] = (100 - 100 / (1 + rs)).astype(float).fillna(50.0)
    # confirmed swings: swing at j is usable from bar j+3 onward
    k = 3
    df["swing_hi"] = (df["high"] == df["high"].rolling(2 * k + 1, center=True).max())
    df["swing_lo"] = (df["low"] == df["low"].rolling(2 * k + 1, center=True).min())
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df


def confirmed_swings(df, i, col, k=3):
    """Indices j <= i-k where a swing was confirmed by bar i."""
    upto = max(0, i - k + 1)
    flags = df[col].iloc[:upto]
    return list(flags[flags.fillna(False)].index)


# ------------------------------------------------------------- generators
# Each returns None or dict(side, stop, target_structure, reason).
# The engine applies RSI gate, target_mode, sessions, caps, cost floor.

def gen_sweep(df, i, state, params, levels):
    date = df["date"].iloc[i]
    if date not in levels:
        return None
    prev_hi, prev_lo = levels[date]
    rng = prev_hi - prev_lo
    if rng <= 0:
        return None
    frac = float(params["sweep_frac"])
    atr = df["atr"].iloc[i]
    if atr != atr:
        return None
    lo, hi, c = df["low"].iloc[i], df["high"].iloc[i], df["close"].iloc[i]
    if lo <= prev_lo - frac * rng and c > prev_lo:
        return {"side": 1, "stop": lo - 0.25 * atr,
                "t_struct": prev_lo + 0.382 * rng, "reason": "sweep_long"}
    if hi >= prev_hi + frac * rng and c < prev_hi:
        return {"side": -1, "stop": hi + 0.25 * atr,
                "t_struct": prev_hi - 0.382 * rng, "reason": "sweep_short"}
    return None


def gen_triangle(df, i, state, params, levels):
    if i < 30:
        return None
    frac = float(params["contraction_frac"])
    w10 = df.iloc[i - 10:i]        # prior 10 bars - breakout bar excluded
    w20 = df.iloc[i - 30:i - 10]
    r10 = float(w10["high"].max() - w10["low"].min())
    r20 = float(w20["high"].max() - w20["low"].min())
    if r20 <= 0 or r10 > frac * r20:
        state.pop("tri", None)
        return None
    hi10, lo10 = float(w10["high"].max()), float(w10["low"].min())
    c = float(df["close"].iloc[i])
    prev_c = float(df["close"].iloc[i - 1])
    if prev_c <= hi10 and c > hi10:
        return {"side": 1, "stop": lo10, "t_struct": c + r10,
                "reason": "tri_break_long"}
    if prev_c >= lo10 and c < lo10:
        return {"side": -1, "stop": hi10, "t_struct": c - r10,
                "reason": "tri_break_short"}
    return None


def _line_value(df, j1, j2, col, i):
    v1, v2 = float(df[col].iloc[j1]), float(df[col].iloc[j2])
    return v1 + (v2 - v1) * (i - j1) / (j2 - j1)


def gen_tline(df, i, state, params, levels):
    if i < 12:
        return None
    tol_atr = float(params["retest_tol_atr"])
    atr = df["atr"].iloc[i]
    if atr != atr or atr <= 0:
        return None
    c, lo, hi = (float(df["close"].iloc[i]), float(df["low"].iloc[i]),
                 float(df["high"].iloc[i]))

    # descending line through last two confirmed swing highs -> long setup
    shs = confirmed_swings(df, i, "swing_hi")
    if len(shs) >= 2:
        j1, j2 = shs[-2], shs[-1]
        if float(df["high"].iloc[j2]) < float(df["high"].iloc[j1]) and j2 > j1:
            v = _line_value(df, j1, j2, "high", i)
            brk = state.get("tl_up_break")
            if brk is None:
                pv = _line_value(df, j1, j2, "high", i - 1)
                if float(df["close"].iloc[i - 1]) <= pv and c > v:
                    state["tl_up_break"] = {"j1": j1, "j2": j2, "bar": i}
            else:
                if i - brk["bar"] > 20:
                    state.pop("tl_up_break", None)
                else:
                    v_b = _line_value(df, brk["j1"], brk["j2"], "high", i)
                    if lo <= v_b + tol_atr * atr and c > v_b:
                        state.pop("tl_up_break", None)
                        return {"side": 1, "stop": lo - 0.5 * atr,
                                "t_struct": float(df["high"].iloc[brk["j2"]]),
                                "reason": "tline_long"}

    # ascending line through last two confirmed swing lows -> short setup
    sls = confirmed_swings(df, i, "swing_lo")
    if len(sls) >= 2:
        j1, j2 = sls[-2], sls[-1]
        if float(df["low"].iloc[j2]) > float(df["low"].iloc[j1]) and j2 > j1:
            v = _line_value(df, j1, j2, "low", i)
            brk = state.get("tl_dn_break")
            if brk is None:
                pv = _line_value(df, j1, j2, "low", i - 1)
                if float(df["close"].iloc[i - 1]) >= pv and c < v:
                    state["tl_dn_break"] = {"j1": j1, "j2": j2, "bar": i}
            else:
                if i - brk["bar"] > 20:
                    state.pop("tl_dn_break", None)
                else:
                    v_b = _line_value(df, brk["j1"], brk["j2"], "low", i)
                    if hi >= v_b - tol_atr * atr and c < v_b:
                        state.pop("tl_dn_break", None)
                        return {"side": -1, "stop": hi + 0.5 * atr,
                                "t_struct": float(df["low"].iloc[brk["j2"]]),
                                "reason": "tline_short"}
    return None


def gen_zone(df, i, state, params, levels):
    imp = float(params["impulse_atr"])
    atr_i = df["atr"].iloc[i]
    if atr_i != atr_i or atr_i <= 0 or i < 10:
        return None
    c, lo, hi = (float(df["close"].iloc[i]), float(df["low"].iloc[i]),
                 float(df["high"].iloc[i]))

    # refresh zones from confirmed swings with an impulse away
    for col, key in (("swing_hi", "supply"), ("swing_lo", "demand")):
        for j in confirmed_swings(df, i, col)[-3:]:
            if j + 3 > i:
                continue
            atr_j = df["atr"].iloc[j]
            if atr_j != atr_j or atr_j <= 0:
                continue
            z_lo, z_hi = float(df["low"].iloc[j]), float(df["high"].iloc[j])
            if key == "supply" and float(df["close"].iloc[j + 3]) <= z_hi - imp * atr_j:
                state["supply"] = {"lo": z_lo, "hi": z_hi, "j": j}
            if key == "demand" and float(df["close"].iloc[j + 3]) >= z_lo + imp * atr_j:
                state["demand"] = {"lo": z_lo, "hi": z_hi, "j": j}

    sup = state.get("supply")
    if sup and i > sup["j"] + 3 and hi >= sup["lo"] and c < sup["lo"]:
        state.pop("supply", None)
        h = sup["hi"] - sup["lo"]
        return {"side": -1, "stop": sup["hi"] + 0.25 * atr_i,
                "t_struct": sup["lo"] - 2 * max(h, 0.5 * atr_i),
                "reason": "zone_short"}
    dem = state.get("demand")
    if dem and i > dem["j"] + 3 and lo <= dem["hi"] and c > dem["hi"]:
        state.pop("demand", None)
        h = dem["hi"] - dem["lo"]
        return {"side": 1, "stop": dem["lo"] - 0.25 * atr_i,
                "t_struct": dem["hi"] + 2 * max(h, 0.5 * atr_i),
                "reason": "zone_long"}
    return None


GENERATORS = {"sweep": gen_sweep, "triangle": gen_triangle,
              "tline": gen_tline, "zone": gen_zone}
FAMILY_GRIDS = {
    "sweep": {"sweep_frac": [0.05, 0.15]},
    "triangle": {"contraction_frac": [0.5, 0.65]},
    "tline": {"retest_tol_atr": [0.25, 0.5]},
    "zone": {"impulse_atr": [1.5, 2.5]},
}


# ------------------------------------------------------------------ engine

def simulate(df, instrument, family, params, cfg):
    """Shared engine: RSI gate, sessions, caps, cost floor, honest fills."""
    gen = GENERATORS[family]
    max_td = int(params["max_trades_day"])
    target_mode = params["target_mode"]
    min_mult = float(cfg.get("min_stop_cost_mult", 2.0))
    hs = float(cfg["half_spread"][instrument])
    levels = prev_day_levels(df)

    trades = []
    pos = None
    pending = None
    state = {}
    cur_date, trades_today = None, 0
    n = len(df)

    def close_trade(i, exit_px, reason):
        row = df.iloc[i]
        factor = (1.0 / pos["entry"]) if instrument.endswith("JPY") else 1.0
        gross = (exit_px - pos["entry"]) * pos["shares"] * pos["side"] * factor
        r = (exit_px - pos["entry"]) * pos["side"] / pos["risk_ps"]
        trades.append(Trade(
            symbol=instrument, strategy=f"s2_{family}",
            date=pos["entry_date"], entry_time=pos["entry_time"],
            exit_time=str(row["ts"]), entry=round(pos["entry"], 6),
            exit=round(exit_px, 6), shares=pos["shares"],
            stop=round(pos["stop"], 6), target=round(pos["target"], 6),
            pnl=round(gross, 2), r_multiple=round(r, 3),
            exit_reason=reason, signal_reason=pos["reason"]))

    for i in range(n):
        row = df.iloc[i]
        date = row["date"]
        minute = int(row["minute"])
        if date != cur_date:
            cur_date, trades_today = date, 0
            pending = None

        if pos is not None and minute >= FLAT_MIN:
            px = float(row["open"]) - pos["side"] * hs
            close_trade(i, px, "flat_2100")
            pos = None

        if pos is None and pending is not None and minute < FLAT_MIN and in_session(minute):
            side = pending["side"]
            entry_px = float(row["open"]) + side * hs
            stop = pending["stop"]
            risk_ps = (entry_px - stop) * side
            if target_mode == "rr2":
                target = entry_px + side * 2.0 * risk_ps
            else:
                target = pending["t_struct"]
            good = (risk_ps > 0 and (target - entry_px) * side > 0
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

        if (pos is None and trades_today < max_td and in_session(minute)
                and i < n - 1):
            sig = gen(df, i, state, params, levels)
            if sig is not None:
                rsi = float(df["rsi"].iloc[i])
                if ((sig["side"] == 1 and rsi <= RSI_LONG_MAX)
                        or (sig["side"] == -1 and rsi >= RSI_SHORT_MIN)):
                    pending = sig

    if pos is not None and n > 0:
        row = df.iloc[n - 1]
        px = float(row["close"]) - pos["side"] * hs
        close_trade(n - 1, px, "data_end")
    return trades


# -------------------------------------------------------------------- run

def run():
    cfg = load_config()
    s2 = cfg["fx_strategy2"]
    val_end = s2.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    instruments = s2["instruments"]
    gate = s2["gate"]
    train_floor = s2.get("min_train_expectancy_r", 0.02)
    train_end = pd.to_datetime(s2["train_end"]).date()
    val_start = pd.to_datetime(s2["val_start"]).date()
    half_spread = {inst: float(s2["spread_pips"][inst]) * pip_size(inst) / 2.0
                   for inst in instruments}
    cfg_sim = {"risk": cfg["risk"], "half_spread": half_spread,
               "min_stop_cost_mult": s2.get("min_stop_cost_mult", 2.0)}

    frames = {}
    for inst in instruments:
        print(f"downloading {inst} M15, {s2['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, s2["train_start"], val_end, granularity="M15")
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = prep(df)
    if not frames:
        slackbot.post(f"[FX-STRATEGY2] {ts} - FAILED: no candles returned.")
        return

    report = [f"# Strategy #2 sweep (4 trigger families) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Owner rules: both directions,",
              "London+NY, max 1-2 trades/day, RSI confirm (<=40 long / >=60 short).",
              "DECLARED DEBT: families #2-5 on the already-used FX 2026 window -",
              "any PASS is provisional until re-confirmed on Q4 data.",
              f"instruments {instruments} - spreads {s2['spread_pips']} - 0.5% risk",
              f"train {s2['train_start']} -> {s2['train_end']} - "
              f"validation {s2['val_start']} -> {val_end}", ""]
    slack_lines = []

    for family in ("sweep", "triangle", "tline", "zone"):
        grid = {**FAMILY_GRIDS[family],
                "target_mode": ["rr2", "structure"],
                "max_trades_day": [1, 2]}
        combos = expand_grid(grid)
        results = []
        for idx, combo in enumerate(combos, 1):
            all_trades = []
            for inst, df in frames.items():
                all_trades.extend(simulate(df, inst, family, combo, cfg_sim))
            tr, va = split_trades(all_trades, train_end, val_start)
            m = metrics.summarize(tr)
            print(f"  [{family} {idx}/{len(combos)}] {combo} -> "
                  f"train {m.get('trades', 0)}t {m.get('expectancy_r', 0)}R", flush=True)
            results.append((combo, m, va))

        report += [f"## Family: {family}", ""]
        for combo, m, _ in results:
            cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
            report.append(f"- {cs}: {m.get('trades', 0)}t, "
                          f"{m.get('expectancy_r', 0)}R, PF {m.get('profit_factor', 0)}")
        eligible = [r for r in results if r[1].get("trades", 0) >= s2["min_train_trades"]]
        if not eligible:
            report += ["", f"### {family}: SKIP - no combo reached "
                       f"{s2['min_train_trades']} train trades", ""]
            slack_lines.append(f"{family}: SKIP (too few trades)")
            continue
        eligible.sort(key=lambda r: r[1]["expectancy_r"], reverse=True)
        best_combo, best_train, best_va = eligible[0]
        vm = metrics.summarize(best_va)
        combo_str = ", ".join(f"{k}={v}" for k, v in sorted(best_combo.items()))
        if vm.get("trades", 0) == 0:
            report += ["", f"### {family}: FAIL - 0 validation trades", ""]
            slack_lines.append(f"{family}: FAIL (0 val trades)")
            continue
        verdict, why = metrics.gate_verdict(vm, gate)
        weak = verdict == "PASS" and best_train["expectancy_r"] < train_floor
        label = "WEAK PASS" if weak else verdict
        report += ["", f"### {family}: {label}",
                   f"- winner: {combo_str}",
                   f"- train: {best_train['trades']}t, {best_train['expectancy_r']}R, "
                   f"PF {best_train['profit_factor']}",
                   f"- validation: {vm['trades']}t, win {vm['win_rate']}%, "
                   f"{vm['expectancy_r']}R, PF {vm['profit_factor']}, "
                   f"{vm['quarters_positive']}/{vm['quarters_total']} quarters+",
                   f"- gate: {why}", ""]
        slack_lines.append(
            f"{family}: train {best_train['expectancy_r']:+}R ({best_train['trades']}t) -> "
            f"val {vm['expectancy_r']:+}R (PF {vm['profit_factor']}, {vm['trades']}t) -> *{label}*")

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy2_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy2_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY2]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"4 trigger families (sweep/triangle/tline/zone), {instruments}, "
              "RSI confirm, London+NY, 0.5% risk\n"
              "declared: families 2-5 on this window - any PASS is provisional "
              "until Q4 re-confirmation")
    footer = f"Full detail: reports/fx_strategy2_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_lines + [footer]))


if __name__ == "__main__":
    run()
