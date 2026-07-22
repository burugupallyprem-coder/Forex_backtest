"""Peter's 3-step "boring" scalp on FOREX - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED (2026-07-22, rules from Peter's own infographic, modeling
choices confirmed by him before this run). The system, mechanized:

  1. DIRECTION - daily 9/21 EMA. Uptrend = prev daily close > EMA9 > EMA21
     (longs only); downtrend = prev close < EMA9 < EMA21 (shorts only);
     anything else = range. `trend_filter` grid knob decides whether range
     days are skipped (true) or traded in the breakout's own direction
     (false - Peter chose to ALSO trade ranges, so false is in the grid).
  2. SETUP - previous-day High/Low turned into small ZONES (half-width
     zone_frac x prev-day range, the "wiggle room"), PLUS the first
     5-minute opening range at the NY open (13:30 UTC). Four battleground
     levels/day: PDH-zone, PDL-zone, OR-high, OR-low.
  3. ENTRY - break-and-retest on the 1-minute chart. A 1m CLOSE clears the
     zone (breakout), price RETESTS the zone (a later bar trades back to
     it), then a strong 1m candle CLOSES back through in-direction
     (trigger). Fill NEXT bar open +/- half-spread (entry bar gets no free
     pass).
  4. EXIT - Peter picked a STRUCTURE/TRAILING stop (ride it to premarket/
     ATH-style targets, which aren't mechanical). Initial stop sits behind
     the broken level (stop_buf_frac x prev-day range). It then trails
     behind the swing low/high = extreme of the last `trail_lookback` 1m
     bars, ratcheting one way only. Flat by 21:00 UTC. Max 1-2 trades/day.

WHY FOREX / WHY HONEST: same discipline as the rest of this repo. Fills at
next-bar open +/- half-spread; stop checked before anything else each bar
(conservative); gaps fill at the open on the bad side; cost floor skips
setups whose initial stop < min_stop_cost_mult x round-trip cost; fixed
$100k equity, 0.5% risk sizing, 20% notional cap, no compounding. JPY-quote
PnL converted to USD at entry price (declared approximation). Sessions use
fixed UTC windows (DST drift accepted and declared). PDH/PDL for Monday come
from the previous TRADING day in the data (Friday). The trailing stop has
NO fixed target, so there is no target_r knob - "let it run" is modeled by
the swing-trail, a declared stand-in for "premarket high / all-time high".

Grid (8 combos, declared): trail_lookback [10, 20], trend_filter
[false, true], max_trades_day [1, 2]. Train 2024-07 -> 2025-12, winner on
train (>=100 trades), judged ONCE on untouched 2026 validation. Gate: 100
trades, 0.05R, PF 1.15, 60% quarters+. WEAK PASS labeling. Spread x2
sensitivity. NOTE: 2026 FX validation window carries multiple-testing debt
(prior fx_* studies used it); any PASS is provisional, declared honestly.

Run: python -m src.backtest.strategy10_boring_scalp
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


def daily_ema_dir(df, fast, slow):
    """direction per date in {+1,-1,0}, known at that day's open (prior days only)."""
    d = df["ts"].dt.strftime("%Y-%m-%d")
    daily = df.groupby(d)["close"].last()
    ema_f = daily.ewm(span=fast, adjust=False).mean()
    ema_s = daily.ewm(span=slow, adjust=False).mean()
    pc, pf, ps = daily.shift(1), ema_f.shift(1), ema_s.shift(1)
    out = {}
    for date in daily.index:
        a, b, c = pc[date], pf[date], ps[date]
        if pd.isna(a) or pd.isna(b) or pd.isna(c):
            out[date] = 0
        elif a > b and b > c:
            out[date] = 1
        elif a < b and b < c:
            out[date] = -1
        else:
            out[date] = 0
    return out


def prev_day_levels(df):
    """{date: (prev_hi, prev_lo)} from the previous trading day present in data."""
    d = df["ts"].dt.strftime("%Y-%m-%d")
    g = df.groupby(d).agg(hi=("high", "max"), lo=("low", "min"))
    g["prev_hi"] = g["hi"].shift(1)
    g["prev_lo"] = g["lo"].shift(1)
    return {idx: (row["prev_hi"], row["prev_lo"])
            for idx, row in g.iterrows() if row["prev_hi"] == row["prev_hi"]}


def opening_ranges(df, ny_open, or_min):
    """{date: (or_high, or_low)} from the first or_min minutes at/after NY open."""
    minute = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    mask = (minute >= ny_open) & (minute < ny_open + or_min)
    if not mask.any():
        return {}
    d = df["ts"].dt.strftime("%Y-%m-%d")
    sub = df[mask]
    g = sub.groupby(d[mask]).agg(orh=("high", "max"), orl=("low", "min"))
    return {idx: (row["orh"], row["orl"]) for idx, row in g.iterrows()}


def simulate_instrument(df, instrument, params, cfg):
    """1-minute break-and-retest with swing-trailing stop for one FX instrument."""
    tl = int(params["trail_lookback"])
    use_trend = bool(params["trend_filter"])
    max_td = int(params["max_trades_day"])
    min_mult = float(cfg.get("min_stop_cost_mult", 2.0))
    hs = float(cfg["half_spread"][instrument])
    ny_open = int(cfg["ny_open_min"])
    or_min = int(cfg["or_minutes"])
    flat_min = int(cfg["flat_min"])
    zone_frac = float(cfg["zone_frac"])
    buf_frac = float(cfg["stop_buf_frac"])

    ema_dir = daily_ema_dir(df, int(cfg["ema_fast"]), int(cfg["ema_slow"]))
    pdlevels = prev_day_levels(df)
    orlevels = opening_ranges(df, ny_open, or_min)

    trades = []
    pos = None
    pending = None
    cur_date = None
    trades_today = 0
    day_dir = 0
    rng = 0.0
    zone = 0.0
    buf = 0.0
    state = {}          # level-key -> {"broke":bool,"retested":bool,"used":bool}
    n = len(df)

    def close_trade(i, exit_px, reason):
        row = df.iloc[i]
        f = usd_pnl_factor(instrument, pos["entry"])
        gross = (exit_px - pos["entry"]) * pos["shares"] * pos["side"] * f
        r = (exit_px - pos["entry"]) * pos["side"] / pos["risk_ps"]
        trades.append(Trade(
            symbol=instrument, strategy="strategy10_boring_scalp",
            date=pos["entry_date"], entry_time=pos["entry_time"],
            exit_time=str(row["ts"]), entry=round(pos["entry"], 6),
            exit=round(exit_px, 6), shares=pos["shares"],
            stop=round(pos["stop_init"], 6), target=0.0,
            pnl=round(gross, 2), r_multiple=round(r, 3),
            exit_reason=reason, signal_reason=pos["reason"]))

    for i in range(n):
        row = df.iloc[i]
        date = row["ts"].strftime("%Y-%m-%d")
        minute = row["ts"].hour * 60 + row["ts"].minute
        o, hi_b, lo_b, c = (float(row["open"]), float(row["high"]),
                            float(row["low"]), float(row["close"]))

        if date != cur_date:
            cur_date = date
            trades_today = 0
            pending = None
            day_dir = ema_dir.get(date, 0)
            state = {}
            if date in pdlevels:
                phi, plo = pdlevels[date]
                rng = phi - plo
                zone = zone_frac * rng if rng > 0 else 0.0
                buf = buf_frac * rng if rng > 0 else 0.0
            else:
                rng, zone, buf = 0.0, 0.0, 0.0

        # 1) flat cutoff
        if pos is not None and minute >= flat_min:
            px = o - pos["side"] * hs
            close_trade(i, px, "flat_2100")
            pos = None

        # 2) pending entry fills at THIS bar open
        if pos is None and pending is not None and minute < flat_min:
            side = pending["side"]
            entry_px = o + side * hs
            stop = pending["stop"]
            risk_ps = (entry_px - stop) * side
            good = risk_ps > 0 and risk_ps >= min_mult * 2 * hs
            if good:
                units = size_units(entry_px, stop, instrument, cfg)
                if units > 0:
                    pos = {"side": side, "entry": entry_px, "stop": stop,
                           "stop_init": stop, "risk_ps": risk_ps,
                           "buf": pending["buf"], "shares": units,
                           "entry_date": date, "entry_time": str(row["ts"]),
                           "reason": pending["reason"], "exit_label": "stop",
                           "win_low": [], "win_high": []}
                    trades_today += 1
        pending = None

        # 3) manage open position - stop checked first (no free pass on entry bar)
        if pos is not None:
            side = pos["side"]
            if side == 1 and lo_b <= pos["stop"]:
                close_trade(i, min(o, pos["stop"]) - hs, pos["exit_label"])
                pos = None
            elif side == -1 and hi_b >= pos["stop"]:
                close_trade(i, max(o, pos["stop"]) + hs, pos["exit_label"])
                pos = None
            if pos is not None:                      # ratchet the trailing stop
                pos["win_low"].append(lo_b)
                pos["win_high"].append(hi_b)
                if len(pos["win_low"]) >= tl:
                    if pos["side"] == 1:
                        new_stop = min(pos["win_low"][-tl:]) - pos["buf"]
                        if new_stop > pos["stop"]:
                            pos["stop"] = new_stop
                            pos["exit_label"] = "trail_stop"
                    else:
                        new_stop = max(pos["win_high"][-tl:]) + pos["buf"]
                        if new_stop < pos["stop"]:
                            pos["stop"] = new_stop
                            pos["exit_label"] = "trail_stop"

        # 4) signal on THIS close -> pending for next bar
        in_ny = ny_open <= minute < flat_min
        or_ready = minute >= ny_open + or_min
        if (pos is None and in_ny and trades_today < max_td and rng > 0
                and i < n - 1):
            levels = [("PDH", pdlevels[date][0], 1), ("PDL", pdlevels[date][1], -1)]
            if or_ready and date in orlevels:
                orh, orl = orlevels[date]
                levels += [("ORH", orh, 1), ("ORL", orl, -1)]

            for key, L, lside in levels:
                if use_trend and lside != day_dir:
                    continue
                st = state.setdefault(key, {"broke": False, "retested": False, "used": False})
                if st["used"]:
                    continue
                if lside == 1:
                    band = L + zone
                    if not st["broke"]:
                        if c > band:
                            st["broke"] = True
                    elif not st["retested"]:
                        if lo_b <= band:
                            st["retested"] = True
                    elif c > band and c > o:
                        pending = {"side": 1, "stop": L - buf, "buf": buf,
                                   "reason": f"{key.lower()}_break_retest_long"}
                        st["used"] = True
                        break
                else:
                    band = L - zone
                    if not st["broke"]:
                        if c < band:
                            st["broke"] = True
                    elif not st["retested"]:
                        if hi_b >= band:
                            st["retested"] = True
                    elif c < band and c < o:
                        pending = {"side": -1, "stop": L + buf, "buf": buf,
                                   "reason": f"{key.lower()}_break_retest_short"}
                        st["used"] = True
                        break

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
    fx = cfg["fx_strategy10"]
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
               "ny_open_min": fx["ny_open_min"], "or_minutes": fx["or_minutes"],
               "flat_min": fx["flat_min"], "ema_fast": fx["ema_fast"],
               "ema_slow": fx["ema_slow"], "zone_frac": fx["zone_frac"],
               "stop_buf_frac": fx["stop_buf_frac"]}

    frames = {}
    for inst in instruments:
        print(f"downloading {inst} M1, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end,
                                      granularity=fx.get("granularity", "M1"))
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[inst] = df
    if not frames:
        slackbot.post(f"[FX-STRATEGY10] {ts} - FAILED: no candles returned.")
        return

    combos = expand_grid(fx["grids"])
    results = []
    for idx, combo in enumerate(combos, 1):
        all_trades = []
        for inst, df in frames.items():
            all_trades.extend(simulate_instrument(df, inst, combo, cfg_sim))
        tr, va = split_trades(all_trades, train_end, val_start)
        m = metrics.summarize(tr)
        print(f"  [s10 {idx}/{len(combos)}] {combo} -> train {m.get('trades', 0)} trades, "
              f"{m.get('expectancy_r', 0)}R", flush=True)
        results.append((combo, m, va))

    report = [f"# 3-step boring scalp (Peter's strategy #10) on FOREX - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Rules from Peter's infographic; "
              "modeling confirmed by him 2026-07-22.",
              f"instruments {instruments} - spreads (pips): {fx['spread_pips']} - "
              "1m entries, daily 9/21 EMA direction, PDH/PDL zones + 5m NY opening "
              "range, break-and-retest, swing-trailing stop, 0.5% risk, flat 21:00 UTC",
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
    (out_dir / f"fx_strategy10_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy10_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY10]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Peter's 3-step boring scalp, {instruments}, 1m break-and-retest, "
              "9/21 EMA dir + PDH/PDL + 5m OR, swing-trail, 0.5% risk")
    footer = f"Full detail: reports/fx_strategy10_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
