"""Gold ORB-momentum (strategy #11) - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED 2026-07-22. A DISTINCT gold-intraday family from the failed
break-and-retest scalp: a NY opening-range breakout that goes for BIG targets
(2-3R) so gold's ~0.35 spread is a small fraction of each winner. The design
principle is fewer, larger captures - the only way an intraday edge can clear a
fat cost on gold. Honest expectation: prior gold-intraday tests all died on
cost/regime, so this is a fair shot, not a promise.

Rules (declared): opening range = first `or_minutes` 1-min bars from the NY open
(13:30 UTC). After the range is set, the FIRST 1-min CLOSE beyond it arms a
breakout (long above OR high, short below OR low), filled at the NEXT bar open
+/- half-spread (entry bar gets no free pass). Stop = the opposite side of the
opening range (R = range width). Target = target_r x R, fixed (no trailing - a
clean R test). Optional daily 50/200 EMA trend filter. Volatility floor: skip
days whose opening range < min_or_width_frac of price (dead tape). No new entries
after entry_cutoff; one trade/day; flat 21:00 UTC. Stop checked before target in
the same bar; gaps fill at the open on the bad side; 0.5% risk; cost floor skips
setups whose stop < min_stop_cost_mult x round trip.

VALIDATION (same honesty as the gold floor tune): SELECT the grid winner on the
already-spent window (2024-07 -> 2025-12), judge ONCE on the UNTOUCHED holdout
(<= 2024-06-30, back to history_start), with gate + walk-forward + a bootstrap CI
on holdout expectancy_R + gross-vs-net. 2026 shown as a contaminated reference
only. Grid (8 combos): or_minutes [15,30] x target_r [2.0,3.0] x trend_filter
[false,true].

Run: python -m src.backtest.strategy11_gold_orb
"""

import itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import oanda_data, slackbot
from src.backtest import metrics
from src.backtest.strategy10_boring_scalp import (Trade, daily_ema_dir,
                                                  opening_ranges, size_units,
                                                  usd_pnl_factor)
from src.backtest.strategy10_gold import between, bootstrap_ci, walk_forward

ROOT = Path(__file__).resolve().parent.parent.parent


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def expand_grid(grid):
    keys = sorted(grid.keys())
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[k] for k in keys))]


def simulate_orb(df, inst, params, cfg):
    """NY opening-range breakout, fixed R target, one trade/day, for XAU_USD."""
    or_min = int(params["or_minutes"])
    target_r = float(params["target_r"])
    use_trend = bool(params["trend_filter"])
    hs = float(cfg["half_spread"][inst])
    ny_open = int(cfg["ny_open_min"])
    cutoff = int(cfg["entry_cutoff_min"])
    flat_min = int(cfg["flat_min"])
    floor = cfg.get("min_or_width_frac")
    min_mult = float(cfg.get("min_stop_cost_mult", 2.0))

    ema = daily_ema_dir(df, int(cfg["ema_fast"]), int(cfg["ema_slow"]))
    orl = opening_ranges(df, ny_open, or_min)
    df = df.reset_index(drop=True)
    n = len(df)

    trades = []
    pos = None
    pending = None
    cur_date = None
    trades_today = 0
    day_dir = 0
    day_blocked = False

    def close_trade(row, exit_px, reason):
        d = pos["side"]
        risk_ps = abs(pos["entry"] - pos["stop"])
        f = usd_pnl_factor(inst, pos["entry"])
        pnl = (exit_px - pos["entry"]) * pos["shares"] * d * f
        trades.append(Trade(
            symbol=inst, strategy="strategy11_gold_orb", date=str(row["ts"].date()),
            entry_time=pos["entry_time"], exit_time=str(row["ts"].time()),
            entry=round(pos["entry"], 4), exit=round(exit_px, 4), shares=pos["shares"],
            stop=round(pos["stop"], 4), target=round(pos["target"], 4), pnl=round(pnl, 2),
            r_multiple=round((exit_px - pos["entry"]) * d / risk_ps, 3) if risk_ps > 0 else 0.0,
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
            day_dir = ema.get(date, 0)
            day_blocked = False
            if floor and date in orl:
                orh, orlo = orl[date]
                px = c if c > 0 else 1.0
                day_blocked = (orh - orlo) / px < float(floor)

        # flat cutoff
        if pos is not None and minute >= flat_min:
            close_trade(row, o - pos["side"] * hs, "flat_2100")
            pos = None

        # pending fills at this bar open
        if pos is None and pending is not None and minute < flat_min:
            side = pending["side"]
            entry_px = o + side * hs
            stop = pending["stop"]
            risk_ps = (entry_px - stop) * side
            if risk_ps > 0 and risk_ps >= min_mult * 2 * hs:
                units = size_units(entry_px, stop, inst, cfg)
                if units > 0:
                    target = entry_px + side * target_r * risk_ps
                    pos = {"side": side, "entry": entry_px, "stop": stop, "target": target,
                           "shares": units, "entry_time": str(row["ts"].time()),
                           "reason": pending["reason"]}
                    trades_today += 1
        pending = None

        # manage: stop before target (entry bar included)
        if pos is not None:
            d = pos["side"]
            if d == 1 and lo_b <= pos["stop"]:
                close_trade(row, min(o, pos["stop"]) - hs, "stop")
                pos = None
            elif d == 1 and hi_b >= pos["target"]:
                close_trade(row, max(o, pos["target"]) - hs, "target")
                pos = None
            elif d == -1 and hi_b >= pos["stop"]:
                close_trade(row, max(o, pos["stop"]) + hs, "stop")
                pos = None
            elif d == -1 and lo_b <= pos["target"]:
                close_trade(row, min(o, pos["target"]) + hs, "target")
                pos = None

        # signal: first breakout close beyond the opening range
        or_ready = minute >= ny_open + or_min
        if (pos is None and pending is None and not day_blocked and trades_today < 1
                and or_ready and ny_open <= minute < cutoff and date in orl
                and i < n - 1 and df.iloc[i + 1]["ts"].strftime("%Y-%m-%d") == date):
            orh, orlo = orl[date]
            if orh > orlo:
                if c > orh and (not use_trend or day_dir == 1):
                    pending = {"side": 1, "stop": orlo, "reason": "orb_break_long"}
                elif c < orlo and (not use_trend or day_dir == -1):
                    pending = {"side": -1, "stop": orh, "reason": "orb_break_short"}

    if pos is not None and n > 0:
        row = df.iloc[n - 1]
        close_trade(row, float(row["close"]) - pos["side"] * hs, "data_end")
    return trades


def sim_cfg(g, half_spread):
    return {"risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
            "half_spread": {g["instrument"]: half_spread},
            "min_stop_cost_mult": g.get("min_stop_cost_mult", 2.0),
            "ny_open_min": g["ny_open_min"], "entry_cutoff_min": g["entry_cutoff_min"],
            "flat_min": g["flat_min"], "ema_fast": g["ema_fast"], "ema_slow": g["ema_slow"],
            "min_or_width_frac": g["min_or_width_frac"]}


def run():
    cfg = load_config()
    g = cfg["fx_strategy11_gold_orb"]
    inst = g["instrument"]
    hs = float(g["spread_price"]) / 2.0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    end = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    sel_lo = pd.to_datetime(g["select_start"]).date()
    sel_hi = pd.to_datetime(g["select_end"]).date()
    hold_hi = pd.to_datetime(g["holdout_end"]).date()
    gate = g["gate"]

    print(f"downloading {inst} M1, {g['history_start']} -> {end}", flush=True)
    df = oanda_data.fetch_candles(inst, g["history_start"], end, granularity=g["granularity"])
    if df.empty:
        slackbot.post(f"[FX-STRATEGY11-GOLDORB] {ts} - FAILED: no candles.")
        return
    first, last = df["ts"].min().date(), df["ts"].max().date()
    print(f"  {inst}: {len(df):,} bars, {first} -> {last}", flush=True)

    combos = expand_grid(g["grid"])
    scored = []
    trades_by = {}
    for combo in combos:
        tr = simulate_orb(df, inst, combo, sim_cfg(g, hs))
        trades_by[repr(sorted(combo.items()))] = tr
        ms = metrics.summarize(between(tr, sel_lo, sel_hi))
        scored.append((combo, ms))
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        print(f"  {cs}: select {ms.get('trades',0)}t {ms.get('expectancy_r',0)}R", flush=True)

    report = [f"# Gold ORB-momentum (strategy #11) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Distinct family: NY opening-range breakout, "
              "big R targets; judged on UNTOUCHED holdout.",
              f"data {first} -> {last} - select {g['select_start']}..{g['select_end']} - "
              f"holdout <= {g['holdout_end']}", "", "## Selection grid (spent window)", ""]
    for combo, m in scored:
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        report.append(f"- {cs}: {m.get('trades',0)} trades, {m.get('expectancy_r',0)}R, "
                      f"PF {m.get('profit_factor',0)}")
    report.append("")

    eligible = [(c, m) for c, m in scored if m.get("trades", 0) >= g["min_select_trades"]]
    if not eligible:
        line = f"SKIP - no combo reached {g['min_select_trades']} selection trades"
        report += [f"## Verdict: {line}", ""]
        slack_body = [line]
    else:
        eligible.sort(key=lambda x: x[1]["expectancy_r"], reverse=True)
        win, sel_m = eligible[0]
        cs = ", ".join(f"{k}={v}" for k, v in sorted(win.items()))
        wt = trades_by[repr(sorted(win.items()))]
        hold = between(wt, hi=hold_hi)
        hm = metrics.summarize(hold)
        gross = simulate_orb(df, inst, win, sim_cfg(g, 0.0))
        gm = metrics.summarize(between(gross, hi=hold_hi))
        ref = metrics.summarize(between(wt, lo=pd.to_datetime("2026-01-01").date()))
        if hm.get("trades", 0) == 0:
            line = f"FAIL - winner {cs} produced 0 holdout trades"
            report += [f"## Verdict: {line}", ""]
            slack_body = [line]
        else:
            verdict, why = metrics.gate_verdict(hm, gate)
            wf_pos, wf_tot, wf_per = walk_forward(hold, int(g["walkforward_folds"]))
            if verdict == "PASS" and not (wf_tot and wf_pos / wf_tot >= float(g["min_positive_frac"])):
                verdict = "FAIL"
                why = (f"{why}; " if why != "all gate checks met" else "") + \
                      f"walk-forward only {wf_pos}/{wf_tot} folds+"
            lo, hi, frac_pos, pt = bootstrap_ci(hold)
            report += [
                f"## Verdict: {verdict}",
                f"- winner: {cs}",
                f"- selection: {sel_m['trades']} trades, {sel_m['expectancy_r']}R, PF {sel_m['profit_factor']}",
                f"- HOLDOUT (untouched judge): {hm['trades']} trades, win {hm['win_rate']}%, "
                f"{hm['expectancy_r']}R (${hm['expectancy_usd']}/trade), PF {hm['profit_factor']}, "
                f"{hm['quarters_positive']}/{hm['quarters_total']} quarters+, maxDD ${hm['max_drawdown']:,}",
                f"- holdout GROSS (0 spread): {gm.get('trades',0)} trades, {gm.get('expectancy_r',0)}R "
                f"(spread eats {gm.get('expectancy_r',0)-hm['expectancy_r']:+.3f}R)",
                f"- bootstrap holdout 90% CI: [{lo:+.3f}R, {hi:+.3f}R], P(>0)={frac_pos*100:.1f}% "
                f"-> CI {'clears 0' if lo > 0 else 'includes 0'}",
                f"- walk-forward: {wf_pos}/{wf_tot} folds+ (per-fold R: {', '.join(f'{r:+.3f}' for r in wf_per)})",
                f"- 2026 reference (CONTAMINATED, not a gate): {ref.get('trades',0)} trades, "
                f"{ref.get('expectancy_r',0)}R, PF {ref.get('profit_factor',0)}",
                f"- gate: {why}", ""]
            slack_body = [
                f"winner {cs}",
                f"holdout {hm['expectancy_r']:+}R (PF {hm['profit_factor']}, {hm['trades']}t, "
                f"{hm['quarters_positive']}/{hm['quarters_total']}q+) -> *{verdict}*",
                f"bootstrap 90% CI [{lo:+.3f},{hi:+.3f}]R P(>0)={frac_pos*100:.0f}% | "
                f"WF {wf_pos}/{wf_tot} | gross {gm.get('expectancy_r',0):+}R"]

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy11goldorb_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy11goldorb_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY11-GOLDORB]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Gold ORB-momentum (big-target breakout), judged on UNTOUCHED holdout <= {g['holdout_end']}")
    footer = f"Full detail: reports/fx_strategy11goldorb_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
