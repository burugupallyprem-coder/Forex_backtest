"""Filtered ORB ported from the stock champion to an FX MAJORS BASKET - RESEARCH ONLY.

PRE-REGISTERED 2026-07-22. The stock ORB's edge lives in its FILTERS, not the
bare breakout, so this port reproduces all three on a 7-pair USD-majors basket:

  1. SESSION-ANCHORED opening range - FX has no equity open, so we anchor to a
     session open (London 07:00 UTC or NY 13:30 UTC, grid) and take the first
     `open_bars` 5-min bars as the range.
  2. USD-STRENGTH REGIME (the SPY analog) - build a USD index from the basket:
     each pair's opening move is sign-flipped to "USD direction" (USD-base pairs
     +, USD-quote pairs -) and averaged. A trade must align with the USD regime
     (regime_filter=true), i.e. don't fight the dollar tape.
  3. RELATIVE STRENGTH (rs_topk, the biggest edge driver on stocks) - rank the
     pairs by how strongly they moved in the USD-regime direction this session
     and trade only the top K. A single pair has no cross-section, which is why
     the basket is essential.
  Plus the VOLATILITY FLOOR (skip dead-tape opens: OR width < min_or_width_frac
  of price).

Entry: first 5-min CLOSE beyond the opening range (long above / short below) in a
permitted direction, filled NEXT bar open +/- half-spread. Stop = opposite range
side (R = width); target = entry + rr x R, fixed. One trade/pair/session; flat by
open+hold_hours; no new entries after open+entry_cutoff_hours. Stop before target;
entry bar can stop; gaps fill at the open on the bad side; 0.5% risk; cost floor.

VALIDATION (honest, given the 2026 FX window is spent by prior studies): SELECT the
grid winner on 2024-07 -> 2025-12, judge ONCE on the UNTOUCHED holdout
(<= 2024-06-30, back to history_start) with gate + walk-forward + bootstrap CI +
slippage x2. The intraday-ORB family never used the holdout (other FX families saw
FX data at other granularities - declared; walk-forward across years is the real
defense). 2026 printed as a contaminated reference only. Grid (8 combos):
session [london, ny] x regime_filter [F,T] x rs_topk [null, 3].

Run: python -m src.backtest.strategy12_fx_orb
"""

import itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import oanda_data, slackbot
from src.backtest import metrics
from src.backtest.strategy10_boring_scalp import (Trade, pip_size, size_units,
                                                  usd_pnl_factor)
from src.backtest.strategy10_gold import between, bootstrap_ci, walk_forward

ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS = {"london": 7 * 60, "ny": 13 * 60 + 30}


def usd_side(pair):
    """+1 if USD is the base currency (pair up = USD up), else -1."""
    return 1 if pair.startswith("USD") else -1


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def expand_grid(grid):
    keys = sorted(grid.keys())
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[k] for k in keys))]


def build_context(frames, sopen, open_bars):
    """Per date: USD-regime direction, pair ranking by regime-aligned strength,
    and each pair's opening-range levels. Computed once per session."""
    per = {}
    win_end = sopen + open_bars * 5
    for pair, df in frames.items():
        minute = df["ts"].dt.hour * 60 + df["ts"].dt.minute
        d = df["ts"].dt.strftime("%Y-%m-%d")
        mask = (minute >= sopen) & (minute < win_end)
        if not mask.any():
            continue
        for date, sub in df[mask].groupby(d[mask]):
            oropen = float(sub.iloc[0]["open"])
            if oropen <= 0:
                continue
            per.setdefault(date, {})[pair] = {
                "orh": float(sub["high"].max()), "orl": float(sub["low"].min()),
                "oropen": oropen,
                "early": (float(sub.iloc[-1]["close"]) - oropen) / oropen}
    ctx = {}
    for date, pairs in per.items():
        usd = {p: v["early"] * usd_side(p) for p, v in pairs.items()}
        idx = sum(usd.values()) / len(usd)
        rdir = 1 if idx > 0 else -1
        ranked = sorted(pairs, key=lambda p: usd[p] * rdir, reverse=True)
        ctx[date] = {"regime_dir": rdir, "ranked": ranked, "levels": pairs,
                     "aligned": {p: rdir * usd_side(p) for p in pairs}}
    return ctx


def simulate_pair(df, pair, combo, sim, ctx):
    """Session ORB for one pair, using the pre-built cross-sectional context."""
    regime = bool(combo["regime_filter"])
    topk = combo["rs_topk"]
    rr = float(sim["rr"])
    hs = float(sim["half_spread"][pair])
    sopen = sim["sopen"]
    or_ready_min = sopen + int(sim["open_bars"]) * 5
    cutoff = sopen + int(sim["entry_cutoff_hours"] * 60)
    flat_min = sopen + int(sim["hold_hours"] * 60)
    floor = sim.get("min_or_width_frac")
    min_mult = float(sim.get("min_stop_cost_mult", 2.0))
    df = df.reset_index(drop=True)
    n = len(df)

    trades = []
    pos = None
    pending = None
    cur_date = None
    trades_today = 0
    day = None

    def close_trade(row, exit_px, reason):
        d = pos["side"]
        risk_ps = abs(pos["entry"] - pos["stop"])
        f = usd_pnl_factor(pair, pos["entry"])
        pnl = (exit_px - pos["entry"]) * pos["shares"] * d * f
        trades.append(Trade(
            symbol=pair, strategy="strategy12_fx_orb", date=str(row["ts"].date()),
            entry_time=pos["entry_time"], exit_time=str(row["ts"].time()),
            entry=round(pos["entry"], 6), exit=round(exit_px, 6), shares=pos["shares"],
            stop=round(pos["stop"], 6), target=round(pos["target"], 6), pnl=round(pnl, 2),
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
            day = ctx.get(date)

        if pos is not None and minute >= flat_min:
            close_trade(row, o - pos["side"] * hs, "flat")
            pos = None

        if pos is None and pending is not None and minute < flat_min:
            side = pending["side"]
            entry_px = o + side * hs
            stop = pending["stop"]
            risk_ps = (entry_px - stop) * side
            if risk_ps > 0 and risk_ps >= min_mult * 2 * hs:
                units = size_units(entry_px, stop, pair, sim)
                if units > 0:
                    pos = {"side": side, "entry": entry_px, "stop": stop,
                           "target": entry_px + side * rr * risk_ps, "shares": units,
                           "entry_time": str(row["ts"].time()), "reason": pending["reason"]}
                    trades_today += 1
        pending = None

        if pos is not None:
            d = pos["side"]
            if d == 1 and lo_b <= pos["stop"]:
                close_trade(row, min(o, pos["stop"]) - hs, "stop"); pos = None
            elif d == 1 and hi_b >= pos["target"]:
                close_trade(row, max(o, pos["target"]) - hs, "target"); pos = None
            elif d == -1 and hi_b >= pos["stop"]:
                close_trade(row, max(o, pos["stop"]) + hs, "stop"); pos = None
            elif d == -1 and lo_b <= pos["target"]:
                close_trade(row, min(o, pos["target"]) + hs, "target"); pos = None

        # signal
        if (pos is None and pending is None and day is not None and trades_today < 1
                and minute >= or_ready_min and sopen <= minute < cutoff
                and pair in day["levels"] and i < n - 1
                and df.iloc[i + 1]["ts"].strftime("%Y-%m-%d") == date):
            lv = day["levels"][pair]
            orh, orl, oropen = lv["orh"], lv["orl"], lv["oropen"]
            width_ok = orh > orl and (not floor or (orh - orl) / oropen >= float(floor))
            rs_ok = (topk is None) or (pair in day["ranked"][:int(topk)])
            if width_ok and rs_ok:
                permit = {day["aligned"][pair]} if regime else {1, -1}
                if c > orh and 1 in permit:
                    pending = {"side": 1, "stop": orl, "reason": "fxorb_break_long"}
                elif c < orl and -1 in permit:
                    pending = {"side": -1, "stop": orh, "reason": "fxorb_break_short"}

    if pos is not None and n > 0:
        row = df.iloc[n - 1]
        close_trade(row, float(row["close"]) - pos["side"] * hs, "data_end")
    return trades


def sim_base(g, half_spread, sopen, slip_mult=1.0):
    return {"risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
            "half_spread": {p: hs * slip_mult for p, hs in half_spread.items()},
            "min_stop_cost_mult": g.get("min_stop_cost_mult", 2.0),
            "rr": g["rr"], "open_bars": g["open_bars"], "sopen": sopen,
            "entry_cutoff_hours": g["entry_cutoff_hours"], "hold_hours": g["hold_hours"],
            "min_or_width_frac": g["min_or_width_frac"]}


def run_combo(frames, combo, g, half_spread, ctx_cache, slip_mult=1.0):
    sopen = SESSIONS[combo["session"]]
    ctx = ctx_cache.setdefault(combo["session"], build_context(frames, sopen, g["open_bars"]))
    sim = sim_base(g, half_spread, sopen, slip_mult)
    trades = []
    for pair, df in frames.items():
        trades.extend(simulate_pair(df, pair, combo, sim, ctx))
    return trades


def run():
    cfg = load_config()
    g = cfg["fx_strategy12_orb"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    end = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    sel_lo = pd.to_datetime(g["select_start"]).date()
    sel_hi = pd.to_datetime(g["select_end"]).date()
    hold_hi = pd.to_datetime(g["holdout_end"]).date()
    gate = g["gate"]
    half_spread = {p: float(g["spread_pips"][p]) * pip_size(p) / 2.0 for p in g["instruments"]}

    frames = {}
    for p in g["instruments"]:
        print(f"downloading {p} {g['granularity']}, {g['history_start']} -> {end}", flush=True)
        df = oanda_data.fetch_candles(p, g["history_start"], end, granularity=g["granularity"])
        print(f"  {p}: {len(df):,} bars", flush=True)
        if not df.empty:
            frames[p] = df
    if not frames:
        slackbot.post(f"[FX-STRATEGY12-ORB] {ts} - FAILED: no candles.")
        return

    ctx_cache = {}
    combos = expand_grid(g["grid"])
    scored = []
    for combo in combos:
        tr = run_combo(frames, combo, g, half_spread, ctx_cache)
        ms = metrics.summarize(between(tr, sel_lo, sel_hi))
        scored.append((combo, ms, tr))
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        print(f"  {cs}: select {ms.get('trades',0)}t {ms.get('expectancy_r',0)}R", flush=True)

    report = [f"# FX majors ORB port (strategy #12) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Stock ORB filters (vol floor + USD regime + "
              "relative strength) reproduced on a 7-pair basket; judged on an UNTOUCHED holdout.",
              f"pairs {g['instruments']} - select {g['select_start']}..{g['select_end']} - "
              f"holdout <= {g['holdout_end']}", "", "## Selection grid (spent window)", ""]
    for combo, m, _ in scored:
        cs = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        report.append(f"- {cs}: {m.get('trades',0)} trades, {m.get('expectancy_r',0)}R, "
                      f"PF {m.get('profit_factor',0)}")
    report.append("")

    eligible = [(c, m, tr) for c, m, tr in scored if m.get("trades", 0) >= g["min_select_trades"]]
    if not eligible:
        line = f"SKIP - no combo reached {g['min_select_trades']} selection trades"
        report += [f"## Verdict: {line}", ""]
        slack_body = [line]
    else:
        eligible.sort(key=lambda x: x[1]["expectancy_r"], reverse=True)
        win, sel_m, wt = eligible[0]
        cs = ", ".join(f"{k}={v}" for k, v in sorted(win.items()))
        hold = between(wt, hi=hold_hi)
        hm = metrics.summarize(hold)
        gross = run_combo(frames, win, g, half_spread, ctx_cache, slip_mult=0.0)
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
            sens = []
            for sm in g.get("slippage_mult", [1.0]):
                hs_trades = run_combo(frames, win, g, half_spread, ctx_cache, slip_mult=sm)
                sens.append(f"{sm}x -> {metrics.summarize(between(hs_trades, hi=hold_hi)).get('expectancy_r',0)}R")
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
                f"- slippage sensitivity: {' | '.join(sens)}",
                f"- 2026 reference (CONTAMINATED, not a gate): {ref.get('trades',0)} trades, "
                f"{ref.get('expectancy_r',0)}R, PF {ref.get('profit_factor',0)}",
                f"- gate: {why}", ""]
            slack_body = [
                f"winner {cs}",
                f"holdout {hm['expectancy_r']:+}R (PF {hm['profit_factor']}, {hm['trades']}t, "
                f"{hm['quarters_positive']}/{hm['quarters_total']}q+) -> *{verdict}*",
                f"bootstrap 90% CI [{lo:+.3f},{hi:+.3f}]R P(>0)={frac_pos*100:.0f}% | "
                f"WF {wf_pos}/{wf_tot} | gross {gm.get('expectancy_r',0):+}R | slip {' '.join(sens)}"]

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy12orb_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy12orb_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY12-ORB]* {ts} - RESEARCH ONLY, nothing deploys\n"
              "Stock ORB ported to FX majors basket (vol floor + USD regime + rel-strength), "
              f"judged on UNTOUCHED holdout <= {g['holdout_end']}")
    footer = f"Full detail: reports/fx_strategy12orb_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
