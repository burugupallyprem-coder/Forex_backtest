"""Gold-DEDICATED tune of strategy #10 - RESEARCH ONLY, nothing here trades.

PRE-REGISTERED 2026-07-22. The train diagnostic showed gold has a real GROSS
edge (+0.124R) that the spread toll cuts to +0.050R net - so the only lever
worth pulling is the one that raises quality per trade: a VOLATILITY FLOOR that
skips dead-tape opens (opening-range width < min_or_width_frac of price), so the
fixed ~0.35 spread is only paid on days with room to run. Everything else is
FROZEN to gold's diagnostic winner (trail_lookback=20, trend_filter=off,
max_trades_day=1) - we are NOT re-searching those.

HONEST VALIDATION (the point of this run). Earlier gold studies already spent
the 2026 window, so it cannot be the judge. Instead:
  - SELECT the floor on the already-spent window (2024-07 -> 2025-12): choosing a
    value there adds no NEW judge debt.
  - JUDGE ONCE on the UNTOUCHED holdout: every session strategy #10 never looked
    at, i.e. <= holdout_end (2024-06-30), reaching as far back as OANDA M1 serves
    (history_start). Gate + walk-forward folds + a bootstrap CI on holdout
    expectancy_R (is the edge distinguishable from luck?), plus gross-vs-net.
  - The contaminated 2026 window is printed as a REFERENCE ONLY, never as a gate.
Caveat (declared): strategy #9 (a different family - SMC swing) once read gold
2019-2023, so the holdout is untouched by THESE break-retest params but not
pristine; walk-forward across many years is the real defense.

Run: python -m src.backtest.strategy10_gold
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import oanda_data, slackbot
from src.backtest import metrics
from src.backtest.strategy10_boring_scalp import simulate_instrument

ROOT = Path(__file__).resolve().parent.parent.parent


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sim_cfg(g, half_spread, floor):
    return {"risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
            "half_spread": {g["instrument"]: half_spread},
            "min_stop_cost_mult": g.get("min_stop_cost_mult", 2.0),
            "ny_open_min": g["ny_open_min"], "or_minutes": g["or_minutes"],
            "flat_min": g["flat_min"], "ema_fast": g["ema_fast"], "ema_slow": g["ema_slow"],
            "zone_frac": g["zone_frac"], "stop_buf_frac": g["stop_buf_frac"],
            "min_or_width_frac": floor}


def between(trades, lo=None, hi=None):
    out = []
    for t in trades:
        d = pd.to_datetime(t.date).date()
        if lo is not None and d < lo:
            continue
        if hi is not None and d > hi:
            continue
        out.append(t)
    return out


def walk_forward(trades, folds):
    if not trades:
        return 0, 0, []
    df = pd.DataFrame([t.__dict__ for t in trades])
    dates = sorted(set(df["date"]))
    size = max(1, len(dates) // folds)
    per = []
    for k in range(folds):
        lo = k * size
        hi = (k + 1) * size if k < folds - 1 else len(dates)
        if lo >= len(dates):
            break
        window = set(dates[lo:hi])
        sub = df[df["date"].isin(window)]
        per.append(float(sub["r_multiple"].mean()) if len(sub) else 0.0)
    return sum(1 for r in per if r > 0), len(per), per


def bootstrap_ci(rs, n=2000, seed=0):
    r = np.asarray([t.r_multiple for t in rs], dtype=float) if rs and hasattr(rs[0], "r_multiple") \
        else np.asarray(list(rs), dtype=float)
    if r.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n, r.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 5)), float(np.percentile(means, 95)),
            float((means > 0).mean()), float(r.mean()))


def run():
    cfg = load_config()
    g = cfg["fx_strategy10_gold"]
    inst = g["instrument"]
    half_spread = float(g["spread_price"]) / 2.0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    end = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    sel_lo = pd.to_datetime(g["select_start"]).date()
    sel_hi = pd.to_datetime(g["select_end"]).date()
    hold_hi = pd.to_datetime(g["holdout_end"]).date()
    base = g["base"]
    gate = g["gate"]

    print(f"downloading {inst} M1, {g['history_start']} -> {end}", flush=True)
    df = oanda_data.fetch_candles(inst, g["history_start"], end, granularity=g["granularity"])
    if df.empty:
        slackbot.post(f"[FX-STRATEGY10-GOLD] {ts} - FAILED: no candles.")
        return
    first = df["ts"].min().date()
    last = df["ts"].max().date()
    print(f"  {inst}: {len(df):,} bars, {first} -> {last}", flush=True)

    # ---- SELECT the floor on the already-spent window ----
    sel_rows = []
    trades_by_floor = {}
    for floor in g["grid"]["min_or_width_frac"]:
        params = {**base}
        trades = simulate_instrument(df, inst, params, sim_cfg(g, half_spread, floor))
        trades_by_floor[repr(floor)] = trades
        ms = metrics.summarize(between(trades, sel_lo, sel_hi))
        sel_rows.append((floor, ms))
        print(f"  floor={floor}: select {ms.get('trades',0)}t {ms.get('expectancy_r',0)}R", flush=True)

    eligible = [(f, m) for f, m in sel_rows if m.get("trades", 0) >= g["min_select_trades"]]
    report = [f"# Gold-dedicated strategy #10 (volatility-floor tune) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Base knobs frozen to gold's diagnostic winner; "
              "only the volatility floor is swept; judged on an UNTOUCHED holdout.",
              f"data {first} -> {last} - select {g['select_start']}..{g['select_end']} - "
              f"holdout <= {g['holdout_end']} (untouched by strategy #10)", "",
              "## Floor selection (on the already-spent window)", ""]
    for f, m in sel_rows:
        report.append(f"- floor={f}: {m.get('trades',0)} trades, {m.get('expectancy_r',0)}R, "
                      f"PF {m.get('profit_factor',0)}")
    report.append("")

    if not eligible:
        line = f"SKIP - no floor reached {g['min_select_trades']} selection trades"
        report += [f"## Verdict: {line}", ""]
        slackbot.post(f"*[FX-STRATEGY10-GOLD]* {ts} - RESEARCH ONLY\n{line}")
        (ROOT / "reports").mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        (ROOT / "reports" / f"fx_strategy10gold_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
        return

    eligible.sort(key=lambda x: x[1]["expectancy_r"], reverse=True)
    win_floor = eligible[0][0]
    sel_m = eligible[0][1]
    wt = trades_by_floor[repr(win_floor)]

    # ---- JUDGE ONCE on the untouched holdout ----
    hold = between(wt, hi=hold_hi)
    hm = metrics.summarize(hold)
    # gross (zero spread) on the holdout
    gross_trades = simulate_instrument(df, inst, {**base}, sim_cfg(g, 0.0, win_floor))
    gm = metrics.summarize(between(gross_trades, hi=hold_hi))
    ref = metrics.summarize(between(wt, lo=pd.to_datetime("2026-01-01").date()))  # contaminated ref

    if hm.get("trades", 0) == 0:
        line = f"FAIL - winner floor={win_floor} produced 0 holdout trades"
        report += [f"## Verdict: {line}", ""]
        slack_body = [line]
    else:
        verdict, why = metrics.gate_verdict(hm, gate)
        wf_pos, wf_tot, wf_per = walk_forward(hold, int(g["walkforward_folds"]))
        wf_ok = wf_tot > 0 and wf_pos / wf_tot >= float(g["min_positive_frac"])
        if verdict == "PASS" and not wf_ok:
            verdict = "FAIL"
            why = (f"{why}; " if why != "all gate checks met" else "") + \
                  f"walk-forward only {wf_pos}/{wf_tot} folds+"
        lo, hi, frac_pos, pt = bootstrap_ci(hold)
        report += [
            f"## Verdict: {verdict}",
            f"- winner floor: min_or_width_frac={win_floor}",
            f"- selection window: {sel_m['trades']} trades, {sel_m['expectancy_r']}R, PF {sel_m['profit_factor']}",
            f"- HOLDOUT (untouched, the judge): {hm['trades']} trades, win {hm['win_rate']}%, "
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
            f"winner floor={win_floor}",
            f"holdout {hm['expectancy_r']:+}R (PF {hm['profit_factor']}, {hm['trades']}t, "
            f"{hm['quarters_positive']}/{hm['quarters_total']}q+) -> *{verdict}*",
            f"bootstrap 90% CI [{lo:+.3f},{hi:+.3f}]R P(>0)={frac_pos*100:.0f}% | "
            f"WF {wf_pos}/{wf_tot} | gross {gm.get('expectancy_r',0):+}R"]

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy10gold_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy10gold_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY10-GOLD]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"Gold volatility-floor tune, judged on UNTOUCHED holdout <= {g['holdout_end']}")
    footer = f"Full detail: reports/fx_strategy10gold_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_body + [footer]))


if __name__ == "__main__":
    run()
