"""Strategy #10 TRAIN-ONLY diagnostic for GOLD + S&P 500 - RESEARCH ONLY.

Not a new hypothesis and NOT judged on validation - it spends zero validation
debt. It only dissects the TRAIN window of the strategy #10 CFD study to answer
"where does the (small, positive) edge come from and why is it lumpy?" for
XAU_USD first, then SPX500_USD.

For each instrument it re-selects the same train winner (highest train
expectancy with >= min_train_trades, exactly as strategy10_cfd does) and then
breaks the TRAIN trades down by:
  - quarter (expectancy_R, count, PF) - exposes lumpiness
  - exit reason (stop / trail_stop / flat_2100 / data_end)
  - level family (PDH/PDL zone break vs 5-minute opening-range break)
  - direction (long vs short)
  - GROSS vs NET: the same combo re-run with zero spread, to see how much of
    the signal the cost toll eats (declared: zero-cost also lifts the cost
    floor, so the gross trade population is slightly larger).

Engine, sizing and honesty rules are IMPORTED unchanged from strategy10 - this
file only reads and reports. Run: python -m src.backtest.strategy10_diag
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import oanda_data, slackbot
from src.backtest import metrics
from src.backtest.strategy10_boring_scalp import (expand_grid, simulate_instrument,
                                                  split_trades)

ROOT = Path(__file__).resolve().parent.parent.parent


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _level_family(reason):
    r = reason or ""
    if r.startswith("pdh") or r.startswith("pdl"):
        return "prev_day_level"
    if r.startswith("orh") or r.startswith("orl"):
        return "opening_range"
    return "other"


def _grp(df, keyfn, label):
    """Mean R / count / win% per group, as report lines."""
    if df.empty:
        return [f"  by {label}: (no trades)"]
    g = df.copy()
    g["_k"] = g.apply(keyfn, axis=1) if callable(keyfn) else g[keyfn]
    out = [f"  by {label}:"]
    for k, sub in g.groupby("_k"):
        wins = (sub["r_multiple"] > 0).mean() * 100
        pf_den = -sub.loc[sub["r_multiple"] <= 0, "r_multiple"].sum()
        pf = (sub.loc[sub["r_multiple"] > 0, "r_multiple"].sum() / pf_den) if pf_den > 0 else float("inf")
        out.append(f"    {k}: {len(sub)} trades, {sub['r_multiple'].mean():+.3f}R, "
                   f"win {wins:.0f}%, PF {pf:.2f}")
    return out


def diagnose(df, inst, fx, cfg_sim, half_spread):
    train_end = pd.to_datetime(fx["train_end"]).date()
    val_start = pd.to_datetime(fx["val_start"]).date()
    combos = expand_grid(fx["grids"])

    scored = []
    for combo in combos:
        tr, _ = split_trades(simulate_instrument(df, inst, combo, cfg_sim), train_end, val_start)
        scored.append((combo, metrics.summarize(tr), tr))
    eligible = [s for s in scored if s[1].get("trades", 0) >= fx["min_train_trades"]]
    if not eligible:
        return [f"### {inst}", "", "no combo reached the train-trade floor - nothing to dissect.", ""], \
               f"{inst}: no eligible combo"

    eligible.sort(key=lambda s: s[1]["expectancy_r"], reverse=True)
    combo, m, tr = eligible[0]
    combo_str = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
    tdf = pd.DataFrame([t.__dict__ for t in tr])
    tdf["date"] = pd.to_datetime(tdf["date"])
    tdf["quarter"] = pd.PeriodIndex(tdf["date"], freq="Q").astype(str)
    tdf["dir"] = tdf["r_multiple"].where(False)  # placeholder to keep column order
    tdf["dir"] = tdf["signal_reason"].str.contains("long").map({True: "long", False: "short"})

    # gross (zero cost) rerun of the same combo
    cfg0 = {**cfg_sim, "half_spread": {k: 0.0 for k in half_spread}}
    tr0, _ = split_trades(simulate_instrument(df, inst, combo, cfg0), train_end, val_start)
    m0 = metrics.summarize(tr0)

    lines = [f"### {inst}", "",
             f"train winner: {combo_str}",
             f"NET train:   {m['trades']} trades, {m['expectancy_r']:+.3f}R, "
             f"PF {m['profit_factor']}, win {m['win_rate']}%",
             f"GROSS train: {m0.get('trades', 0)} trades, {m0.get('expectancy_r', 0):+.3f}R, "
             f"PF {m0.get('profit_factor', 0)} (zero-cost; cost toll = "
             f"{m0.get('expectancy_r', 0) - m['expectancy_r']:+.3f}R eaten)", ""]
    lines += _grp(tdf, "quarter", "quarter")
    lines += _grp(tdf, lambda r: r["exit_reason"], "exit reason")
    lines += _grp(tdf, lambda r: _level_family(r["signal_reason"]), "level family")
    lines += _grp(tdf, "dir", "direction")
    lines.append("")

    pos_q = (tdf.groupby("quarter")["r_multiple"].mean() > 0).sum()
    tot_q = tdf["quarter"].nunique()
    slack = (f"{inst}: net {m['expectancy_r']:+.3f}R vs gross {m0.get('expectancy_r', 0):+.3f}R, "
             f"{pos_q}/{tot_q} train quarters+ (winner {combo_str})")
    return lines, slack


def run():
    cfg = load_config()
    fx = cfg["fx_strategy10_cfd"]
    val_end = fx.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    instruments = fx["instruments"]
    half_spread = {inst: float(fx["spread_price"][inst]) / 2.0 for inst in instruments}
    cfg_sim = {"risk": cfg["risk"], "half_spread": half_spread,
               "min_stop_cost_mult": fx.get("min_stop_cost_mult", 2.0),
               "ny_open_min": fx["ny_open_min"], "or_minutes": fx["or_minutes"],
               "flat_min": fx["flat_min"], "ema_fast": fx["ema_fast"],
               "ema_slow": fx["ema_slow"], "zone_frac": fx["zone_frac"],
               "stop_buf_frac": fx["stop_buf_frac"]}

    report = [f"# Strategy #10 TRAIN-ONLY diagnostic - gold + S&P 500 - {ts}", "",
              "RESEARCH ONLY. Dissects the TRAIN window only - no validation debt spent.",
              f"train {fx['train_start']} -> {fx['train_end']}", ""]
    slack_lines = []
    for inst in instruments:
        print(f"downloading {inst} M1", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end,
                                      granularity=fx.get("granularity", "M1"))
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if df.empty:
            report += [f"### {inst}", "", "no candles returned.", ""]
            slack_lines.append(f"{inst}: no candles")
            continue
        lines, slack = diagnose(df, inst, fx, cfg_sim, half_spread)
        report += lines
        slack_lines.append(slack)
        print(f"  {slack}", flush=True)

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy10diag_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy10diag_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY10-DIAG]* {ts} - RESEARCH ONLY (train-only, no val debt)\n"
              "Where does strategy #10's gold/S&P edge come from, and why lumpy?")
    footer = f"Full detail: reports/fx_strategy10diag_{stamp}.md"
    slackbot.post("\n\n".join([header] + slack_lines + [footer]))


if __name__ == "__main__":
    run()
