"""Multi-TF SMC swing engine on a 9-instrument basket - RESEARCH ONLY, nothing trades.

PRE-REGISTERED (2026-07-18, before any run). Origin: Peter's video strategy #13
(Daily HH/HL bias -> H4 liquidity sweep + Market Structure Shift + 50% discount ->
H1 liquidity grab + stop-entry, stop at sweep extreme, TP 1:2). Prototyped on gold
M15 (2019-2026): 113 trades, net +0.37R train / +0.24R validation, positive 7 of 8
years, survives 2x costs and a k=3 swing-definition perturbation. Caveats known:
32 validation trades only; 13th hypothesis on the gold window; long side carries
the edge in a bull regime.

WHY A BASKET (declared): Peter wants 3-4 trades/week. Timeframe compression is
measured dead (M15 family -0.03R gross over 3,909 trades; H1/M30 rung negative
in-sample). The honest route to frequency is BREADTH: the identical engine on 9
instruments at the only scale where the edge showed a pulse (~0.3 trades/wk each
-> ~2.5-4/wk portfolio). Frequency from diversification, not compression.

ZERO KNOBS (declared): engine parameters are FROZEN exactly as prototyped on gold
(k=2 fractals, 60-H4-bar setup expiry, 120-H1-bar trigger window, 720-H1-bar max
hold, RR 1:2). There is no grid. One configuration, judged once on validation.
Anything else would be tuning on reused data.

Bankroll rules (portfolio-wide): 0.5% risk per trade via stop-distance sizing,
max 3 concurrent positions (later entries skipped), 20% notional cap per trade.

Honesty notes: per-instrument edges must individually make sense ("infra
generalizes, edge does NOT") - per-instrument validation is reported; correlated
metals/majors are NOT independent samples; USD_CAD/USD_CHF PnL uses the repo's
declared quote-currency approximation; entry at stop level +/- half-spread, costs
= full spread charged once per round trip; a PASS here is still backtest evidence
ONLY and graduates to a paper trial, never to money.

Run: python -m src.backtest.strategy9_smc_basket
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import oanda_data, slackbot
from src.backtest import metrics

ROOT = Path(__file__).resolve().parent.parent.parent

# ---- frozen engine parameters (from the gold prototype; NOT tunable) ----
K, EXP_H4, WIN_H1, HOLD_H1, RR = 2, 60, 120, 720, 2.0


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


def usd_pnl_factor(instrument, price):
    return (1.0 / price) if instrument.endswith("JPY") else 1.0


def unit_notional_usd(instrument, price):
    return 1.0 if instrument.startswith("USD") else price


def size_units(entry, stop, instrument, risk_cfg):
    risk_dollars = risk_cfg["equity"] * risk_cfg["risk_pct"] / 100.0
    risk_ps_usd = abs(entry - stop) * usd_pnl_factor(instrument, entry)
    if risk_ps_usd <= 0:
        return 0.0
    max_units = (risk_cfg["equity"] * risk_cfg["max_position_pct"] / 100.0) / unit_notional_usd(instrument, entry)
    return round(max(min(risk_dollars / risk_ps_usd, max_units), 0.0), 2)


def resample(m15, rule):
    return m15.resample(rule).agg({"open": "first", "high": "max",
                                   "low": "min", "close": "last"}).dropna()


def fractals(h, l, k=K):
    hs, ls = [], []
    for i in range(k, len(h) - k):
        wh, wl = h[i-k:i+k+1], l[i-k:i+k+1]
        if h[i] == wh.max() and (wh == h[i]).sum() == 1: hs.append((i + k, h[i]))
        if l[i] == wl.min() and (wl == l[i]).sum() == 1: ls.append((i + k, l[i]))
    return hs, ls


def simulate_m15(m15):
    """Pure engine. Returns [(entry_ts, exit_ts, pnl_price, risk_price, dir)]. No costs."""
    H1, H4, D1 = resample(m15, "1h"), resample(m15, "4h"), resample(m15, "1D")
    dh, dl = D1["high"].values, D1["low"].values
    dhs, dls = fractals(dh, dl)
    bias_vals = np.zeros(len(D1), int)
    hi_h, lo_h = [], []; ei = ej = 0; bias = 0
    for di in range(len(D1)):
        while ei < len(dhs) and dhs[ei][0] <= di: hi_h.append(dhs[ei][1]); ei += 1
        while ej < len(dls) and dls[ej][0] <= di: lo_h.append(dls[ej][1]); ej += 1
        if len(hi_h) >= 2 and len(lo_h) >= 2:
            if hi_h[-1] > hi_h[-2] and lo_h[-1] > lo_h[-2]: bias = 1
            elif hi_h[-1] < hi_h[-2] and lo_h[-1] < lo_h[-2]: bias = -1
        bias_vals[di] = bias
    bias_known = (D1.index + pd.Timedelta(days=1)).asi8
    h4_bias_idx = np.searchsorted(bias_known, H4.index.asi8, side="right") - 1
    h4_bias = np.where(h4_bias_idx >= 0, bias_vals[np.clip(h4_bias_idx, 0, None)], 0)
    h4h, h4l, h4c = H4["high"].values, H4["low"].values, H4["close"].values
    h4hs, h4ls = fractals(h4h, h4l)
    h1o, h1h, h1l, h1c = (H1[x].values for x in ["open", "high", "low", "close"])
    h1hs, h1ls = fractals(h1h, h1l)
    t_times = H1.index

    def h1_pos(ts):
        return int(np.searchsorted(t_times, ts, side="right"))

    trades = []; state = None; ei = ej = 0; lastH4hi = lastH4lo = None
    for t in range(len(H4)):
        while ei < len(h4hs) and h4hs[ei][0] <= t: lastH4hi = h4hs[ei][1]; ei += 1
        while ej < len(h4ls) and h4ls[ej][0] <= t: lastH4lo = h4ls[ej][1]; ej += 1
        ts = H4.index[t]; b = int(h4_bias[t])
        if state is None:
            if b == 1 and lastH4lo is not None and h4l[t] < lastH4lo:
                state = dict(d=1, sweep=h4l[t], phase="swept", t0=t, leghi=h4h[t])
            elif b == -1 and lastH4hi is not None and h4h[t] > lastH4hi:
                state = dict(d=-1, sweep=h4h[t], phase="swept", t0=t, leglo=h4l[t])
            continue
        if t - state["t0"] > EXP_H4: state = None; continue
        d = state["d"]
        if d == 1:
            state["sweep"] = min(state["sweep"], h4l[t]); state["leghi"] = max(state["leghi"], h4h[t])
            if state["phase"] == "swept" and lastH4hi is not None and h4c[t] > lastH4hi:
                state["phase"] = "mss"; state["mid"] = (state["sweep"] + state["leghi"]) / 2
            elif state["phase"] == "mss" and h4l[t] <= state["mid"]:
                state["phase"] = "h1"; state["h1start"] = h1_pos(ts)
        else:
            state["sweep"] = max(state["sweep"], h4h[t]); state["leglo"] = min(state["leglo"], h4l[t])
            if state["phase"] == "swept" and lastH4lo is not None and h4c[t] < lastH4lo:
                state["phase"] = "mss"; state["mid"] = (state["sweep"] + state["leglo"]) / 2
            elif state["phase"] == "mss" and h4h[t] >= state["mid"]:
                state["phase"] = "h1"; state["h1start"] = h1_pos(ts)
        if state.get("phase") != "h1": continue
        s0 = state["h1start"]; d = state["d"]; sweep = state["sweep"]
        hi_i = [x for x in h1hs if x[0] <= s0]; lo_i = [x for x in h1ls if x[0] <= s0]
        ph = [x for x in h1hs if x[0] > s0]; pl = [x for x in h1ls if x[0] > s0]
        lastHi = hi_i[-1][1] if hi_i else None; lastLo = lo_i[-1][1] if lo_i else None
        grab = False; trigger = None; entry_i = None
        j = s0; pi = qi = 0
        while j < min(s0 + WIN_H1, len(h1c)):
            while pi < len(ph) and ph[pi][0] <= j: lastHi = ph[pi][1]; pi += 1
            while qi < len(pl) and pl[qi][0] <= j: lastLo = pl[qi][1]; qi += 1
            if d == 1:
                if h1l[j] < sweep: break
                if not grab and lastLo is not None and h1l[j] < lastLo: grab = True
                elif grab and trigger is None and lastHi is not None: trigger = lastHi
                elif trigger is not None and h1h[j] >= trigger: entry_i = j; break
            else:
                if h1h[j] > sweep: break
                if not grab and lastHi is not None and h1h[j] > lastHi: grab = True
                elif grab and trigger is None and lastLo is not None: trigger = lastLo
                elif trigger is not None and h1l[j] <= trigger: entry_i = j; break
            j += 1
        state = None
        if entry_i is None: continue
        entry = trigger; risk = (entry - sweep) if d == 1 else (sweep - entry)
        if risk <= 0: continue
        stop = sweep; pnl = None
        for k2 in range(entry_i, min(entry_i + HOLD_H1, len(h1c))):
            if d == 1:
                if h1l[k2] <= stop: pnl = (min(h1o[k2], stop) if k2 > entry_i else stop) - entry; break
                if h1h[k2] >= entry + RR * risk: pnl = RR * risk; break
            else:
                if h1h[k2] >= stop: pnl = entry - (max(h1o[k2], stop) if k2 > entry_i else stop); break
                if h1l[k2] <= entry - RR * risk: pnl = RR * risk; break
        if pnl is None:
            k2 = min(entry_i + HOLD_H1, len(h1c) - 1); pnl = (h1c[k2] - entry) * d
        trades.append((t_times[entry_i], t_times[min(k2, len(h1c) - 1)], pnl, risk, d))
    return trades


def cap_concurrent(rows, max_open=3):
    """Portfolio bankroll rule: accept only if < max_open positions active at entry.
    rows: (entry_ts, exit_ts, pnl, risk, dir, instrument)."""
    rows = sorted(rows, key=lambda x: x[0])
    accepted, exits = [], []
    for tr in rows:
        exits = [e for e in exits if e > tr[0]]
        if len(exits) < max_open:
            accepted.append(tr); exits.append(tr[1])
    return accepted


def to_trades(rows, spreads, risk_cfg, cost_mult=1.0):
    out = []
    for ets, xts, pnl, risk, d, inst in rows:
        cost = spreads[inst] * cost_mult
        net = pnl - cost
        entry = 0.0  # price bookkeeping is per-move; entry level not needed for R metrics
        units = size_units(100.0, 100.0 - risk, inst, risk_cfg)  # sizing off risk distance
        f = usd_pnl_factor(inst, 100.0)
        out.append(Trade(symbol=inst, strategy="strategy9_smc_basket",
                         date=str(ets)[:10], entry_time=str(ets), exit_time=str(xts),
                         entry=entry, exit=entry + net * d, shares=units, stop=0.0, target=0.0,
                         pnl=round(net / risk * risk_cfg["equity"] * risk_cfg["risk_pct"] / 100.0, 2),
                         r_multiple=round(net / risk, 3),
                         exit_reason="resolved", signal_reason="smc_" + ("long" if d == 1 else "short")))
    return out


def run():
    cfg = load_config()
    fx = cfg["fx_strategy9"]
    risk_cfg = cfg["risk"]
    val_end = fx.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    instruments = fx["instruments"]
    spreads = fx["spread_price"]
    gate = fx["gate"]
    train_end = pd.to_datetime(fx["train_end"]).date()
    val_start = pd.to_datetime(fx["val_start"]).date()
    max_open = int(fx.get("max_concurrent", 3))

    raw = []
    for inst in instruments:
        print(f"downloading {inst} M15, {fx['train_start']} -> {val_end}", flush=True)
        df = oanda_data.fetch_candles(inst, fx["train_start"], val_end, granularity="M15")
        print(f"  {inst}: {len(df):,} bars", flush=True)
        if df.empty:
            continue
        df = df.set_index("ts").sort_index()
        sim = simulate_m15(df)
        print(f"  {inst}: {len(sim)} trades", flush=True)
        raw.extend([(a, b, p, r, d, inst) for (a, b, p, r, d) in sim])
    if not raw:
        slackbot.post(f"[FX-STRATEGY9] {ts} - FAILED: no data/trades.")
        return

    capped = cap_concurrent(raw, max_open)
    skipped = len(raw) - len(capped)
    span_wk = max((max(x[0] for x in capped) - min(x[0] for x in capped)).days / 7.0, 1.0)

    def split(trs):
        tr = [t for t in trs if pd.to_datetime(t.date).date() <= train_end]
        va = [t for t in trs if pd.to_datetime(t.date).date() >= val_start]
        return tr, va

    trades = to_trades(capped, spreads, risk_cfg)
    tr, va = split(trades)
    m_tr, m_va = metrics.summarize(tr), metrics.summarize(va)
    trades2 = to_trades(capped, spreads, risk_cfg, cost_mult=2.0)
    _, va2 = split(trades2)
    m_va2 = metrics.summarize(va2)

    report = [f"# SMC swing basket (9 instruments, frozen engine) - {ts}", "",
              "RESEARCH ONLY - nothing deploys. Pre-registered 2026-07-18. ZERO tunable knobs.",
              f"instruments {instruments}",
              f"spread (price units): {spreads}",
              f"engine frozen from gold prototype: k={K}, exp={EXP_H4} H4 bars, trigger window {WIN_H1} H1 bars, "
              f"hold {HOLD_H1} H1 bars, RR 1:{RR:.0f}",
              f"bankroll: {risk_cfg['risk_pct']}% risk/trade, max {max_open} concurrent "
              f"({skipped} entries skipped by cap), {risk_cfg['max_position_pct']}% notional cap",
              f"train {fx['train_start']} -> {fx['train_end']} - validation {fx['val_start']} -> {val_end}",
              f"portfolio frequency: {len(capped)/span_wk:.1f} trades/week", "",
              "## Per-instrument validation (net, uncapped) - edge must make sense per market", ""]
    for inst in instruments:
        rows = [x for x in raw if x[5] == inst]
        _, iva = split(to_trades(rows, spreads, risk_cfg))
        m = metrics.summarize(iva)
        if m.get("trades", 0):
            report.append(f"- {inst}: {m['trades']} val trades, {m['expectancy_r']}R, PF {m['profit_factor']}")
        else:
            report.append(f"- {inst}: no validation trades")
    report.append("")

    if m_va.get("trades", 0) == 0:
        label, why = "FAIL", "0 validation trades"
    else:
        verdict, why = metrics.gate_verdict(m_va, gate)
        weak = verdict == "PASS" and m_tr.get("expectancy_r", 0) < fx.get("min_train_expectancy_r", 0.02)
        label = "WEAK PASS" if weak else verdict
    report += [f"## Verdict: {label}",
               f"- train: {m_tr.get('trades', 0)} trades, {m_tr.get('expectancy_r', 0)}R, "
               f"PF {m_tr.get('profit_factor', 0)}",
               f"- validation: {m_va.get('trades', 0)} trades, win {m_va.get('win_rate', 0)}%, "
               f"{m_va.get('expectancy_r', 0)}R, PF {m_va.get('profit_factor', 0)}, "
               f"{m_va.get('quarters_positive', 0)}/{m_va.get('quarters_total', 0)} quarters+",
               f"- spread x2 validation: {m_va2.get('expectancy_r', 0)}R, PF {m_va2.get('profit_factor', 0)}",
               f"- gate (informational): {why}",
               "", "A PASS graduates to a PAPER TRIAL only. Never to real capital from a backtest."]

    out_dir = ROOT / "reports"; out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"fx_strategy9_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"report written: reports/fx_strategy9_{stamp}.md", flush=True)

    header = (f"*[FX-STRATEGY9]* {ts} - RESEARCH ONLY, nothing deploys\n"
              f"SMC swing basket, {len(instruments)} instruments, frozen engine, "
              f"{len(capped)/span_wk:.1f} trades/wk, max {max_open} concurrent")
    body = [f"train {m_tr.get('expectancy_r', 0):+}R ({m_tr.get('trades', 0)}t) -> "
            f"val {m_va.get('expectancy_r', 0):+}R (PF {m_va.get('profit_factor', 0)}, "
            f"{m_va.get('trades', 0)}t) -> *{label}*",
            f"spread x2 val: {m_va2.get('expectancy_r', 0):+}R"]
    slackbot.post("\n\n".join([header] + body + [f"Full detail: reports/fx_strategy9_{stamp}.md"]))


if __name__ == "__main__":
    run()
