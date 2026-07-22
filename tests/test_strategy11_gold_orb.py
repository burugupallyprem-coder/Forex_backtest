"""Gold ORB-momentum (strategy #11) correctness. Run: python tests/test_strategy11_gold_orb.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy11_gold_orb import simulate_orb

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"XAU_USD": 0.175},
    "min_stop_cost_mult": 2.0,
    "ny_open_min": 810, "entry_cutoff_min": 1020, "flat_min": 1260,
    "ema_fast": 50, "ema_slow": 200, "min_or_width_frac": 0.0010,
}
LONG = {"or_minutes": 15, "target_r": 2.0, "trend_filter": False}
D = datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)


def _bar(m, o, h, l, c):
    return {"ts": D + timedelta(minutes=m), "open": o, "high": h, "low": l, "close": c, "volume": 10}


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _wide_or_day(extra):
    # OR window 810-824: high 2405 / low 2395 (width 10, ~0.42% of price -> clears floor)
    base = [
        _bar(810, 2400, 2405, 2398, 2402),
        _bar(811, 2402, 2404, 2395, 2400),
        _bar(812, 2400, 2403, 2399, 2401),
        _bar(825, 2406, 2410, 2405, 2408),   # first close above OR high -> breakout long
        _bar(826, 2409, 2411, 2408, 2410),   # fill next open
    ]
    return _df(base + extra)


def test_orb_long_hits_target():
    df = _wide_or_day([
        _bar(827, 2410, 2420, 2409, 2418),
        _bar(828, 2418, 2435, 2417, 2433),
        _bar(829, 2433, 2441, 2432, 2438),   # target ~2437.5 hit
    ])
    trades = simulate_orb(df, "XAU_USD", LONG, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "orb_break_long"
    assert t.exit_reason == "target"
    assert abs(t.stop - 2395) < 1e-6
    assert t.r_multiple > 1.8 and t.pnl > 0


def test_vol_floor_blocks_narrow_open():
    # OR high 2401 / low 2400 -> width 1, ~0.04% of price < 0.10% floor -> whole day skipped
    df = _df([
        _bar(810, 2400.0, 2401.0, 2400.0, 2400.5),
        _bar(811, 2400.5, 2401.0, 2400.0, 2400.5),
        _bar(825, 2402.0, 2404.0, 2401.5, 2403.0),
        _bar(826, 2403.0, 2405.0, 2402.5, 2404.0),
        _bar(827, 2404.0, 2409.0, 2403.5, 2408.0),
    ])
    assert simulate_orb(df, "XAU_USD", LONG, CFG) == []


def test_trend_filter_blocks_without_established_trend():
    # single-day data -> daily 50/200 EMA undefined -> dir 0 -> trend filter blocks longs
    df = _wide_or_day([_bar(827, 2410, 2420, 2409, 2418), _bar(828, 2418, 2435, 2417, 2433)])
    on = {"or_minutes": 15, "target_r": 2.0, "trend_filter": True}
    assert len(simulate_orb(df, "XAU_USD", LONG, CFG)) == 1
    assert simulate_orb(df, "XAU_USD", on, CFG) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
