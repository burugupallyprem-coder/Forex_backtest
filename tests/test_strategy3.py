"""Overlap breakout (strategy #3) correctness. Run: python tests/test_strategy3.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy3_overlap import (daily_trend_map, pip_size,
                                            simulate_instrument, size_units)

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"USD_JPY": 0.008, "EUR_USD": 0.00007},
    "min_stop_cost_mult": 2.0,
    "overlap_start_min": 780,   # 13:00 UTC
    "overlap_end_min": 1020,    # 17:00 UTC
    "sma_days": 50,
}
OFF = {"or_minutes": 30, "target_r": 1.0, "trend_filter": False}


def bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 10}


def make_day(specs, day=datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)):
    rows = [bar(day + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in specs]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_basics():
    assert pip_size("USD_JPY") == 0.01 and pip_size("EUR_USD") == 0.0001


def test_long_breakout_target():
    # OR 13:00-13:30 -> hi 1.1002, lo 1.0990; close breaks hi -> long; target 1.1014
    df = make_day([
        (780, 1.1000, 1.1000, 1.0990, 1.0996),
        (795, 1.0996, 1.1002, 1.0990, 1.0998),
        (810, 1.0999, 1.1010, 1.0998, 1.1006),   # breakout close
        (825, 1.1007, 1.1009, 1.1005, 1.1008),   # entry fill 1.1007+hs
        (840, 1.1010, 1.1016, 1.1009, 1.1015),   # target 1.1014 hit
    ])
    trades = simulate_instrument(df, "EUR_USD", OFF, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "overlap_break_long"
    assert t.exit_reason == "target"
    assert abs(t.stop - 1.0990) < 1e-9
    assert abs(t.target - 1.1014) < 1e-9
    assert t.pnl > 0 and t.r_multiple > 0


def test_short_breakout_stop_conservative():
    # close breaks OR low -> short; gap up through stop -> fill at open (worse than stop)
    df = make_day([
        (780, 1.1000, 1.1002, 1.0998, 1.1000),
        (795, 1.1000, 1.1002, 1.0990, 1.0995),   # OR hi 1.1002 lo 1.0990
        (810, 1.0991, 1.0992, 1.0984, 1.0986),   # breakout close below lo
        (825, 1.0985, 1.0987, 1.0983, 1.0985),   # entry short 1.0985-hs
        (840, 1.1005, 1.1010, 1.1004, 1.1008),   # gap through stop 1.1002
    ])
    trades = simulate_instrument(df, "EUR_USD", OFF, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "overlap_break_short"
    assert t.exit_reason == "stop"
    assert t.r_multiple <= -1.0 and t.pnl < 0


def test_pre_overlap_ignored():
    # identical breakout shape but entirely before 13:00 UTC -> no trades
    df = make_day([
        (690, 1.1000, 1.1000, 1.0990, 1.0996),
        (705, 1.0996, 1.1002, 1.0990, 1.0998),
        (720, 1.0999, 1.1010, 1.0998, 1.1006),
        (735, 1.1007, 1.1009, 1.1005, 1.1008),
        (750, 1.1010, 1.1016, 1.1009, 1.1015),
    ])
    assert simulate_instrument(df, "EUR_USD", OFF, CFG) == []


def test_daily_trend_map_uptrend():
    base = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    rows = [bar(base + timedelta(days=k), c, c, c, c)
            for k, c in enumerate([1.10, 1.11, 1.12, 1.13, 1.14])]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    tm = daily_trend_map(df, sma_days=2)
    assert tm["2026-02-04"] == 1 and tm["2026-02-05"] == 1


def _uptrend_break_day():
    """3 prior FALLING closes (downtrend) then a day-D up-break in the overlap."""
    rows = [bar(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc) + timedelta(days=k), c, c, c, c)
            for k, c in enumerate([1.1400, 1.1300, 1.1200])]
    D = datetime(2026, 2, 4, 0, 0, tzinfo=timezone.utc)
    for (m, o, h, l, c) in [
        (780, 1.1000, 1.1000, 1.0990, 1.0996),
        (795, 1.0996, 1.1002, 1.0990, 1.0998),
        (810, 1.0999, 1.1010, 1.0998, 1.1006),
        (825, 1.1007, 1.1009, 1.1005, 1.1008),
        (840, 1.1010, 1.1016, 1.1009, 1.1015),
    ]:
        rows.append(bar(D + timedelta(minutes=m), o, h, l, c))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_trend_filter_blocks_counter_trend():
    df = _uptrend_break_day()
    cfg = {**CFG, "sma_days": 2}
    on = {"or_minutes": 30, "target_r": 1.0, "trend_filter": True}
    # trend filter OFF: the up-break trades; ON: downtrend blocks the long
    assert len(simulate_instrument(df, "EUR_USD", OFF, cfg)) == 1
    assert simulate_instrument(df, "EUR_USD", on, cfg) == []


def test_sizing_notional_cap():
    u = size_units(1.0850, 1.0830, "EUR_USD", CFG)
    assert abs(u - round(20000.0 / 1.085, 2)) < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
