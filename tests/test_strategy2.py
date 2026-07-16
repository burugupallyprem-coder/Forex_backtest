"""Strategy #2 sweep correctness. Run: python tests/test_strategy2.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy2_sweep import (RSI_LONG_MAX, _line_value, gen_sweep,
                                          gen_triangle, gen_zone, prep, simulate)

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"USD_JPY": 0.008},
    "min_stop_cost_mult": 2.0,
}


def bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 10}


def two_days(day2):
    """Day 1 sets prev levels hi=150 lo=149; day 2 = provided (minute, o,h,l,c)."""
    rows = []
    d1 = datetime(2026, 3, 2, tzinfo=timezone.utc)
    for k in range(96):
        rows.append(bar(d1 + timedelta(minutes=15 * k), 149.5, 149.6, 149.4, 149.5))
    rows[30] = bar(d1 + timedelta(minutes=450), 149.5, 150.0, 149.4, 149.6)
    rows[60] = bar(d1 + timedelta(minutes=900), 149.5, 149.6, 149.0, 149.4)
    d2 = d1 + timedelta(days=1)
    for spec in day2:
        rows.append(bar(d2 + timedelta(minutes=spec[0]), *spec[1:]))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return prep(df)


def test_prep_indicators():
    df = two_days([(9 * 60, 149.5, 149.6, 149.4, 149.5)])
    assert df["atr"].iloc[-1] > 0
    assert 0 <= df["rsi"].iloc[-1] <= 100
    assert "minute" in df and "date" in df


def test_sweep_needs_pierce_not_touch():
    from src.backtest.prevday_fib_fx import prev_day_levels
    # touch only (low == prev_lo exactly): NOT a sweep at frac 0.05 (needs 149.0-0.05 = 148.95)
    df = two_days([(9 * 60, 149.10, 149.15, 149.00, 149.06)])
    levels = prev_day_levels(df)
    i = len(df) - 1
    assert gen_sweep(df, i, {}, {"sweep_frac": 0.05}, levels) is None
    # decisive pierce to 148.90 then close back above 149.0 -> long signal
    df2 = two_days([(9 * 60, 149.10, 149.15, 148.90, 149.06)])
    sig = gen_sweep(df2, len(df2) - 1, {}, {"sweep_frac": 0.05}, prev_day_levels(df2))
    assert sig is not None and sig["side"] == 1
    assert sig["stop"] < 148.90                      # beyond the sweep extreme
    assert abs(sig["t_struct"] - (149.0 + 0.382)) < 1e-9


def test_triangle_compression_breakout():
    # 20 wide-range bars then 10 tight bars then breakout close above tight high
    rows = []
    t0 = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
    for k in range(20):
        rows.append(bar(t0 + timedelta(minutes=15 * k), 149.5, 150.4, 148.6, 149.5))
    for k in range(20, 30):
        rows.append(bar(t0 + timedelta(minutes=15 * k), 149.5, 149.62, 149.42, 149.5))
    rows.append(bar(t0 + timedelta(minutes=15 * 30), 149.5, 149.9, 149.5, 149.85))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = prep(df)
    sig = gen_triangle(df, len(df) - 1, {}, {"contraction_frac": 0.5}, {})
    assert sig is not None and sig["side"] == 1
    assert sig["reason"] == "tri_break_long"


def test_zone_supply_tap_short():
    # swing high at j (with enough history for ATR) + strong impulse away ->
    # later tap back into the zone that rejects -> short signal
    rows = []
    t0 = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
    for k in range(15):                                                          # ATR warmup
        rows.append(bar(t0 + timedelta(minutes=15 * k), 149.5, 149.7, 149.3, 149.5))
    rows.append(bar(t0 + timedelta(minutes=15 * 15), 149.6, 150.2, 149.5, 149.9))  # swing hi j=15
    for k, px in [(16, 149.2), (17, 148.8), (18, 148.4)]:                        # impulse away
        rows.append(bar(t0 + timedelta(minutes=15 * k), px + 0.2, px + 0.3, px - 0.1, px))
    for k in range(19, 25):                                                      # drift low
        rows.append(bar(t0 + timedelta(minutes=15 * k), 148.4, 148.6, 148.2, 148.4))
    rows.append(bar(t0 + timedelta(minutes=15 * 25), 148.5, 149.6, 148.4, 149.2))  # tap + reject
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = prep(df)
    sig = gen_zone(df, len(df) - 1, {}, {"impulse_atr": 1.5}, {})
    assert sig is not None and sig["side"] == -1
    assert sig["reason"] == "zone_short"


def test_line_value_interpolation():
    df = pd.DataFrame({"high": [150.0, 0, 0, 0, 149.0], "low": [0] * 5})
    assert abs(_line_value(df, 0, 4, "high", 8) - 148.0) < 1e-9


def test_engine_rsi_gate_blocks_long():
    # a valid sweep-long signal but preceding bars all rising -> RSI high -> blocked
    riser = [(8 * 60 + 15 * k, 149.0 + 0.02 * k, 149.05 + 0.02 * k,
              148.98 + 0.02 * k, 149.03 + 0.02 * k) for k in range(20)]
    sweep_bar = (13 * 60, 149.45, 149.5, 148.90, 149.06)
    entry_bar = (13 * 60 + 15, 149.10, 149.2, 149.0, 149.15)
    df = two_days(riser + [sweep_bar, entry_bar])
    assert df["rsi"].iloc[-2] > RSI_LONG_MAX          # gate condition holds
    trades = simulate(df, "USD_JPY", "sweep",
                      {"sweep_frac": 0.05, "target_mode": "rr2", "max_trades_day": 2}, CFG)
    assert trades == []                               # blocked by RSI


def test_engine_takes_gated_trade():
    # falling bars -> RSI low -> the same sweep signal passes and fills next bar
    faller = [(8 * 60 + 15 * k, 149.8 - 0.03 * k, 149.85 - 0.03 * k,
               149.75 - 0.03 * k, 149.78 - 0.03 * k) for k in range(20)]
    sweep_bar = (13 * 60, 149.2, 149.25, 148.90, 149.06)
    entry_bar = (13 * 60 + 15, 149.10, 149.2, 149.0, 149.15)
    later = (13 * 60 + 30, 149.15, 150.0, 149.1, 149.9)
    df = two_days(faller + [sweep_bar, entry_bar, later])
    trades = simulate(df, "USD_JPY", "sweep",
                      {"sweep_frac": 0.05, "target_mode": "rr2", "max_trades_day": 2}, CFG)
    assert len(trades) == 1
    assert trades[0].signal_reason == "sweep_long"
    assert trades[0].exit_reason in ("target", "data_end")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
