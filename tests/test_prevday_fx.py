"""FX prev-day fib bounce correctness. Run: python tests/test_prevday_fx.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.prevday_fib_fx import (in_session, pip_size, prev_day_levels,
                                         simulate_instrument, size_units)

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"USD_JPY": 0.008, "EUR_USD": 0.00007},
    "min_stop_cost_mult": 2.0,
}
PARAMS = {"target_fib": 0.382, "stop_buf_frac": 0.1, "max_trades_day": 2}


def bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 10}


def make_two_days(day2_bars):
    """Day 1 (full, sets hi=150.0 lo=149.0), then day 2 = provided bars."""
    rows = []
    d1 = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)
    for k in range(96):
        rows.append(bar(d1 + timedelta(minutes=15 * k), 149.5, 149.6, 149.4, 149.5))
    rows[30] = bar(d1 + timedelta(minutes=15 * 30), 149.5, 150.0, 149.4, 149.6)  # hi
    rows[60] = bar(d1 + timedelta(minutes=15 * 60), 149.5, 149.6, 149.0, 149.4)  # lo
    d2 = d1 + timedelta(days=1)
    for spec in day2_bars:
        rows.append(bar(d2 + timedelta(minutes=spec[0]), *spec[1:]))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_basics():
    assert pip_size("USD_JPY") == 0.01 and pip_size("EUR_USD") == 0.0001
    assert in_session(9 * 60) and not in_session(5 * 60)
    df = make_two_days([(9 * 60, 149.5, 149.6, 149.4, 149.5)])
    assert prev_day_levels(df)["2026-03-03"] == (150.0, 149.0)


def test_long_bounce_target_jpy():
    # touch 149.0, close back above -> long; target 149.382, stop 148.9
    df = make_two_days([
        (9 * 60,       149.10, 149.15, 148.98, 149.06),
        (9 * 60 + 15,  149.07, 149.20, 149.02, 149.15),   # entry 149.07+0.008
        (9 * 60 + 30,  149.15, 149.50, 149.10, 149.45),   # target hit
    ])
    trades = simulate_instrument(df, "USD_JPY", PARAMS, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "pd_low_bounce_long"
    assert t.exit_reason == "target"
    assert abs(t.target - 149.382) < 1e-9
    assert abs(t.stop - 148.9) < 1e-9
    assert t.pnl > 0
    # JPY conversion sanity: pnl ~ (exit-entry)*units/entry_price
    expected = (t.exit - t.entry) * t.shares / t.entry
    assert abs(t.pnl - round(expected, 2)) < 0.05


def test_asia_touch_ignored():
    df = make_two_days([
        (5 * 60,       149.10, 149.15, 148.98, 149.06),   # asia touch -> no trade
        (5 * 60 + 15,  149.07, 149.20, 149.02, 149.15),
    ])
    assert simulate_instrument(df, "USD_JPY", PARAMS, CFG) == []


def test_sizing_jpy_notional_cap():
    # USD_JPY: uncapped units would be huge; 20% notional cap -> 20,000 units
    u = size_units(149.078, 148.9, "USD_JPY", CFG)
    assert abs(u - 20000.0) < 1e-6
    # EUR_USD: cap = 20000/1.085 units
    u2 = size_units(1.0850, 1.0830, "EUR_USD", CFG)
    assert abs(u2 - round(20000.0 / 1.085, 2)) < 0.01


def test_short_bounce_stop_conservative():
    # short at prev high 150, gaps up through stop 150.1 -> fill at open (worse)
    df = make_two_days([
        (14 * 60,      149.90, 150.05, 149.85, 149.95),   # touch 150, close under
        (14 * 60 + 15, 149.95, 150.00, 149.90, 149.95),   # entry short
        (14 * 60 + 30, 150.60, 150.70, 150.50, 150.65),   # gap through stop
    ])
    trades = simulate_instrument(df, "USD_JPY", PARAMS, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "pd_high_bounce_short"
    assert t.exit_reason == "stop"
    assert t.r_multiple <= -1.0
    assert t.pnl < 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
