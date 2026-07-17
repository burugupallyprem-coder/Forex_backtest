"""Daily FX trend (strategy #4) correctness. Run: python tests/test_strategy4.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy4_daily_trend import (pip_size, simulate_instrument,
                                                size_units)

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"USD_JPY": 0.008, "EUR_USD": 0.00007},
    "min_stop_cost_mult": 2.0,
    "atr_len": 2,
    "atr_k": 1.0,
}
P = {"fast": 2, "slow": 4}


def bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 10}


def daily(closes, start=datetime(2020, 1, 1, tzinfo=timezone.utc)):
    rows = []
    for k, c in enumerate(closes):
        o = closes[k - 1] if k > 0 else c
        rows.append(bar(start + timedelta(days=k), o, max(o, c) + 0.004, min(o, c) - 0.004, c))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_basics():
    assert pip_size("USD_JPY") == 0.01 and pip_size("EUR_USD") == 0.0001


def test_uptrend_goes_long_and_profits():
    df = daily([1.00, 1.00, 1.00, 1.00, 1.02, 1.05, 1.08, 1.11, 1.14, 1.17, 1.20])
    trades = simulate_instrument(df, "EUR_USD", P, CFG)
    longs = [t for t in trades if t.signal_reason == "ma2x4_long"]
    assert longs and longs[0].pnl > 0


def test_downtrend_goes_short_and_profits():
    df = daily([1.20, 1.20, 1.20, 1.20, 1.18, 1.15, 1.12, 1.09, 1.06, 1.03, 1.00])
    trades = simulate_instrument(df, "EUR_USD", P, CFG)
    shorts = [t for t in trades if t.signal_reason == "ma2x4_short"]
    assert shorts and shorts[0].pnl > 0


def test_flip_reverses_direction():
    df = daily([1.00, 1.00, 1.00, 1.00, 1.03, 1.06, 1.09, 1.06, 1.03, 1.00, 0.97, 0.94])
    reasons = [t.signal_reason for t in simulate_instrument(df, "EUR_USD", P, CFG)]
    assert any("long" in r for r in reasons) and any("short" in r for r in reasons)


def test_needs_history_no_lookahead():
    assert simulate_instrument(daily([1.00, 1.01, 1.02]), "EUR_USD", P, CFG) == []


def test_sizing_cap():
    u = size_units(1.0850, 1.0830, "EUR_USD", CFG)
    assert abs(u - round(20000.0 / 1.085, 2)) < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
