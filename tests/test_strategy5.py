"""Regime-conditional trend (strategy #5) correctness. Run: python tests/test_strategy5.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy5_regime_trend import (efficiency_ratio, pip_size,
                                                 simulate_instrument, size_units)

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"EUR_USD": 0.00007},
    "min_stop_cost_mult": 2.0,
    "fast": 2, "slow": 4, "er_len": 3, "atr_len": 2, "atr_k": 1.0,
}
UPTREND = [1.00, 1.00, 1.00, 1.00, 1.02, 1.05, 1.08, 1.11, 1.14, 1.17, 1.20]


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
    assert pip_size("EUR_USD") == 0.0001
    assert abs(efficiency_ratio(pd.Series([1.0, 2, 3, 4, 5]), 4).iloc[-1] - 1.0) < 1e-9
    assert efficiency_ratio(pd.Series([1.0, 2, 1, 2, 1]), 4).iloc[-1] < 0.5


def test_gate_allows_trend():
    # clean uptrend has ER=1.0; a permissive gate lets the long through and it profits
    trades = simulate_instrument(daily(UPTREND), "EUR_USD", {"er_min": 0.5}, CFG)
    longs = [t for t in trades if "long" in t.signal_reason]
    assert longs and longs[0].pnl > 0


def test_gate_blocks_at_unreachable_threshold():
    # SAME uptrend (direction is defined, entry would fire) but require ER >= 1.01
    # (impossible) -> proves the regime gate is actually enforced on entries
    assert simulate_instrument(daily(UPTREND), "EUR_USD", {"er_min": 1.01}, CFG) == []


def test_needs_history():
    assert simulate_instrument(daily([1.00, 1.01, 1.02]), "EUR_USD", {"er_min": 0.25}, CFG) == []


def test_sizing_cap():
    u = size_units(1.0850, 1.0830, "EUR_USD", CFG)
    assert abs(u - round(20000.0 / 1.085, 2)) < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
