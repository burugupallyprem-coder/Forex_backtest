"""News-reaction (strategy #7) correctness. Run: python tests/test_strategy7.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy7_news_reaction import (pip_size, simulate_instrument,
                                                  size_units)

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"EUR_USD": 0.00007},
    "atr_len": 3, "vol_len": 3, "vol_mult": 3.0, "stop_atr": 1.0,
    "news_spread_mult": 2.0,
}
P = {"spike_k": 2.5, "hold_bars": 4}
CALM = (1.1000, 1.1002, 1.0998, 1.1000, 10)
UPSPIKE = [CALM] * 6 + [
    (1.1000, 1.1032, 1.0999, 1.1030, 100),   # up spike (big range + volume)
    (1.1030, 1.1052, 1.1028, 1.1050, 20),
    (1.1050, 1.1072, 1.1048, 1.1070, 20),
    (1.1070, 1.1092, 1.1068, 1.1090, 20),
    (1.1090, 1.1112, 1.1088, 1.1110, 20),
    (1.1110, 1.1132, 1.1108, 1.1130, 20),
]


def m15(rows, start=datetime(2023, 1, 2, 13, 0, tzinfo=timezone.utc)):
    data = [{"ts": start + timedelta(minutes=15 * k), "open": o, "high": h,
             "low": l, "close": c, "volume": v} for k, (o, h, l, c, v) in enumerate(rows)]
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_basics():
    assert pip_size("AUD_USD") == 0.0001 and pip_size("USD_JPY") == 0.01


def test_no_spike_no_trade():
    assert simulate_instrument(m15([CALM] * 13), "EUR_USD", P, CFG) == []


def test_spike_triggers_continuation():
    trades = simulate_instrument(m15(UPSPIKE), "EUR_USD", P, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "news_up"
    assert t.pnl > 0 and t.r_multiple > 0


def test_news_spread_widens_cost():
    df = m15(UPSPIKE)
    lo = simulate_instrument(df, "EUR_USD", P, {**CFG, "news_spread_mult": 1.0})
    hi = simulate_instrument(df, "EUR_USD", P, {**CFG, "news_spread_mult": 50.0})
    assert lo and hi
    assert hi[0].pnl < lo[0].pnl   # wider news spread -> worse entry -> less profit


def test_sizing_cap():
    u = size_units(1.0850, 1.0830, "EUR_USD", CFG)
    assert abs(u - round(20000.0 / 1.085, 2)) < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
