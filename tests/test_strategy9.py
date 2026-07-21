"""SMC basket (strategy #9) correctness. Run: python tests/test_strategy9.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest.strategy9_smc_basket import (cap_concurrent, fractals,
                                               resample, simulate_m15)


def test_fractals():
    h = np.array([1, 2, 5, 2, 1, 2, 3, 2, 1.0])
    l = h - 0.5
    hs, ls = fractals(h, l, k=2)
    assert any(abs(v - 5.0) < 1e-9 for _, v in hs)      # the peak is found
    assert all(ci >= 2 for ci, _ in hs)                  # confirmed k bars late


def test_cap_concurrent():
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    mk = lambda s, e: (t0 + pd.Timedelta(hours=s), t0 + pd.Timedelta(hours=e), 1.0, 1.0, 1, "X")
    rows = [mk(0, 10), mk(1, 10), mk(2, 10), mk(3, 10), mk(11, 12)]
    acc = cap_concurrent(rows, max_open=3)
    assert len(acc) == 4                                 # 4th overlapping entry skipped, later one accepted


def _synth_m15(days=90, seed=7):
    rng = np.random.default_rng(seed)
    n = days * 96
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    drift = np.linspace(0, 30, n)
    noise = np.cumsum(rng.normal(0, 0.6, n))
    c = 2000 + drift + noise
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0.05, 0.6, n)
    l = np.minimum(o, c) - rng.uniform(0.05, 0.6, n)
    idx = pd.DatetimeIndex([t0 + timedelta(minutes=15 * k) for k in range(n)])
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def test_resample_shapes():
    df = _synth_m15(10)
    h1, h4, d1 = resample(df, "1h"), resample(df, "4h"), resample(df, "1D")
    assert len(h1) > len(h4) > len(d1) >= 9
    assert abs(h1["high"].max() - df["high"].max()) < 1e-9


def test_engine_runs_no_lookahead_crash():
    trades = simulate_m15(_synth_m15())
    assert isinstance(trades, list)
    for ets, xts, pnl, risk, d in trades:
        assert xts >= ets and risk > 0 and d in (1, -1)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
