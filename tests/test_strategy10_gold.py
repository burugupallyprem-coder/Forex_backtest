"""Gold volatility-floor tune correctness. Run: python tests/test_strategy10_gold.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy10_boring_scalp import Trade, simulate_instrument
from src.backtest.strategy10_gold import bootstrap_ci, walk_forward

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"EUR_USD": 0.00007},
    "min_stop_cost_mult": 2.0,
    "ny_open_min": 810, "or_minutes": 5, "flat_min": 1260,
    "ema_fast": 9, "ema_slow": 21, "zone_frac": 0.05, "stop_buf_frac": 0.10,
}
BASE = {"trail_lookback": 20, "trend_filter": False, "max_trades_day": 1}
D1 = datetime(2026, 3, 2, 13, 30, tzinfo=timezone.utc)
D2 = datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)


def _bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 10}


def _scenario():
    # OR high 1.1010 / low 1.1000 -> width 0.0010, price ~1.101 -> frac ~0.00091
    rows = [_bar(D1 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in [
        (0, 1.1950, 1.2000, 1.1900, 1.1950), (1, 1.1950, 1.1990, 1.1910, 1.1960)]]
    rows += [_bar(D2 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in [
        (810, 1.1005, 1.1010, 1.1000, 1.1008), (811, 1.1008, 1.1009, 1.1002, 1.1005),
        (812, 1.1005, 1.1008, 1.1003, 1.1006), (813, 1.1006, 1.1009, 1.1004, 1.1007),
        (814, 1.1007, 1.1008, 1.1005, 1.1007),
        (815, 1.1012, 1.1020, 1.1011, 1.1018), (816, 1.1017, 1.1019, 1.1014, 1.1016),
        (817, 1.1015, 1.1025, 1.1015, 1.1022), (818, 1.1023, 1.1024, 1.1021, 1.1023),
        (819, 1.1022, 1.1023, 1.0990, 1.0995)]]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_floor_off_or_low_allows_trade():
    df = _scenario()
    assert len(simulate_instrument(df, "EUR_USD", BASE, {**CFG, "min_or_width_frac": None})) == 1
    assert len(simulate_instrument(df, "EUR_USD", BASE, {**CFG, "min_or_width_frac": 0.0005})) == 1


def test_floor_high_blocks_dead_open():
    df = _scenario()
    # OR width frac ~0.00091 < 0.0020 -> the volatility floor rejects the day
    assert simulate_instrument(df, "EUR_USD", BASE, {**CFG, "min_or_width_frac": 0.0020}) == []


def _mk(date, r):
    return Trade("XAU_USD", "s", date, "", "", 1, 1, 1, 1, 0, r, r, "stop", "orh_break_retest_long")


def test_walk_forward_counts_folds():
    trades = [_mk(f"2022-{mo:02d}-05", r) for mo, r in
              [(1, 1.0), (2, 0.5), (5, -1.0), (6, 0.8), (9, 0.3), (12, -0.2)]]
    pos, tot, per = walk_forward(trades, 3)
    assert tot == 3 and len(per) == 3 and 0 <= pos <= 3
    assert walk_forward([], 4) == (0, 0, [])


def test_bootstrap_seeded_and_bounds():
    rs = [1.0, -1.0, 0.5, -1.0, 0.9, -1.0, 2.0, -1.0]
    a = bootstrap_ci(rs, n=500, seed=0)
    b = bootstrap_ci(rs, n=500, seed=0)
    assert a == b
    lo, hi, frac, pt = a
    assert lo <= pt <= hi and 0.0 <= frac <= 1.0
    assert bootstrap_ci([], n=10) == (0.0, 0.0, 0.0, 0.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
