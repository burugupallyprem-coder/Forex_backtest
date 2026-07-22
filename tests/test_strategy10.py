"""3-step boring scalp (strategy #10) correctness. Run: python tests/test_strategy10.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy10_boring_scalp import (daily_ema_dir, opening_ranges,
                                                  pip_size, prev_day_levels,
                                                  simulate_instrument, size_units)

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"USD_JPY": 0.008, "EUR_USD": 0.00007},
    "min_stop_cost_mult": 2.0,
    "ny_open_min": 810,      # 13:30 UTC
    "or_minutes": 5,
    "flat_min": 1260,        # 21:00 UTC
    "ema_fast": 9,
    "ema_slow": 21,
    "zone_frac": 0.05,
    "stop_buf_frac": 0.10,
}
OFF = {"trail_lookback": 3, "trend_filter": False, "max_trades_day": 2}


def bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 10}


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# Day 1 = previous trading day; sets PDH=1.2000 / PDL=1.1900 far from day-2 price
# so previous-day levels never fire and we isolate the 5-min opening range.
D1 = datetime(2026, 3, 2, 13, 30, tzinfo=timezone.utc)
D2 = datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)


def _prev_day():
    return [bar(D1 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in [
        (0, 1.1950, 1.2000, 1.1900, 1.1950),
        (1, 1.1950, 1.1990, 1.1910, 1.1960),
    ]]


def _or_bars():
    # OR window (minutes 810-814): OR high 1.1010, OR low 1.1000
    return [bar(D2 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in [
        (810, 1.1005, 1.1010, 1.1000, 1.1008),
        (811, 1.1008, 1.1009, 1.1002, 1.1005),
        (812, 1.1005, 1.1008, 1.1003, 1.1006),
        (813, 1.1006, 1.1009, 1.1004, 1.1007),
        (814, 1.1007, 1.1008, 1.1005, 1.1007),
    ]]


def _break_retest_trigger_fill():
    # band = OR_high + zone = 1.1010 + 0.0005 = 1.10150
    return [bar(D2 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in [
        (815, 1.1012, 1.1020, 1.1011, 1.1018),   # breakout: close > band
        (816, 1.1017, 1.1019, 1.1014, 1.1016),   # retest: low touches band
        (817, 1.1015, 1.1025, 1.1015, 1.1022),   # trigger: strong close > band
        (818, 1.1023, 1.1024, 1.1021, 1.1023),   # fill next bar open
    ]]


def test_basics():
    assert pip_size("USD_JPY") == 0.01 and pip_size("EUR_USD") == 0.0001


def test_levels_helpers():
    df = _df(_prev_day() + _or_bars())
    pdl = prev_day_levels(df)
    assert abs(pdl["2026-03-03"][0] - 1.2000) < 1e-9   # PDH from day 1
    assert abs(pdl["2026-03-03"][1] - 1.1900) < 1e-9   # PDL from day 1
    orl = opening_ranges(df, 810, 5)
    assert abs(orl["2026-03-03"][0] - 1.1010) < 1e-9
    assert abs(orl["2026-03-03"][1] - 1.1000) < 1e-9


def test_or_long_hits_initial_stop():
    rows = _prev_day() + _or_bars() + _break_retest_trigger_fill()
    rows.append(bar(D2 + timedelta(minutes=819), 1.1022, 1.1023, 1.0990, 1.0995))
    trades = simulate_instrument(_df(rows), "EUR_USD", OFF, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "orh_break_retest_long"
    assert t.exit_reason == "stop"
    assert abs(t.stop - 1.1000) < 1e-6      # initial stop = OR_high - buf
    assert t.r_multiple < 0 and t.pnl < 0


def test_or_long_trails_to_profit():
    rows = _prev_day() + _or_bars() + _break_retest_trigger_fill()
    rally = [
        (819, 1.1025, 1.1040, 1.1030, 1.1038),
        (820, 1.1038, 1.1055, 1.1040, 1.1052),
        (821, 1.1052, 1.1065, 1.1050, 1.1062),
        (822, 1.1062, 1.1075, 1.1060, 1.1072),
        (823, 1.1072, 1.1085, 1.1070, 1.1082),
        (824, 1.1080, 1.1082, 1.1035, 1.1040),   # pullback hits trailed stop
    ]
    rows += [bar(D2 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in rally]
    trades = simulate_instrument(_df(rows), "EUR_USD", OFF, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "trail_stop"
    assert t.r_multiple > 0 and t.pnl > 0


def test_trend_filter_blocks_when_no_trend():
    # Only 2 days of data -> daily EMA undefined -> direction 0 -> trend filter blocks all.
    rows = _prev_day() + _or_bars() + _break_retest_trigger_fill()
    rows.append(bar(D2 + timedelta(minutes=819), 1.1022, 1.1023, 1.0990, 1.0995))
    on = {"trail_lookback": 3, "trend_filter": True, "max_trades_day": 2}
    assert len(simulate_instrument(_df(rows), "EUR_USD", OFF, CFG)) == 1
    assert simulate_instrument(_df(rows), "EUR_USD", on, CFG) == []


def test_pre_ny_open_ignored():
    early = [bar(D2 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in [
        (700, 1.1012, 1.1020, 1.1011, 1.1018),
        (701, 1.1017, 1.1019, 1.1014, 1.1016),
        (702, 1.1015, 1.1025, 1.1015, 1.1022),
        (703, 1.1023, 1.1024, 1.1021, 1.1023),
        (704, 1.1022, 1.1030, 1.1000, 1.1005),
    ]]
    assert simulate_instrument(_df(_prev_day() + early), "EUR_USD", OFF, CFG) == []


def test_daily_ema_dir_up_and_down():
    up = [bar(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=k),
              1.10 + 0.001 * k, 1.10 + 0.001 * k, 1.10 + 0.001 * k, 1.10 + 0.001 * k)
          for k in range(30)]
    dn = [bar(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=k),
              1.40 - 0.001 * k, 1.40 - 0.001 * k, 1.40 - 0.001 * k, 1.40 - 0.001 * k)
          for k in range(30)]
    du = daily_ema_dir(_df(up), 9, 21)
    dd = daily_ema_dir(_df(dn), 9, 21)
    assert du["2026-01-30"] == 1
    assert dd["2026-01-30"] == -1


def test_sizing_notional_cap():
    u = size_units(1.0850, 1.0830, "EUR_USD", CFG)
    assert abs(u - round(20000.0 / 1.085, 2)) < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
