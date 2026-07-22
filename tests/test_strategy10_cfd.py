"""Strategy #10 CFD variant (gold + S&P) sanity. Run: python tests/test_strategy10_cfd.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy10_boring_scalp import (simulate_instrument, unit_notional_usd,
                                                  usd_pnl_factor)

# Price-unit spreads (half per side), as the CFD run builds them.
CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"XAU_USD": 0.175, "SPX500_USD": 0.30},
    "min_stop_cost_mult": 2.0,
    "ny_open_min": 810, "or_minutes": 5, "flat_min": 1260,
    "ema_fast": 9, "ema_slow": 21, "zone_frac": 0.05, "stop_buf_frac": 0.10,
}
OFF = {"trail_lookback": 3, "trend_filter": False, "max_trades_day": 2}

D1 = datetime(2026, 3, 2, 13, 30, tzinfo=timezone.utc)
D2 = datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)


def bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 10}


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_cfd_usd_helpers():
    # USD-quote CFDs: $1 per 1-point move per unit; notional = price (not USD-prefixed)
    assert usd_pnl_factor("XAU_USD", 2400.0) == 1.0
    assert usd_pnl_factor("SPX500_USD", 5000.0) == 1.0
    assert unit_notional_usd("XAU_USD", 2400.0) == 2400.0
    assert unit_notional_usd("SPX500_USD", 5000.0) == 5000.0


def test_gold_or_break_retest_long():
    # prev day sets PDH/PDL far away; isolate the 5-min opening range on gold.
    rows = [bar(D1 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in [
        (0, 2500, 2600, 2400, 2500), (1, 2500, 2590, 2410, 2510)]]
    # OR window (810-814): OR high 2451, OR low 2449  -> zone = 0.05*(2600-2400)=10
    rows += [bar(D2 + timedelta(minutes=m), o, h, l, c) for (m, o, h, l, c) in [
        (810, 2450, 2451, 2449, 2450), (811, 2450, 2451, 2449, 2450),
        (812, 2450, 2451, 2449, 2450), (813, 2450, 2451, 2449, 2450),
        (814, 2450, 2451, 2449, 2450),
        (815, 2452, 2470, 2452, 2465),   # breakout: close > OR_high + zone (2461)
        (816, 2464, 2466, 2458, 2460),   # retest: low dips into zone
        (817, 2460, 2475, 2460, 2472),   # trigger: strong close > band
        (818, 2473, 2474, 2471, 2473),   # fill
        (819, 2472, 2473, 2300, 2320),   # slams initial stop
    ]]
    trades = simulate_instrument(_df(rows), "XAU_USD", OFF, CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "orh_break_retest_long"
    assert t.exit_reason == "stop"
    assert t.r_multiple < 0 and t.pnl < 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
