"""FX majors ORB port (strategy #12) correctness. Run: python tests/test_strategy12_fx_orb.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy12_fx_orb import build_context, simulate_pair, usd_side

SIM = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "half_spread": {"USD_JPY": 0.008},
    "min_stop_cost_mult": 2.0, "rr": 1.5, "open_bars": 3, "sopen": 810,
    "entry_cutoff_hours": 3, "hold_hours": 8, "min_or_width_frac": 0.0008,
}
D = datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)


def _bar(m, o, h, l, c):
    return {"ts": D + timedelta(minutes=m), "open": o, "high": h, "low": l, "close": c, "volume": 10}


def _df(rows):
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _jpy_breakout():
    # OR (810/815/820): high 150.20 low 149.95 open 150.00; up-break at 825; target at 835
    return _df([
        _bar(810, 150.00, 150.10, 149.95, 150.05),
        _bar(815, 150.05, 150.15, 150.00, 150.10),
        _bar(820, 150.10, 150.20, 150.05, 150.15),
        _bar(825, 150.16, 150.30, 150.15, 150.25),
        _bar(830, 150.26, 150.35, 150.24, 150.30),
        _bar(835, 150.27, 150.80, 150.26, 150.75),
    ])


def _ctx(orh=150.20, orl=149.95, oropen=150.00, ranked=("USD_JPY",), aligned=1, rdir=1):
    return {"2026-03-03": {"regime_dir": rdir, "ranked": list(ranked),
                           "aligned": {"USD_JPY": aligned},
                           "levels": {"USD_JPY": {"orh": orh, "orl": orl, "oropen": oropen}}}}


OFF = {"regime_filter": False, "rs_topk": None}


def test_usd_side():
    assert usd_side("USD_JPY") == 1 and usd_side("EUR_USD") == -1


def test_build_context_regime_and_rank():
    eur = _df([_bar(810, 1.1000, 1.1005, 1.0998, 1.1002), _bar(815, 1.1002, 1.1010, 1.1000, 1.1008),
               _bar(820, 1.1008, 1.1022, 1.1006, 1.1020)])          # EUR up -> USD down
    jpy = _df([_bar(810, 150.00, 150.10, 149.95, 150.05), _bar(815, 150.05, 150.2, 150.0, 150.15),
               _bar(820, 150.15, 150.35, 150.10, 150.30)])          # JPY-pair up -> USD up (stronger)
    ctx = build_context({"EUR_USD": eur, "USD_JPY": jpy}, 810, 3)
    day = ctx["2026-03-03"]
    assert day["regime_dir"] == 1                 # USD net up
    assert day["ranked"][0] == "USD_JPY"          # strongest USD mover
    assert day["aligned"]["EUR_USD"] == -1 and day["aligned"]["USD_JPY"] == 1


def test_fxorb_long_hits_target():
    trades = simulate_pair(_jpy_breakout(), "USD_JPY", OFF, SIM, _ctx())
    assert len(trades) == 1
    t = trades[0]
    assert t.signal_reason == "fxorb_break_long" and t.exit_reason == "target"
    assert abs(t.stop - 149.95) < 1e-6 and t.r_multiple > 1.4 and t.pnl > 0


def test_vol_floor_blocks_narrow_range():
    # ctx range width 0.05/150 ~ 0.0003 < 0.0008 floor -> blocked
    assert simulate_pair(_jpy_breakout(), "USD_JPY", OFF, SIM,
                         _ctx(orh=150.05, orl=150.00)) == []


def test_regime_filter_blocks_counter_aligned():
    on = {"regime_filter": True, "rs_topk": None}
    # aligned = -1 (only shorts permitted) but the tape gives an up-break -> no trade
    assert simulate_pair(_jpy_breakout(), "USD_JPY", on, SIM, _ctx(aligned=-1)) == []
    assert len(simulate_pair(_jpy_breakout(), "USD_JPY", on, SIM, _ctx(aligned=1))) == 1


def test_rs_topk_excludes_weak_pair():
    on = {"regime_filter": False, "rs_topk": 1}
    # USD_JPY ranked 2nd -> excluded when topk=1
    assert simulate_pair(_jpy_breakout(), "USD_JPY", on, SIM,
                         _ctx(ranked=("EUR_USD", "USD_JPY"))) == []
    assert len(simulate_pair(_jpy_breakout(), "USD_JPY", on, SIM,
                             _ctx(ranked=("USD_JPY", "EUR_USD")))) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
