"""Intraday stat-arb (strategy #8) correctness. Run: python tests/test_strategy8.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy8_statarb import pip_size, simulate_pair

CFG = {
    "risk": {"equity": 100000, "risk_pct": 0.5, "max_position_pct": 20},
    "spread_pips": {"AAA": 1.5, "BBB": 2.0},
    "z_exit": 0.0, "stop_z": 4.0, "max_hold": 48,
}
P = {"z_enter": 3.0, "lookback": 10}
WIGGLE = [1.0000, 1.0003, 0.9997, 1.0003, 0.9997, 1.0003, 0.9997, 1.0003, 0.9997, 1.0003, 0.9997, 1.0000]


def legs(ca, cb, start=datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc)):
    n = len(ca)
    a = pd.DataFrame({"ts": [start + timedelta(hours=k) for k in range(n)], "close": ca})
    b = pd.DataFrame({"ts": [start + timedelta(hours=k) for k in range(n)], "close": cb})
    a["ts"] = pd.to_datetime(a["ts"], utc=True)
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    return a, b


def test_basics():
    assert pip_size("AUD_USD") == 0.0001


def test_flat_no_trade():
    ca = [1.0000, 1.0003, 0.9997] * 6
    a, b = legs(ca, [1.0000] * len(ca))
    assert simulate_pair(a, b, "AAA", "BBB", P, CFG) == []


def test_reversion_gross_profit():
    ca = WIGGLE + [1.0100, 1.0000, 1.0000, 1.0000]   # spread stretches, then reverts
    a, b = legs(ca, [1.0000] * len(ca))
    tr = simulate_pair(a, b, "AAA", "BBB", P, {**CFG, "cost_off": True})
    assert len(tr) >= 1
    assert tr[0].exit_reason == "revert" and tr[0].pnl > 0


def test_cost_reduces_pnl():
    ca = WIGGLE + [1.0100, 1.0000, 1.0000, 1.0000]
    a, b = legs(ca, [1.0000] * len(ca))
    g = simulate_pair(a, b, "AAA", "BBB", P, {**CFG, "cost_off": True})
    net = simulate_pair(a, b, "AAA", "BBB", P, {**CFG, "cost_off": False})
    assert g and net and net[0].pnl < g[0].pnl


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
