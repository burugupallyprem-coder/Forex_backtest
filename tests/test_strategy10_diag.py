"""Strategy #10 diagnostic helpers smoke test. Run: python tests/test_strategy10_diag.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy10_boring_scalp import Trade
from src.backtest.strategy10_diag import _grp, _level_family, _trade_breakdown


def _mk(date, r, exit_reason, signal_reason):
    return Trade(symbol="XAU_USD", strategy="s10", date=date, entry_time="", exit_time="",
                 entry=1.0, exit=1.0, shares=1.0, stop=1.0, target=0.0, pnl=r,
                 r_multiple=r, exit_reason=exit_reason, signal_reason=signal_reason)


def test_trade_breakdown_runs_end_to_end():
    trades = [
        _mk("2025-01-06", 1.2, "trail_stop", "orh_break_retest_long"),
        _mk("2025-02-10", -1.0, "stop", "pdl_break_retest_short"),
        _mk("2025-05-12", 0.4, "trail_stop", "pdh_break_retest_long"),
        _mk("2025-08-18", -1.0, "stop", "orl_break_retest_short"),
    ]
    lines, pos_q, tot_q = _trade_breakdown(trades)
    joined = "\n".join(lines)
    assert "by quarter:" in joined
    assert "by exit reason:" in joined
    assert "by level family:" in joined
    assert "by direction:" in joined
    assert "long:" in joined and "short:" in joined       # the line that used to crash
    assert tot_q == 3 and 0 <= pos_q <= 3     # Jan+Feb share Q1 -> 3 distinct quarters


def test_trade_breakdown_empty():
    assert _trade_breakdown([]) == (["  (no trades)"], 0, 0)


def test_level_family():
    assert _level_family("pdh_break_retest_long") == "prev_day_level"
    assert _level_family("pdl_break_retest_short") == "prev_day_level"
    assert _level_family("orh_break_retest_long") == "opening_range"
    assert _level_family("orl_break_retest_short") == "opening_range"
    assert _level_family("") == "other"


def test_grp_runs_and_counts():
    df = pd.DataFrame({
        "r_multiple": [1.0, -1.0, 0.5, -1.0],
        "exit_reason": ["target", "stop", "trail_stop", "stop"],
    })
    lines = _grp(df, lambda r: r["exit_reason"], "exit reason")
    assert lines[0] == "  by exit reason:"
    joined = "\n".join(lines)
    assert "stop: 2 trades" in joined
    assert _grp(df.iloc[0:0], "exit_reason", "exit reason") == ["  by exit reason: (no trades)"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
