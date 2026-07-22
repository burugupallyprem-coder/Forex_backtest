"""Strategy #10 diagnostic helpers smoke test. Run: python tests/test_strategy10_diag.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.strategy10_diag import _grp, _level_family


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
