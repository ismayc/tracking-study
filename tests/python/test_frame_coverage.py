"""Unit tests for python/09_frame_coverage.py on synthetic frames."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def fc():
    return load_script("python/09_frame_coverage.py", "tracking_frame_coverage")


def _frames(periods: dict[int, tuple[float, float]], repeat: int = 1) -> pl.DataFrame:
    """25 Hz clock values from start down to end per period, each repeated
    `repeat` times (one row per entity, as the parsed moments have)."""
    rows = []
    for period, (start, end) in periods.items():
        clocks = np.round(np.arange(start, end, -0.04), 2)
        rows += [(period, float(c)) for c in clocks for _ in range(repeat)]
    return pl.DataFrame(rows, schema={"period": pl.Int64, "game_clock": pl.Float64},
                        orient="row")


def test_full_game_covers_every_period(fc):
    per = fc.coverage(_frames({p: (720.0, 0.0) for p in (1, 2, 3, 4)}, repeat=3))
    assert per["coverage"].to_list() == pytest.approx([1.0] * 4)
    assert per["start_gap"].to_list() == pytest.approx([0.0] * 4)
    assert fc.judge("g", per)["passed"]


def test_overtime_uses_a_300_second_period(fc):
    per = fc.coverage(_frames({1: (720.0, 0.0), 5: (300.0, 0.0)}))
    assert per.filter(pl.col("period") == 5)["expected"][0] == 300 * 25
    assert per.filter(pl.col("period") == 5)["coverage"][0] == pytest.approx(1.0)


def test_missing_opening_seconds_fail_the_start_gate(fc):
    # the HOU at SAS shape: a period whose frames begin 43 s into the clock
    per = fc.coverage(_frames({1: (677.0, 0.0), 2: (720.0, 0.0),
                               3: (720.0, 0.0), 4: (720.0, 0.0)}))
    verdict = fc.judge("g", per)
    assert not verdict["passed"]
    assert any("start gap" in g and not ok for g, ok in verdict["gates"].items())


def test_excluded_game_prints_instead_of_passing_silently(fc, tmp_path, monkeypatch, capsys):
    moments = tmp_path / "moments"
    moments.mkdir()
    _frames({1: (400.0, 0.0), 2: (720.0, 0.0), 3: (720.0, 0.0),
             4: (720.0, 0.0)}).write_parquet(moments / "short.parquet")
    monkeypatch.setattr(fc, "MOMENTS", moments)
    monkeypatch.setattr(fc, "OUT", tmp_path)
    monkeypatch.setattr(fc, "WORKLOAD_EXCLUDED", {"short": "test reason"})
    assert fc.main([]) == 0
    out = capsys.readouterr().out
    assert "EXCLUDED short:" in out and "out of workload: test reason" in out
    monkeypatch.setattr(fc, "WORKLOAD_EXCLUDED", {})
    assert fc.main([]) == 1


def test_check_fails_when_workload_flags_disagree(fc, tmp_path, monkeypatch, capsys):
    moments = tmp_path / "moments"
    moments.mkdir()
    _frames({1: (400.0, 0.0), 2: (720.0, 0.0), 3: (720.0, 0.0),
             4: (720.0, 0.0)}).write_parquet(moments / "short.parquet")
    monkeypatch.setattr(fc, "MOMENTS", moments)
    monkeypatch.setattr(fc, "OUT", tmp_path)
    monkeypatch.setattr(fc, "WORKLOAD_EXCLUDED", {"short": "test reason"})
    assert fc.main([]) == 0  # writes frame_coverage.csv
    pl.DataFrame({"game": ["short"], "in_workload": [True]}).write_csv(
        tmp_path / "player_workload.csv")
    assert fc.main(["--check"]) == 1
    assert "rerun 03_analysis.py" in capsys.readouterr().out
    pl.DataFrame({"game": ["short"], "in_workload": [False]}).write_csv(
        tmp_path / "player_workload.csv")
    assert fc.main(["--check"]) == 0
