"""Unit tests for tracking-study/python/03_analysis.py geometry and filters."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def test_hull_area_unit_square(tracking):
    xs = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
    ys = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    assert tracking.hull_area(xs, ys) == pytest.approx(1.0)


def test_hull_area_collinear_is_nan(tracking):
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 2.0])
    assert np.isnan(tracking.hull_area(xs, ys))


def _moments(rows):
    return pl.DataFrame(rows, schema={
        "period": pl.Int64, "game_clock": pl.Float64, "entity": pl.Utf8,
        "team_id": pl.Int64, "player_id": pl.Int64,
        "x": pl.Float64, "y": pl.Float64, "z": pl.Float64,
    })


def test_workload_distance_and_filters(tracking):
    # Player 1 moves 0.8 ft per 0.04 s for 3 steps (20 ft/s, kept),
    # then a 50 ft teleport in one frame (glitch, dropped),
    # then a step after a 1.0 s clock gap (clock stopped, dropped).
    clocks = [720.0, 719.96, 719.92, 719.88, 719.84, 718.84]
    xs = [0.0, 0.8, 1.6, 2.4, 52.4, 53.2]
    rows = [{"period": 1, "game_clock": c, "entity": "player", "team_id": 1,
             "player_id": 1, "x": x, "y": 0.0, "z": 0.0}
            for c, x in zip(clocks, xs)]
    roster = pl.DataFrame({"player_id": [1], "player": ["Test Player"]})
    w = tracking.player_workload(_moments(rows), roster).row(0, named=True)
    assert w["dist_ft"] == pytest.approx(2.4)          # 3 valid 0.8-ft steps
    assert w["live_seconds"] == pytest.approx(0.12)    # 3 * 0.04
    assert w["max_speed"] <= tracking.MAX_SPEED_FTS


def test_possession_nearest_player_and_ball_height(tracking):
    def frame(clock, ball_z, near_team):
        return [
            {"period": 1, "game_clock": clock, "entity": "ball", "team_id": -1,
             "player_id": -1, "x": 10.0, "y": 10.0, "z": ball_z},
            {"period": 1, "game_clock": clock, "entity": "player", "team_id": near_team,
             "player_id": 1, "x": 11.0, "y": 10.0, "z": 0.0},
            {"period": 1, "game_clock": clock, "entity": "player", "team_id": 99,
             "player_id": 2, "x": 20.0, "y": 10.0, "z": 0.0},
        ]
    rows = frame(720.0, 5.0, 7) + frame(719.0, 12.0, 7)   # 2nd: ball too high
    poss = tracking.possession_frames(_moments(rows))
    assert poss.height == 1                    # high-ball frame excluded
    r = poss.row(0, named=True)
    assert r["off_team_id"] == 7
    assert r["handler_dist"] == pytest.approx(1.0)


def test_possession_frames_sorted_game_order(tracking):
    rows = []
    for clock in (700.0, 710.0, 705.0):
        rows += [
            {"period": 1, "game_clock": clock, "entity": "ball", "team_id": -1,
             "player_id": -1, "x": 0.0, "y": 0.0, "z": 5.0},
            {"period": 1, "game_clock": clock, "entity": "player", "team_id": 7,
             "player_id": 1, "x": 1.0, "y": 0.0, "z": 0.0},
        ]
    poss = tracking.possession_frames(_moments(rows))
    # Deterministic game order: clock descending within period. The spacing
    # subsample takes every Nth ROW, so this order is load-bearing.
    assert poss["game_clock"].to_list() == [710.0, 705.0, 700.0]
