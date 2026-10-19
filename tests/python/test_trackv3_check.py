"""Unit tests for python/10_trackv3_check.py on synthetic player-games."""
from __future__ import annotations

import polars as pl
import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def tv():
    return load_script("python/10_trackv3_check.py", "tracking_trackv3")


def _workload(rows):
    """rows: (game, player_id, live_minutes, dist_miles, in_workload)."""
    return pl.DataFrame(
        {"game_id": [f"00215{r[0]:05d}" for r in rows], "game": [f"G{r[0]}" for r in rows],
         "player_id": [r[1] for r in rows], "player": [f"P{r[1]}" for r in rows],
         "live_minutes": [r[2] for r in rows], "dist_miles": [r[3] for r in rows],
         "in_workload": [r[4] for r in rows]})


def _official(rows):
    """rows: (game, player_id, off_min, off_dist)."""
    return pl.DataFrame({"game_id": [f"00215{r[0]:05d}" for r in rows],
                         "player_id": [r[1] for r in rows],
                         "off_min": [r[2] for r in rows], "off_dist": [r[3] for r in rows]})


def _good():
    # two kept games, pace ratio 1.0 and dead-ball share of 5 percent
    w = _workload([(1, 1, 28.5, 2.85, True), (1, 2, 19.0, 1.90, True),
                   (2, 3, 28.5, 2.85, True), (2, 4, 19.0, 1.90, True)])
    o = _official([(1, 1, 30.0, 3.0), (1, 2, 20.0, 2.0),
                   (2, 3, 30.0, 3.0), (2, 4, 20.0, 2.0)])
    return w, o


def test_join_matches_on_game_and_player(tv):
    w, o = _good()
    cmp = tv.compare(w, o)
    assert cmp.height == 4
    assert cmp["pace_ratio"].to_list() == pytest.approx([1.0] * 4)
    assert cmp["dist_ratio"].to_list() == pytest.approx([0.95] * 4)


def test_clean_data_passes_every_gate(tv):
    w, o = _good()
    assert all(tv.gates(w, o, tv.compare(w, o)).values())


def test_wrong_distance_fails_pace_gate(tv):
    w, o = _good()
    w = w.with_columns(pl.when(pl.col("player_id") == 1).then(pl.col("dist_miles") * 1.2)
                       .otherwise(pl.col("dist_miles")).alias("dist_miles"))
    res = tv.gates(w, o, tv.compare(w, o))
    assert not all(res.values())
    assert [k for k, v in res.items() if not v and k.startswith("pace")]


def test_short_game_fails_game_distance_gate(tv):
    w, o = _good()
    w = w.with_columns((pl.col("dist_miles") * 0.9).alias("dist_miles"),
                       (pl.col("live_minutes") * 0.9).alias("live_minutes"))
    res = tv.gates(w, o, tv.compare(w, o))
    assert [k for k, v in res.items() if not v and k.startswith("game distance")]


def test_excluded_game_is_not_gated_but_is_reported(tv):
    w, o = _good()
    w = w.with_columns(pl.when(pl.col("game") == "G2").then(pl.col("dist_miles") * 0.8)
                       .otherwise(pl.col("dist_miles")).alias("dist_miles"),
                       pl.when(pl.col("game") == "G2").then(False).otherwise(True)
                       .alias("in_workload"))
    cmp = tv.compare(w, o)
    assert all(tv.gates(w, o, cmp).values())
    games = tv.game_summary(cmp)
    assert tv.excluded_consistent(games)
    assert len(tv.excluded_evidence(games)) == 1


def test_excluded_game_inside_tolerance_is_inconsistent(tv):
    w, o = _good()
    w = w.with_columns(pl.when(pl.col("game") == "G2").then(False).otherwise(True)
                       .alias("in_workload"))
    assert not tv.excluded_consistent(tv.game_summary(tv.compare(w, o)))


def test_unmatched_workload_row_fails_join_gate(tv):
    w, o = _good()
    o = o.filter(pl.col("player_id") != 4)
    res = tv.gates(w, o, tv.compare(w, o))
    assert not res["join: every workload row has an official row"]


def test_official_player_with_minutes_but_no_workload_row_fails(tv):
    w, o = _good()
    o = pl.concat([o, _official([(1, 9, 12.0, 1.0)])])
    res = tv.gates(w, o, tv.compare(w, o))
    assert not res["join: every official row with minutes has a workload row"]


def test_dnp_rows_are_not_compared(tv):
    w, o = _good()
    o = pl.concat([o, _official([(1, 9, 0.0, 0.0)])])
    res = tv.gates(w, o, tv.compare(w, o))
    assert all(res.values())
    assert tv.compare(w, o).height == 4


def test_live_minutes_above_court_time_fails(tv):
    w, o = _good()
    w = w.with_columns(pl.when(pl.col("player_id") == 2).then(21.0)
                       .otherwise(pl.col("live_minutes")).alias("live_minutes"))
    res = tv.gates(w, o, tv.compare(w, o))
    assert [k for k, v in res.items() if not v and k.startswith("live minutes")]


def test_short_minute_players_skip_the_pace_gate(tv):
    w, o = _good()
    w = pl.concat([w, _workload([(1, 8, 2.0, 0.5, True)])])
    o = pl.concat([o, _official([(1, 8, 3.0, 0.3)])])
    assert all(tv.gates(w, o, tv.compare(w, o)).values())


def test_committed_tables_pass_the_real_gate(tv):
    w, o = tv.load_workload(), tv.load_official()
    assert all(tv.gates(w, o, tv.compare(w, o)).values())
    assert set(tv.game_summary(tv.compare(w, o)).filter(~pl.col("in_workload"))["game"]) \
        == {"01.02.2016.DET.at.IND", "01.02.2016.HOU.at.SAS"}
