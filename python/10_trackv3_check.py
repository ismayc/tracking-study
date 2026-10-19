"""Per-player-game gate: our SportVU workload vs the NBA's tracking box score.

Reference data
--------------
data/nba_trackv3_boxscore_2015-16_10games.csv is a FROZEN copy of the
NBA's BoxScorePlayerTrackV3 endpoint (stats.nba.com) for the 10 games of
this study (2015-16 regular season), fetched October 19, 2026 by the
tracking-study audit (nba_api, one call per game). It cannot be refetched
from this repo, so it is committed and never rewritten by any script.
Columns (254 rows, one per player on a game's box score, DNPs included):
  game_id    10-char NBA id with leading zeros (a string, e.g. 0021500490)
  player_id  NBA personId, the same id as player_workload.csv's player_id
  off_min    minutes played, from the endpoint's "mm:ss" string, in minutes.
             This is court time on the game clock, dead balls included.
  off_dist   distance, in miles, rounded by the NBA to 0.01 mi. It also
             includes movement during dead balls.
The join key is (game_id, player_id), both identical in the two tables. All
210 workload rows match; the other 44 official rows are DNPs with 0 minutes.

What the two sides measure
--------------------------
player_workload.csv counts only LIVE play (clock running), so live_minutes
is below off_min by the share of dead-ball time (1 to 10 percent by game in the
kept games) and dist_miles is correspondingly lower. Distance and minutes
ratios therefore run below 1 even when the frames are complete, and a game
with many stoppages (NYK at CHI) looks short without being broken. Distance
PER LIVE MINUTE is not affected by that: the pace ratio

    pace = (dist_miles / live_minutes) / (off_dist / off_min)

compares like with like. Observed pace ratio over all 10 games' players with
5+ official minutes: median 1.01, kept games 0.987 to 1.040.

Output: output/trackv3_comparison.csv (one row per player-game with official
minutes, all 10 games, including the 2 excluded) and
output/trackv3_games.csv (one row per game).

Gates (kept games only; in_workload = true)
-------------------------------------------
Tolerances were read from the data on October 19, 2026 and widened to cover
the NBA's 0.01 mi rounding (about 1.4 percent of a 0.35 mi player-game at
the 5-minute floor) and ordinary left-right noise.
  1. Join: every workload row has an official row; every official row with
     minutes has a workload row.
  2. Pace: each player-game with 5+ official minutes has a pace ratio in
     [0.95, 1.07]. Observed 0.987 to 1.040; a 3 to 4 point margin each side.
  3. Minutes: live_minutes never exceeds off_min by more than 0.5 minutes
     (live play is a subset of court time; observed maximum excess 0.002).
  4. Game distance: each game's summed distance is at least 0.89 of the
     official sum (observed kept minimum 0.909 for NYK at CHI, whose
     shortfall is dead-ball time: its pace ratio is 1.017). 0.89 sits
     between the lowest kept game (0.909) and the highest excluded game
     (DET at IND, 0.874). That margin is two points on ten games, so it is a
     cross-check, not the discriminator: dead-ball time and missing clock
     both lower distance, and only the frame-coverage gate (09) separates
     them. Maximum 1.03 (observed 1.007).

Printed, not gated: the two excluded games (01.02.2016.HOU.at.SAS and
01.02.2016.DET.at.IND). Their distance ratios (0.788, 0.874) fall below the
floor and their pace ratios stay near 1.01, so the shortfall is missing
clock, not bad positions. That is the evidence behind their exclusion.
The run fails if an excluded game sits inside the gate-4 tolerance.

Run:  python python/10_trackv3_check.py          build and gate
      python python/10_trackv3_check.py --check  rerun, gate, compare with committed tables
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data" / "nba_trackv3_boxscore_2015-16_10games.csv"
OUT = ROOT / "output"

MIN_OFFICIAL_MIN = 5.0
PACE_LO, PACE_HI = 0.95, 1.07
LIVE_OVER_MAX = 0.5
GAME_DIST_LO, GAME_DIST_HI = 0.89, 1.03


def load_official() -> pl.DataFrame:
    return pl.read_csv(OFFICIAL, schema_overrides={"game_id": pl.String})


def load_workload() -> pl.DataFrame:
    return pl.read_csv(OUT / "player_workload.csv", schema_overrides={"game_id": pl.String})


def compare(workload: pl.DataFrame, official: pl.DataFrame) -> pl.DataFrame:
    """Player-games on both sides (official minutes > 0), with the ratios."""
    return (workload.join(official, on=["game_id", "player_id"], how="inner")
            .filter(pl.col("off_min") > 0)
            .with_columns(dist_ratio=pl.col("dist_miles") / pl.col("off_dist"),
                          minute_ratio=pl.col("live_minutes") / pl.col("off_min"),
                          minute_diff=pl.col("live_minutes") - pl.col("off_min"))
            .with_columns(pace_ratio=pl.col("dist_ratio") / pl.col("minute_ratio"))
            .select("game", "game_id", "player_id", "player", "in_workload",
                    "live_minutes", "off_min", "minute_diff", "minute_ratio",
                    "dist_miles", "off_dist", "dist_ratio", "pace_ratio")
            .sort(["game", "player_id"]))


def game_summary(cmp: pl.DataFrame) -> pl.DataFrame:
    """One row per game: summed distance and minutes, ratios, median pace."""
    full = cmp.filter(pl.col("off_min") >= MIN_OFFICIAL_MIN)
    pace = full.group_by("game").agg(pace_median=pl.col("pace_ratio").median())
    return (cmp.group_by("game")
            .agg(in_workload=pl.col("in_workload").first(), players=pl.len(),
                 dist_miles=pl.col("dist_miles").sum(), off_dist=pl.col("off_dist").sum(),
                 live_minutes=pl.col("live_minutes").sum(), off_min=pl.col("off_min").sum())
            .with_columns(dist_ratio=pl.col("dist_miles") / pl.col("off_dist"),
                          minute_ratio=pl.col("live_minutes") / pl.col("off_min"))
            .join(pace, on="game")
            .sort("game"))


def gates(workload: pl.DataFrame, official: pl.DataFrame, cmp: pl.DataFrame) -> dict[str, bool]:
    """Named pass/fail for the kept games."""
    kept = cmp.filter(pl.col("in_workload"))
    full = kept.filter(pl.col("off_min") >= MIN_OFFICIAL_MIN)
    games = game_summary(cmp).filter(pl.col("in_workload"))
    unmatched_local = workload.join(official, on=["game_id", "player_id"], how="anti").height
    unmatched_off = (official.filter(pl.col("off_min") > 0)
                     .join(workload, on=["game_id", "player_id"], how="anti").height)
    return {
        "join: every workload row has an official row": unmatched_local == 0,
        "join: every official row with minutes has a workload row": unmatched_off == 0,
        f"pace ratio in [{PACE_LO}, {PACE_HI}] for all {full.height} kept player-games "
        f"with {MIN_OFFICIAL_MIN:g}+ official minutes":
            bool(full["pace_ratio"].is_between(PACE_LO, PACE_HI).all()),
        f"live minutes within {LIVE_OVER_MAX} of official at most":
            bool((kept["minute_diff"] <= LIVE_OVER_MAX).all()),
        f"game distance ratio in [{GAME_DIST_LO}, {GAME_DIST_HI}] for all "
        f"{games.height} kept games":
            bool(games["dist_ratio"].is_between(GAME_DIST_LO, GAME_DIST_HI).all()),
    }


def excluded_evidence(games: pl.DataFrame) -> list[str]:
    """Printed lines for the games flagged out of the workload."""
    return [f"EXCLUDED {r['game']}: distance {r['dist_ratio']:.3f} of official "
            f"({1 - r['dist_ratio']:.1%} short), live minutes {r['minute_ratio']:.3f} "
            f"of official, median pace {r['pace_median']:.3f}; floor {GAME_DIST_LO}"
            for r in games.filter(~pl.col("in_workload")).iter_rows(named=True)]


def excluded_consistent(games: pl.DataFrame) -> bool:
    ok = (games.filter(~pl.col("in_workload"))["dist_ratio"] < GAME_DIST_LO).all()
    return bool(ok)


def main(argv: list[str]) -> int:
    workload, official = load_workload(), load_official()
    cmp = compare(workload, official)
    games = game_summary(cmp)
    ok = True
    for name, passed in gates(workload, official, cmp).items():
        ok &= passed
        print(("PASS   " if passed else "FAIL   ") + name)
    for line in excluded_evidence(games):
        print(line)
    if not excluded_consistent(games):
        ok = False
        print("FAIL   an excluded game sits inside the distance tolerance")

    if "--check" in argv:
        for name, table in (("trackv3_comparison.csv", cmp), ("trackv3_games.csv", games)):
            old = pl.read_csv(OUT / name, schema_overrides={"game_id": pl.String}
                              if "game_id" in table.columns else None)
            if not table.equals(old.cast(table.schema)):
                ok = False
                print(f"committed output/{name} differs from a rerun")
    else:
        cmp.write_csv(OUT / "trackv3_comparison.csv")
        games.write_csv(OUT / "trackv3_games.csv")
    print("TRACKV3 CHECK " + ("PASSED" if ok else "FAILED")
          + f" ({cmp.height} player-games, {games.height} games)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
