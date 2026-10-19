"""Per-game frame coverage: does each game's tracking cover its whole clock?

Why
---
04_validate.py compares the 10-game median workload with the league's
full-season median, so one broken game cannot move it. A per-player-game
comparison against the NBA's own tracking box score (BoxScorePlayerTrackV3,
2026-10-19 audit) found two games whose players' distances ran 13-21% short,
and the cause is visible in the frames themselves: those games are missing
stretches of game clock. Coverage tracks the distance shortfall closely
(HOU at SAS: 0.781 coverage, 0.788 distance ratio; DET at IND: 0.867 and
0.874), so it is a direct, offline measure of how much of each game's
workload the frames can capture.

Measure
-------
SportVU samples at 25 Hz. A period of L seconds of running clock should
hold about 25 * L distinct game_clock values (720 s regulation, 300 s
overtime); stopped-clock frames repeat a clock value and do not count.
For each game and period:
  coverage   distinct clock values / (25 * L)
  start gap  L minus the first (largest) clock value in the period

Gates, per game:
  * whole-game coverage >= 0.85
  * every period's coverage >= 0.75
  * every period starts within 30 s of its full clock

A game that fails is a FAIL unless it is listed in WORKLOAD_EXCLUDED with
a reason, in which case it prints as an EXCLUDED line on every run (never a
silent pass). 03_analysis.py reads the same dict and flags those games
in_workload = false in output/player_workload.csv: they stay in the frame
counts, spacing, and the possession join (missing seconds do not bias
those), but every workload statistic leaves them out. --check also fails
if the committed workload table's flags disagree with the dict.

Output: output/frame_coverage.csv (one row per game and period)
Run:    python python/09_frame_coverage.py          build
        python python/09_frame_coverage.py --check  rerun and compare with the committed table
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
MOMENTS = ROOT / "data" / "moments"
OUT = ROOT / "output"

HZ = 25
GAME_MIN = 0.85
PERIOD_MIN = 0.75
START_GAP_MAX = 30.0

# Games allowed to fail, each with the reason. Decided 2026-10-19 (option A):
# both are dropped from the workload analysis only. Their players' distances
# ran 13-21% short of the NBA's BoxScorePlayerTrackV3 for the same games.
WORKLOAD_EXCLUDED: dict[str, str] = {
    "01.02.2016.DET.at.IND": "33.5 s missing at a period start; workloads understated",
    "01.02.2016.HOU.at.SAS": "coverage 0.781, a period at 0.699; workloads understated",
}


def period_length(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def coverage(frames: pl.DataFrame) -> pl.DataFrame:
    """Per-period coverage for one game's frames (columns period, game_clock)."""
    return (frames.group_by("period")
            .agg(distinct=pl.col("game_clock").n_unique(),
                 first_clock=pl.col("game_clock").max())
            .with_columns(length=pl.col("period").map_elements(period_length,
                                                               return_dtype=pl.Float64))
            .with_columns(expected=pl.col("length") * HZ,
                          start_gap=pl.col("length") - pl.col("first_clock"))
            .with_columns(coverage=pl.col("distinct") / pl.col("expected"))
            .sort("period"))


def judge(game: str, per: pl.DataFrame) -> dict:
    game_cov = float(per["distinct"].sum() / per["expected"].sum())
    gates = {
        f"game coverage {game_cov:.3f} >= {GAME_MIN}": game_cov >= GAME_MIN,
        f"min period coverage {per['coverage'].min():.3f} >= {PERIOD_MIN}":
            float(per["coverage"].min()) >= PERIOD_MIN,
        f"max start gap {per['start_gap'].max():.1f} s <= {START_GAP_MAX:.0f} s":
            float(per["start_gap"].max()) <= START_GAP_MAX,
    }
    return {"game": game, "game_coverage": game_cov, "gates": gates,
            "passed": all(gates.values())}


def build() -> tuple[pl.DataFrame, list[dict]]:
    files = sorted(MOMENTS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no tracking frames in {MOMENTS}; run 01 and 02 first")
    tables, verdicts = [], []
    for f in files:
        per = coverage(pl.read_parquet(f, columns=["period", "game_clock"]))
        verdicts.append(judge(f.stem, per))
        tables.append(per.with_columns(game=pl.lit(f.stem)))
    table = (pl.concat(tables)
             .select("game", "period", "length", "expected", "distinct", "coverage",
                     "first_clock", "start_gap")
             .sort(["game", "period"]))
    return table, verdicts


def main(argv: list[str]) -> int:
    table, verdicts = build()
    ok = True
    for v in verdicts:
        failed = [g for g, passed in v["gates"].items() if not passed]
        if v["passed"]:
            print(f"PASS   {v['game']}: coverage {v['game_coverage']:.3f}")
        elif v["game"] in WORKLOAD_EXCLUDED:
            print(f"EXCLUDED {v['game']}: {'; '.join(failed)} "
                  f"(out of workload: {WORKLOAD_EXCLUDED[v['game']]})")
        else:
            ok = False
            print(f"FAIL   {v['game']}: {'; '.join(failed)}")
    stale = set(WORKLOAD_EXCLUDED) - {v["game"] for v in verdicts if not v["passed"]}
    if stale:
        ok = False
        print(f"FAIL   WORKLOAD_EXCLUDED lists games that no longer fail: {sorted(stale)}")

    if "--check" in argv:
        same = table.equals(pl.read_csv(OUT / "frame_coverage.csv").cast(table.schema))
        if not same:
            ok = False
            print("committed output/frame_coverage.csv differs from a rerun")
        flagged = set(pl.read_csv(OUT / "player_workload.csv")
                      .filter(~pl.col("in_workload"))["game"].unique())
        if flagged != set(WORKLOAD_EXCLUDED):
            ok = False
            print(f"player_workload.csv excludes {sorted(flagged)}, WORKLOAD_EXCLUDED "
                  f"lists {sorted(WORKLOAD_EXCLUDED)}; rerun 03_analysis.py")
    else:
        table.write_csv(OUT / "frame_coverage.csv")

    n_short = sum(1 for v in verdicts if not v["passed"])
    print("FRAME COVERAGE CHECK " + ("PASSED" if ok else "FAILED")
          + f" ({len(verdicts)} games; {n_short} short, "
            f"{len([v for v in verdicts if not v['passed'] and v['game'] in WORKLOAD_EXCLUDED])} "
            f"of them excluded from workload by decision)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
