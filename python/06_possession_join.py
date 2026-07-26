"""Join tracking frames to play-by-play: validate the possession heuristic and
measure spacing at the moment of the shot.

Two things the study's own README named as its next steps (limitations 3 & 5):

1. VALIDATE the nearest-player possession heuristic against ground truth.
   Play-by-play does not label possession continuously, but it does at
   discrete moments: at every field-goal attempt, the shooting team had the
   ball in the seconds before the shot; at every turnover, the team charged
   with it had the ball. Agreement between the heuristic's label just before
   those events and the event's team is a direct accuracy estimate.

2. SPACING -> SHOT QUALITY. Join each shot to the offense's convex-hull area
   shortly before the shot goes up, and compare eFG% across spacing quartiles.
   This is the question spacing exists to answer, and it needs the join.

Timing convention - the part that turned out to matter: play-by-play event
clocks lag the tracking clock by a per-game amount (scorer latency; up to
~5 s in the worst of the ten games). Sampling the heuristic at a fixed small
lead before the event clock therefore lands INSIDE the next possession for
high-latency games and agreement collapses - diagnosed by sweeping the lead
and watching agreement climb from 24% to 96% on the worst game. The fix is a
per-game CALIBRATION: choose the lead that maximizes agreement on shots, then
report agreement on TURNOVERS at that same lead as the held-out check (nothing
about turnovers was used to pick the lead). Pbp clocks have 1-second
resolution; tracking has 0.01 s.

Inputs : data/moments/*.parquet, data/pbp/*.parquet
Outputs: output/possession_validation.csv
         output/spacing_at_shot.csv, output/spacing_efg_quartiles.csv
         figures/fig3_spacing_vs_efg.{png,html}

Run: python python/06_possession_join.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
MOM_DIR = ROOT / "data" / "moments"
PBP_DIR = ROOT / "data" / "pbp"
OUT = ROOT / "output"
FIG = ROOT / "figures"

# Reuse the study's own possession heuristic and figure styling so this file
# cannot drift from 03_analysis.py.
_spec = importlib.util.spec_from_file_location(
    "tracking_analysis", Path(__file__).with_name("03_analysis.py"))
_a3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a3)

BLUE, ORANGE = _a3.BLUE, _a3.ORANGE
MUTED, AXIS = _a3.MUTED, _a3.AXIS

LEAD_GRID = [x * 0.5 for x in range(17)]   # 0..8 s, calibration grid
LOOKBACK_S = 2.0    # label window length
MIN_FRAMES = 5      # need at least this many labelled frames in the window


def parse_clock(col: str) -> pl.Expr:
    mins = pl.col(col).str.extract(r"PT(\d+)M", 1).cast(pl.Float64)
    secs = pl.col(col).str.extract(r"M([\d.]+)S", 1).cast(pl.Float64)
    return (mins * 60 + secs).alias("secs_left")


def load_pbp(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return (df.with_columns(
        parse_clock("clock"),
        period=pl.col("period").cast(pl.Int32),
        team_id=pl.col("teamId").cast(pl.Int64),
        is_fga=(pl.col("isFieldGoal") == "1"),
        made=(pl.col("shotResult") == "Made"),
        shot_value=pl.col("shotValue").cast(pl.Float64, strict=False),
        is_turnover=(pl.col("actionType") == "Turnover"),
    ).filter(pl.col("secs_left").is_not_null()))


def window_label(poss: pl.DataFrame, period: int, clock: float,
                 lead: float) -> int | None:
    """Majority possession label in [clock+lead, clock+lead+LOOKBACK_S]."""
    lo, hi = clock + lead, clock + lead + LOOKBACK_S
    win = poss.filter((pl.col("period") == period)
                      & (pl.col("game_clock") >= lo)
                      & (pl.col("game_clock") <= hi))
    if win.height < MIN_FRAMES:
        return None
    counts = win.group_by("off_team_id").agg(n=pl.len()).sort("n", descending=True)
    return int(counts["off_team_id"][0])


def shot_agreement(poss: pl.DataFrame, pbp: pl.DataFrame, lead: float) -> tuple[int, int]:
    n = agree = 0
    for ev in pbp.filter(pl.col("is_fga")).iter_rows(named=True):
        if ev["team_id"] == 0:
            continue
        label = window_label(poss, ev["period"], ev["secs_left"], lead)
        if label is None:
            continue
        n += 1
        agree += label == ev["team_id"]
    return agree, n


def calibrate_lead(poss: pl.DataFrame, pbp: pl.DataFrame) -> tuple[float, float]:
    """Per-game scorer-latency calibration: the lead maximizing shot agreement."""
    best_lead, best_rate = LEAD_GRID[0], -1.0
    for lead in LEAD_GRID:
        agree, n = shot_agreement(poss, pbp, lead)
        rate = agree / n if n else 0.0
        if rate > best_rate:
            best_lead, best_rate = lead, rate
    return best_lead, best_rate


def hulls_by_clock(mom: pl.DataFrame, poss: pl.DataFrame) -> pl.DataFrame:
    """Offensive hull area for every labelled frame (no subsample - we need
    specific instants, not a survey)."""
    players = mom.filter(pl.col("entity") == "player")
    off = (players.join(poss, on=["period", "game_clock"], how="inner")
           .filter(pl.col("team_id") == pl.col("off_team_id")))
    rows = []
    for (period, clock, team), grp in off.group_by(
            ["period", "game_clock", "team_id"], maintain_order=True):
        if grp.height != 5:
            continue
        area = _a3.hull_area(grp["x"].to_numpy(), grp["y"].to_numpy())
        if np.isfinite(area):
            rows.append({"period": period, "game_clock": clock,
                         "team_id": team, "hull_area": area})
    return pl.DataFrame(rows)


def main() -> int:
    games = sorted(MOM_DIR.glob("*.parquet"))
    val_rows, shot_rows = [], []

    for path in games:
        mom = pl.read_parquet(path)
        pbp = load_pbp(PBP_DIR / path.name)
        poss = _a3.possession_frames(mom)
        hulls = hulls_by_clock(mom, poss)

        # Calibrate the per-game scorer latency on shots...
        lead, fga_rate = calibrate_lead(poss, pbp)
        fga_agree, fga_checked = shot_agreement(poss, pbp, lead)
        # ...and validate on turnovers, which the calibration never saw.
        to_checked = to_agree = 0
        for ev in pbp.filter(pl.col("is_turnover")).iter_rows(named=True):
            if ev["team_id"] == 0:
                continue
            label = window_label(poss, ev["period"], ev["secs_left"], lead)
            if label is None:
                continue
            to_checked += 1
            to_agree += label == ev["team_id"]

        # spacing at the shot, sampled at the calibrated lead
        for ev in pbp.filter(pl.col("is_fga")).iter_rows(named=True):
            if ev["team_id"] == 0:
                continue
            lo = ev["secs_left"] + lead
            hi = lo + LOOKBACK_S
            h = hulls.filter((pl.col("period") == ev["period"])
                             & (pl.col("game_clock") >= lo)
                             & (pl.col("game_clock") <= hi)
                             & (pl.col("team_id") == ev["team_id"]))
            if h.height >= MIN_FRAMES:
                shot_rows.append({
                    "game": path.stem, "period": ev["period"],
                    "secs_left": ev["secs_left"],
                    "hull_area": float(h["hull_area"].mean()),
                    "made": bool(ev["made"]),
                    "shot_value": float(ev["shot_value"] or 0),
                })

        val_rows.append({
            "game": path.stem, "calibrated_lead_s": lead,
            "fga_checked": fga_checked, "fga_agree": fga_agree,
            "fga_rate": fga_agree / fga_checked if fga_checked else None,
            "to_checked": to_checked, "to_agree": to_agree,
            "to_rate": to_agree / to_checked if to_checked else None,
        })
        print(f"  {path.stem}: latency {lead:.1f}s, "
              f"FGA {fga_agree}/{fga_checked} ({fga_rate:.1%}), "
              f"TO(held-out) {to_agree}/{to_checked} "
              f"({to_agree / max(to_checked, 1):.1%})", flush=True)

    validation = pl.DataFrame(val_rows)
    validation.write_csv(OUT / "possession_validation.csv")
    fga_total = validation["fga_checked"].sum()
    fga_rate = validation["fga_agree"].sum() / fga_total
    to_rate = validation["to_agree"].sum() / max(validation["to_checked"].sum(), 1)
    print(f"\nHeuristic vs play-by-play (per-game latency calibrated on shots):")
    print(f"  shots     {fga_rate:.1%} ({fga_total:,} checked; calibration metric)")
    print(f"  turnovers {to_rate:.1%} (held out from calibration)")

    shots = pl.DataFrame(shot_rows)
    shots.write_csv(OUT / "spacing_at_shot.csv")

    # eFG% by spacing quartile (quartiles over all joined shots)
    qs = [shots["hull_area"].quantile(q) for q in (0.25, 0.5, 0.75)]
    shots = shots.with_columns(
        quartile=pl.when(pl.col("hull_area") <= qs[0]).then(pl.lit("Q1 (tightest)"))
        .when(pl.col("hull_area") <= qs[1]).then(pl.lit("Q2"))
        .when(pl.col("hull_area") <= qs[2]).then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4 (widest)")))
    by_q = (shots.group_by("quartile")
            .agg(n=pl.len(),
                 mean_hull=pl.col("hull_area").mean(),
                 efg=(pl.col("made").sum()
                      + 0.5 * (pl.col("made") & (pl.col("shot_value") == 3)).sum())
                     / pl.len(),
                 share_3pt=(pl.col("shot_value") == 3).mean())
            .sort("mean_hull"))
    by_q.write_csv(OUT / "spacing_efg_quartiles.csv")
    print("\neFG% by spacing quartile (hull area in the ~2s before the shot):")
    for r in by_q.iter_rows(named=True):
        print(f"  {r['quartile']:14s} mean {r['mean_hull']:5.0f} sq ft: "
              f"eFG% {r['efg']:.3f}, 3PA share {r['share_3pt']:.1%} (n={r['n']:,})")

    fig = go.Figure(layout=_a3.base_layout(
        "Spacing right before the shot barely moves eFG%",
        f"eFG% by offensive spacing quartile in the ~2s before the shot "
        f"(latency-calibrated join) · {shots.height:,} shots, 10 games, 2015-16 · "
        f"the naive uncalibrated join shows a strong gradient that is an artifact",
        "Offensive convex-hull area before the shot (quartiles)", "Effective FG%"))
    fig.add_trace(go.Bar(
        x=[f"{r['quartile']}<br>{r['mean_hull']:.0f} sq ft"
           for r in by_q.iter_rows(named=True)],
        y=by_q["efg"].to_list(),
        marker=dict(color=BLUE, line=dict(width=0)),
        text=[f"{v:.1%}" for v in by_q["efg"]],
        textposition="outside", textfont=dict(color=MUTED, size=11),
        hovertemplate="%{x}: eFG %{y:.3f}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%",
                     range=[0, float(by_q["efg"].max()) * 1.25])
    _a3.save(fig, "fig3_spacing_vs_efg")

    print(f"\nWrote {OUT} and {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
