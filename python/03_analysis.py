"""Spatiotemporal analysis of raw SportVU tracking data.

Two questions:

  A. Workload — how far and how fast do players actually move during live play?
     Validated externally by 04_validate.py against the NBA's own published
     SpeedDistance aggregates for 2015-16 (median ~2.0 miles per game for
     players with 20+ minutes).

  B. Spacing — how much floor does the offense occupy, and does occupying more of
     it coincide with better shots? Spacing is measured as the convex hull area of
     the five offensive players, the standard geometric definition.

Both are only answerable from raw x/y frames. No public aggregate endpoint exposes
the geometry of who was standing where.

Inputs : data/moments/*.parquet, data/rosters/*.parquet  (from 02_parse_moments.py)
Outputs: output/*.csv, figures/*.png + *.html

Run: python python/03_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
MOM_DIR = ROOT / "data" / "moments"
ROSTER_DIR = ROOT / "data" / "rosters"
OUT = ROOT / "output"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# --- Analysis constants, all documented in the README -----------------------
FRAME_DT = 0.04                # 25 Hz
MAX_DT = 0.2                   # gaps longer than this mean the clock stopped
MAX_SPEED_FTS = 25.0           # ~17 mph; above this is a tracking glitch, not a human
POSSESSION_MAX_DIST = 4.0      # ft from ball to nearest player to call it possession
POSSESSION_MAX_BALL_Z = 10.0   # ft; above this the ball is a shot or high pass
SPACING_SUBSAMPLE = 5          # use every 5th frame (5 Hz) for the spacing sweep


def base_layout(title: str, subtitle: str, x_title: str, y_title: str) -> go.Layout:
    return go.Layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:{INK2}'>"
                        f"{subtitle}</span>",
                   font=dict(size=17, color=INK), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  size=12, color=INK2),
        xaxis=dict(title=dict(text=x_title, font=dict(color=INK2, size=12)),
                   showgrid=True, gridcolor=GRID, gridwidth=0.5, zeroline=False,
                   linecolor=AXIS, tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(title=dict(text=y_title, font=dict(color=INK2, size=12)),
                   showgrid=True, gridcolor=GRID, gridwidth=0.5, zeroline=False,
                   linecolor=AXIS, tickfont=dict(color=MUTED, size=11)),
        showlegend=False, margin=dict(l=75, r=140, t=85, b=55),
        width=900, height=500,
    )


def save(fig: go.Figure, stem: str) -> None:
    fig.write_image(FIG / f"{stem}.png", scale=2)
    fig.write_html(FIG / f"{stem}.html", include_plotlyjs="cdn")


# ---------------------------------------------------------------- A. workload
def player_workload(mom: pl.DataFrame, roster: pl.DataFrame) -> pl.DataFrame:
    """Distance and speed per player, during live play only.

    dt is derived from the game clock, so frames where the clock is stopped
    (dt == 0) drop out automatically. That is the correct denominator: we want
    movement during live basketball, not while players wander during a timeout.
    """
    players = mom.filter(pl.col("entity") == "player").sort(
        ["player_id", "period", "game_clock"], descending=[False, False, True])

    stepped = players.with_columns(
        dt=(pl.col("game_clock").shift(1) - pl.col("game_clock")).over(["player_id", "period"]),
        dx=(pl.col("x") - pl.col("x").shift(1)).over(["player_id", "period"]),
        dy=(pl.col("y") - pl.col("y").shift(1)).over(["player_id", "period"]),
    ).drop_nulls(["dt", "dx", "dy"])

    stepped = stepped.filter(
        (pl.col("dt") > 0) & (pl.col("dt") <= MAX_DT)
    ).with_columns(
        dist=(pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt()
    ).with_columns(
        speed=pl.col("dist") / pl.col("dt")
    ).filter(pl.col("speed") <= MAX_SPEED_FTS)

    agg = (stepped.group_by("player_id")
           .agg(live_seconds=pl.col("dt").sum(),
                dist_ft=pl.col("dist").sum(),
                mean_speed=pl.col("speed").mean(),
                p95_speed=pl.col("speed").quantile(0.95),
                max_speed=pl.col("speed").max(),
                frames=pl.len())
           .with_columns(
               dist_miles=pl.col("dist_ft") / 5280,
               live_minutes=pl.col("live_seconds") / 60,
           ))
    return agg.join(roster, on="player_id", how="left").sort("dist_miles", descending=True)


# ----------------------------------------------------------------- B. spacing
def possession_frames(mom: pl.DataFrame) -> pl.DataFrame:
    """Label each frame with the team in possession.

    Heuristic: the offensive team is the team of the player nearest the ball,
    provided that player is within POSSESSION_MAX_DIST feet and the ball is below
    POSSESSION_MAX_BALL_Z feet. The height condition removes shots in flight and
    lob passes, when the nearest player is not the one controlling the ball.

    This is a heuristic, not ground truth; the README says so and Limitation 3
    covers what it gets wrong.
    """
    ball = (mom.filter(pl.col("entity") == "ball")
            .select("period", "game_clock", "x", "y", "z")
            .rename({"x": "bx", "y": "by", "z": "bz"})
            .unique(subset=["period", "game_clock"], keep="first"))

    players = mom.filter(pl.col("entity") == "player")

    joined = (players.join(ball, on=["period", "game_clock"], how="inner")
              .with_columns(
                  ball_dist=((pl.col("x") - pl.col("bx")) ** 2
                             + (pl.col("y") - pl.col("by")) ** 2).sqrt()))

    nearest = (joined.filter(pl.col("bz") <= POSSESSION_MAX_BALL_Z)
               .sort("ball_dist")
               .group_by(["period", "game_clock"])
               .first()
               .filter(pl.col("ball_dist") <= POSSESSION_MAX_DIST)
               .select("period", "game_clock",
                       pl.col("team_id").alias("off_team_id"),
                       pl.col("ball_dist").alias("handler_dist")))
    # Sort into game order before returning: group_by row order is otherwise
    # nondeterministic, and spacing_by_frame subsamples every Nth ROW — an
    # unsorted frame would make the subsample (and the spacing numbers) vary
    # from run to run.
    return nearest.sort(["period", "game_clock"], descending=[False, True])


def hull_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Convex hull area of five points, in square feet. Degenerate sets -> nan."""
    pts = np.column_stack([xs, ys])
    if len(pts) < 3:
        return float("nan")
    try:
        return float(ConvexHull(pts).volume)  # 'volume' is area in 2D
    except Exception:  # noqa: BLE001 - collinear points
        return float("nan")


def spacing_by_frame(mom: pl.DataFrame, poss: pl.DataFrame) -> pl.DataFrame:
    """Convex hull area of the offensive five, on a subsample of frames."""
    players = mom.filter(pl.col("entity") == "player")

    # Subsample for tractability: 5 Hz is far finer than spacing changes.
    keys = (poss.with_row_index("i")
            .filter(pl.col("i") % SPACING_SUBSAMPLE == 0)
            .drop("i"))

    off = (players.join(keys, on=["period", "game_clock"], how="inner")
           .filter(pl.col("team_id") == pl.col("off_team_id")))

    rows = []
    for (period, clock, team), grp in off.group_by(
            ["period", "game_clock", "team_id"], maintain_order=True):
        if grp.height != 5:
            continue  # substitution frames or a dropped track
        area = hull_area(grp["x"].to_numpy(), grp["y"].to_numpy())
        if np.isfinite(area):
            rows.append({"period": period, "game_clock": clock,
                         "team_id": team, "hull_area": area})
    return pl.DataFrame(rows)


def main() -> int:
    games = sorted(MOM_DIR.glob("*.parquet"))
    if not games:
        raise SystemExit(f"No parsed moments in {MOM_DIR}. Run 02_parse_moments.py first.")

    workloads, spacings = [], []
    for path in games:
        mom = pl.read_parquet(path)
        roster = pl.read_parquet(ROSTER_DIR / path.name)

        w = player_workload(mom, roster).with_columns(game=pl.lit(path.stem))
        workloads.append(w)

        poss = possession_frames(mom)
        s = spacing_by_frame(mom, poss).with_columns(game=pl.lit(path.stem))
        spacings.append(s)
        print(f"  {path.stem}: {mom.height:,} frames, "
              f"{w.height} players, {s.height:,} spacing samples", flush=True)

    workload = pl.concat(workloads)
    spacing = pl.concat(spacings)
    workload.write_csv(OUT / "player_workload.csv")
    spacing.write_csv(OUT / "spacing_frames.csv")

    # ---- external validation: starters should be near 2.5 miles per game ----
    starters = workload.filter(pl.col("live_minutes") >= 20)
    print(f"\nWorkload: {workload.height} player-games; "
          f"{starters.height} with 20+ live minutes")
    print(f"  median distance, 20+ live min : {starters['dist_miles'].median():.2f} miles")
    print(f"  median live minutes           : {starters['live_minutes'].median():.1f}")
    print(f"  median mean speed             : {starters['mean_speed'].median():.2f} ft/s")
    # max_speed is retained in the CSV for diagnostics but is NOT a finding: for most
    # players it sits against the MAX_SPEED_FTS glitch filter, so it measures the
    # filter rather than the athlete. 04_validate.py quantifies this. Use p95.
    print(f"  median 95th-pct speed         : {starters['p95_speed'].median():.2f} ft/s "
          f"({starters['p95_speed'].median() * 0.681818:.1f} mph)")

    print(f"\nSpacing: {spacing.height:,} offensive frames")
    print(f"  median hull area : {spacing['hull_area'].median():.0f} sq ft")
    print(f"  IQR              : {spacing['hull_area'].quantile(0.25):.0f}"
          f" - {spacing['hull_area'].quantile(0.75):.0f} sq ft")

    # ---- figures ----
    top = workload.sort("dist_miles", descending=True).head(12)
    fig = go.Figure(layout=base_layout(
        "Distance covered during live play",
        f"Top 12 player-games across {len(games)} games, 2015-16 SportVU",
        "Miles covered", ""))
    fig.update_layout(height=500, margin=dict(l=190, r=90, t=85, b=55))
    fig.add_trace(go.Bar(
        x=top["dist_miles"].to_list(),
        y=[f"{p} ({g.split('.')[-3]})" for p, g in zip(top["player"], top["game"])],
        orientation="h", marker=dict(color=BLUE, line=dict(width=0)),
        hovertemplate="%{y}<br>%{x:.2f} miles<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    save(fig, "fig1_distance_covered")

    fig = go.Figure(layout=base_layout(
        "Offensive spacing is tightly concentrated",
        "Convex hull area of the five offensive players, 5 Hz samples",
        "Hull area (square feet)", "Share of frames"))
    fig.add_trace(go.Histogram(
        marker_line=dict(color='white', width=1),
        x=spacing["hull_area"].to_list(), nbinsx=70, histnorm="probability",
        marker=dict(color=BLUE, line=dict(width=0)),
        hovertemplate="%{x:.0f} sq ft: %{y:.1%}<extra></extra>"))
    med = spacing["hull_area"].median()
    fig.add_vline(x=med, line=dict(color=ORANGE, width=1.5, dash="dot"))
    fig.add_annotation(x=med, y=1, yref="paper", yanchor="bottom", xanchor="left",
                       xshift=6, text=f"median {med:.0f} sq ft",
                       showarrow=False, font=dict(color=ORANGE, size=12))
    save(fig, "fig2_spacing_distribution")

    print(f"\nWrote {OUT} and {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
