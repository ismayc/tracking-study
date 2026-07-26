"""Modern tracking coverage: league movement from published aggregates,
2013-14 through 2025-26.

The raw-frames half of this study is frozen in 2015-16 because the league
has released no raw tracking since. The aggregate half is not frozen:
`LeagueDashPtStats(pt_measure_type="SpeedDistance")` publishes
vendor-computed distance and speed for every season of the tracking era,
SportVU through Hawk-Eye. This script harvests all thirteen seasons at
player and team level, asks one longitudinal question (how much does the
modern game actually run, and has that changed?), and gates itself the
family way: league totals from the player table must reconcile with
league totals from the independently queried team table.

Run:  python python/07_modern_aggregates.py           harvest + analyze
      python python/07_modern_aggregates.py --check   offline gate replay
                                                      from cached parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "modern_speeddistance"
OUT = ROOT / "output"
FIG = ROOT / "figures"

# ../basketball-analysis-tools (sibling checkout) carries the shared
# stats.nba.com User-Agent fix.
sys.path.insert(0, str(ROOT.parent / "basketball-analysis-tools"))

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, 2026)]
RECONCILE_TOL = 2e-3


def harvest() -> None:
    import nba_api_compat
    nba_api_compat.fix_user_agent()
    from nba_api.stats.endpoints import leaguedashptstats

    CACHE.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        for side in ["Player", "Team"]:
            dest = CACHE / f"{side.lower()}_{season}.parquet"
            if dest.exists():
                continue
            frame = leaguedashptstats.LeagueDashPtStats(
                pt_measure_type="SpeedDistance",
                per_mode_simple="Totals",
                player_or_team=side,
                season=season,
                season_type_all_star="Regular Season",
                timeout=60,
            ).get_data_frames()[0]
            pl.from_pandas(frame).write_parquet(dest)
            print(f"harvested {side} {season}: {len(frame)} rows")
            time.sleep(1.0)


def load(side: str, season: str) -> pl.DataFrame:
    return (pl.read_parquet(CACHE / f"{side}_{season}.parquet")
            .with_columns(pl.col("GP", "MIN", "DIST_MILES", "DIST_MILES_OFF",
                                 "DIST_MILES_DEF", "AVG_SPEED").cast(pl.Float64)))


def analyze() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, checks = [], []
    for season in SEASONS:
        players = load("player", season)
        teams = load("team", season)
        p_total = float(players["DIST_MILES"].sum())
        t_total = float(teams["DIST_MILES"].sum())
        rel = abs(p_total - t_total) / t_total
        checks.append({
            "check": f"{season}: player-table league miles reconcile with "
                     "independently queried team table (relative)",
            "value": rel, "threshold": RECONCILE_TOL,
            "passed": rel <= RECONCILE_TOL})
        team_games = float(teams["GP"].sum())
        w = players["MIN"]
        rows.append({
            "season": season,
            "n_players": players.height,
            "team_games": int(team_games),
            "miles_per_team_game": t_total / team_games,
            "off_miles_per_team_game":
                float(teams["DIST_MILES_OFF"].sum()) / team_games,
            "def_miles_per_team_game":
                float(teams["DIST_MILES_DEF"].sum()) / team_games,
            "avg_speed_mph_min_weighted":
                float((players["AVG_SPEED"] * w).sum() / w.sum()),
        })
    return pl.DataFrame(rows), pl.DataFrame(checks)


def main(argv: list[str]) -> int:
    if "--check" not in argv:
        harvest()
    trend, checks = analyze()

    if "--check" in argv:
        committed = pl.read_csv(OUT / "modern_movement_trend.csv")
        drift = (trend["miles_per_team_game"]
                 - committed["miles_per_team_game"]).abs().max()
        ok = bool(checks["passed"].all()) and float(drift) < 1e-9
        print("MODERN AGGREGATES CHECK "
              + ("PASSED" if ok else "FAILED")
              + f" ({checks.height} season reconciliations; committed trend "
                f"reproduces, max drift {float(drift):.2e})")
        return 0 if ok else 1

    trend.write_csv(OUT / "modern_movement_trend.csv")
    checks.write_csv(OUT / "modern_validation.csv")
    with pl.Config(tbl_rows=-1):
        print(trend)
        print(checks)

    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=trend["season"], y=trend["miles_per_team_game"],
        mode="lines+markers",
        customdata=trend.select("avg_speed_mph_min_weighted",
                                "off_miles_per_team_game",
                                "def_miles_per_team_game").to_numpy(),
        hovertemplate=("%{x}<br>%{y:.2f} miles per team-game"
                       "<br>avg speed %{customdata[0]:.2f} mph"
                       "<br>offense %{customdata[1]:.2f} · defense "
                       "%{customdata[2]:.2f}<extra></extra>")))
    fig.update_layout(
        title="How far a team runs per game, every tracking-era season "
              "(2013-14 through 2025-26, official published aggregates)",
        xaxis_title="season",
        yaxis_title="team miles per game",
        template="plotly_white")
    fig.write_html(FIG / "fig4_modern_movement.html", include_plotlyjs="cdn")

    ok = bool(checks["passed"].all())
    print("MODERN AGGREGATES VALIDATION "
          + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
