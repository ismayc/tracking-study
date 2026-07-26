"""Validate the raw-frame pipeline against the NBA's own published tracking aggregates.

The strongest check available for this pipeline is not a blog post or a rule of
thumb — it is the league's own numbers, computed by the vendor from the same
SportVU feed and published through `LeagueDashPtStats(pt_measure_type="SpeedDistance")`.

If independently flattening, de-duplicating, and differencing 25 Hz frames
reproduces the official per-game distance and average speed, then the chain
(parse -> de-duplicate -> clock-based dt -> distance) is behaving. If it does not,
something upstream is wrong and no downstream finding can be trusted.

The de-duplication step is what this check really tests. Skipping it inflates
distance by ~3x, which would show up here immediately.

Note on comparability: the official table covers all 82 games for every player,
while the local sample is 10 games from early January 2016. The comparison is
therefore between *distributions* (medians across players), not player-by-player.

Run: python python/04_validate.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from nba_api.stats.endpoints import leaguedashptstats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

SEASON = "2015-16"
MIN_MINUTES = 20
FT_PER_S_TO_MPH = 0.681818


def official_medians() -> tuple[dict, int]:
    """Medians of the published reference table (all the check uses), cached
    after the first successful fetch so the gate keeps working when
    stats.nba.com is unreachable (it rate-limits aggressively after heavy
    use). The reference is a frozen 2015-16 aggregate - the cache cannot go
    stale; delete data/official_speeddistance_medians.parquet to force a
    re-fetch."""
    cache = ROOT / "data" / "official_speeddistance_medians.parquet"
    if cache.exists():
        row = pl.read_parquet(cache).row(0, named=True)
        return row, int(row.pop("n_players"))
    df = leaguedashptstats.LeagueDashPtStats(
        season=SEASON, player_or_team="Player", pt_measure_type="SpeedDistance",
        per_mode_simple="PerGame", timeout=60,
    ).get_data_frames()[0]
    df = df[df["MIN"] >= MIN_MINUTES]
    med = {
        "dist_miles": float(df["DIST_MILES"].astype(float).median()),
        "minutes": float(df["MIN"].astype(float).median()),
        "avg_speed_mph": float(df["AVG_SPEED"].astype(float).median()),
    }
    pl.DataFrame([{**med, "n_players": len(df)}]).write_parquet(cache)
    return med, len(df)


def main() -> int:
    local = pl.read_csv(OUT / "player_workload.csv").filter(
        pl.col("live_minutes") >= MIN_MINUTES)
    off, off_n = official_medians()

    rows = [
        ("median distance per game (miles)",
         local["dist_miles"].median(), off["dist_miles"], 0.15),
        ("median minutes per game",
         local["live_minutes"].median(), off["minutes"], 2.0),
        ("median average speed (mph)",
         local["mean_speed"].median() * FT_PER_S_TO_MPH,
         off["avg_speed_mph"], 0.25),
    ]

    print(f"Raw-frame pipeline vs NBA published aggregates, {SEASON}")
    print(f"  local : {local.height} player-games from 10 games, {MIN_MINUTES}+ live min")
    src = f"{off_n} players" if off_n else "cached medians (see official_medians)"
    print(f"  official: {src}, full season, {MIN_MINUTES}+ MPG\n")
    print(f"{'metric':38s} {'mine':>8s} {'official':>9s} {'diff':>8s}  status")

    results, ok_all = [], True
    for name, mine, ref, tol in rows:
        diff = mine - ref
        ok = abs(diff) <= tol
        ok_all &= ok
        print(f"{name:38s} {mine:8.2f} {ref:9.2f} {diff:+8.2f}  {'PASS' if ok else 'FAIL'}")
        results.append({"metric": name, "local": mine, "official": ref,
                        "diff": diff, "tolerance": tol, "status": "PASS" if ok else "FAIL"})

    pl.DataFrame(results).write_csv(OUT / "validation_vs_official.csv")

    # Diagnostic: top speed is NOT usable. The 25 ft/s filter in 03_analysis.py is
    # a glitch guard, and for most players the observed maximum sits against it,
    # so "max speed" measures the threshold rather than the athlete.
    at_cap = local.filter(pl.col("max_speed") >= 24.5).height / local.height
    print(f"\nDiagnostic — {at_cap:.0%} of players' max speed sits within 0.5 ft/s "
          f"of the 25 ft/s glitch filter.")
    print("  Top speed is therefore NOT reported as a finding; the 95th percentile "
          f"({local['p95_speed'].median():.1f} ft/s = "
          f"{local['p95_speed'].median() * FT_PER_S_TO_MPH:.1f} mph) is stable and is used instead.")

    print("\n" + ("VALIDATION PASSED" if ok_all else "VALIDATION FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
