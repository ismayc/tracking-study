"""Flatten raw SportVU game JSON into tidy per-frame parquet.

Input : data/raw_sportvu/*.json      (~100 MB each)
Output: data/moments/<game>.parquet  (~15 MB each)
        data/rosters/<game>.parquet  player metadata for the game

One output row is one entity in one frame:

    game_id, period, game_clock, shot_clock, entity ('ball'|'player'),
    team_id, player_id, x, y, z

THE DE-DUPLICATION STEP IS THE WHOLE POINT OF THIS FILE.

SportVU logs are organized by play-by-play event, and consecutive events re-report
overlapping windows of frames — the same instant appears in several events. Counting
raw frames therefore inflates every distance, every duration, and every average by
roughly 2-4x. Public analyses of this dataset get this wrong routinely.

Frames are keyed on (period, game_clock, player_id) and de-duplicated on that key.
The script reports the inflation factor it removed so the effect is visible rather
than silent.

Usage:
    python 02_parse_moments.py
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_sportvu"
MOM_DIR = ROOT / "data" / "moments"
ROSTER_DIR = ROOT / "data" / "rosters"

BALL_ID = -1


def parse_game(path: Path) -> tuple[int, int]:
    """Return (raw_rows, deduped_rows) for one game."""
    game = json.loads(path.read_text())
    game_id = game["gameid"]

    out_mom = MOM_DIR / f"{path.stem}.parquet"
    out_ros = ROSTER_DIR / f"{path.stem}.parquet"

    # ---- roster ----
    roster_rows = []
    if game["events"]:
        ev = game["events"][0]
        for side in ("home", "visitor"):
            for p in ev[side]["players"]:
                roster_rows.append({
                    "game_id": game_id,
                    "side": side,
                    "team_id": ev[side]["teamid"],
                    "team_abbrev": ev[side]["abbreviation"],
                    "player_id": p["playerid"],
                    "player": f"{p['firstname']} {p['lastname']}",
                    "jersey": p["jersey"],
                    "position": p["position"],
                })
    pl.DataFrame(roster_rows).write_parquet(out_ros)

    # ---- moments ----
    periods, clocks, shots = [], [], []
    team_ids, player_ids = [], []
    xs, ys, zs = [], [], []

    for ev in game["events"]:
        for m in ev.get("moments") or []:
            period, _utc, gc, sc = m[0], m[1], m[2], m[3]
            if gc is None:
                continue
            for team_id, player_id, x, y, z in m[5]:
                periods.append(period)
                clocks.append(gc)
                shots.append(sc)
                team_ids.append(team_id)
                player_ids.append(player_id)
                xs.append(x)
                ys.append(y)
                zs.append(z)

    df = pl.DataFrame({
        "period": periods, "game_clock": clocks, "shot_clock": shots,
        "team_id": team_ids, "player_id": player_ids,
        "x": xs, "y": ys, "z": zs,
    }, schema_overrides={"shot_clock": pl.Float64})

    raw_n = df.height
    df = (df
          .unique(subset=["period", "game_clock", "player_id"], keep="first")
          .with_columns(
              game_id=pl.lit(game_id),
              entity=pl.when(pl.col("player_id") == BALL_ID)
                       .then(pl.lit("ball")).otherwise(pl.lit("player")),
          )
          .sort(["period", "game_clock"], descending=[False, True]))
    df.write_parquet(out_mom)
    return raw_n, df.height


def main() -> int:
    MOM_DIR.mkdir(parents=True, exist_ok=True)
    ROSTER_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No SportVU JSON in {RAW_DIR}. Run 01_download_sportvu.py first.")

    tot_raw = tot_dedup = 0
    for path in files:
        if (MOM_DIR / f"{path.stem}.parquet").exists():
            existing = pl.read_parquet(MOM_DIR / f"{path.stem}.parquet").height
            print(f"  {path.stem}: cached ({existing:,} frames)", flush=True)
            tot_dedup += existing
            continue
        raw_n, dedup_n = parse_game(path)
        tot_raw += raw_n
        tot_dedup += dedup_n
        print(f"  {path.stem}: {raw_n:,} raw -> {dedup_n:,} unique "
              f"({raw_n / max(dedup_n, 1):.2f}x duplication removed)", flush=True)

    print(f"\n{len(files)} games -> {MOM_DIR}")
    if tot_raw:
        print(f"Overall duplication factor removed: {tot_raw / tot_dedup:.2f}x")
        print("Counting raw event frames would have inflated every distance and "
              "duration by that factor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
