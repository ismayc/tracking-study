"""Fetch play-by-play for the 10 tracked games, for the possession join.

The tracking parquet files carry their NBA game_id, so the join target is
exact. One parquet per game, resumable.

Run:  python python/05_harvest_pbp.py
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl
from nba_api.stats.endpoints import playbyplayv3

ROOT = Path(__file__).resolve().parents[1]
MOM_DIR = ROOT / "data" / "moments"
PBP_DIR = ROOT / "data" / "pbp"
PBP_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    games = sorted(MOM_DIR.glob("*.parquet"))
    if not games:
        raise SystemExit("No parsed moments; run 02_parse_moments.py first.")

    for path in games:
        out = PBP_DIR / path.name
        if out.exists():
            print(f"  {path.stem}: cached", flush=True)
            continue
        game_id = pl.scan_parquet(path).select("game_id").first().collect()["game_id"][0]
        df = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60).get_data_frames()[0]
        if df.empty:
            raise SystemExit(f"Empty play-by-play for {path.stem} ({game_id})")
        pl.from_pandas(df.astype(str)).write_parquet(out)
        print(f"  {path.stem}: {len(df)} events ({game_id})", flush=True)
        time.sleep(1.0)

    print("Play-by-play harvest complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
