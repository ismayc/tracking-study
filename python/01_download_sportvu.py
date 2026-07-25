"""Download raw SportVU player-tracking logs for the 2015-16 NBA season.

Source and provenance
---------------------
The NBA installed SportVU optical tracking (six cameras per arena, 25 frames per
second) in every arena from 2013-14. For part of 2015-16 the raw game logs were
served publicly from stats.nba.com; the league stopped publishing them and later
switched vendors to Second Spectrum. Before access closed, the logs were archived
on GitHub, and those mirrors are what this script reads:

    https://github.com/linouk23/NBA-Player-Movements
        data/2016.NBA.Raw.SportVU.Game.Logs/   (636 games, ~6 MB each, 7z)

    Original upstream archive:
        https://github.com/neilmj/BasketballData

This is the only genuinely public source of *raw* NBA spatiotemporal tracking data.
Everything the league publishes today (LeagueDashPtStats and friends) is
pre-aggregated, so it cannot answer questions about geometry — spacing, separation,
who was where when.

Schema of each game's JSON
--------------------------
    gameid, gamedate
    events[]                       one entry per play-by-play event
        eventId
        visitor / home             team metadata + player roster
        moments[]                  ~25 per second
            [ period,
              utc_ms,
              game_clock_sec,      seconds remaining in period
              shot_clock_sec,      may be null
              None,
              positions[] ]        11 entries: ball first, then 10 players
                                   each [team_id, player_id, x, y, z]

Court coordinates are in feet on a 94 x 50 court: x in [0, 94] along the length,
y in [0, 50] across. For the ball, z is height in feet; for players z is 0.

Caveat carried into the analysis: `events` overlap heavily. Consecutive events
re-report the same moments, so frames MUST be de-duplicated on
(period, game_clock) before anything is counted. This is the single most common
error in public analyses of this dataset.

Usage:
    python 01_download_sportvu.py --games 12
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

import py7zr
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_sportvu"

REPO_API = ("https://api.github.com/repos/linouk23/NBA-Player-Movements/"
            "contents/data/2016.NBA.Raw.SportVU.Game.Logs")
TIMEOUT = 120


def list_available() -> list[dict]:
    r = requests.get(REPO_API, timeout=TIMEOUT)
    r.raise_for_status()
    return [f for f in r.json() if f["name"].endswith(".7z")]


def download_and_extract(entry: dict) -> Path | None:
    """Download one 7z game log and write the inner JSON to data/raw_sportvu/."""
    stem = entry["name"].replace(".7z", "")
    out = RAW_DIR / f"{stem}.json"
    if out.exists():
        return out

    try:
        resp = requests.get(entry["download_url"], timeout=TIMEOUT)
        resp.raise_for_status()
        # py7zr >= 1.0 dropped readall(); extract to a temp dir and move the JSON.
        with tempfile.TemporaryDirectory() as tmp:
            with py7zr.SevenZipFile(io.BytesIO(resp.content), mode="r") as z:
                z.extractall(path=tmp)
            for path in Path(tmp).rglob("*.json"):
                shutil.move(str(path), out)
                return out
        print(f"  no JSON inside {entry['name']}", flush=True)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED {entry['name']}: {type(exc).__name__} {str(exc)[:120]}", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12,
                    help="how many games to pull (each ~6 MB compressed)")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = list_available()
    print(f"{len(files)} game logs available in the archive", flush=True)

    # Deterministic selection: alphabetical, which is date-then-matchup ordered.
    # Not random, so the sample is reproducible without carrying a seed.
    chosen = files[:args.games]

    got = []
    for i, entry in enumerate(chosen, 1):
        path = download_and_extract(entry)
        if path:
            got.append(path)
            print(f"  [{i}/{len(chosen)}] {path.name} "
                  f"({path.stat().st_size / 1e6:.0f} MB JSON)", flush=True)
        time.sleep(0.4)

    if not got:
        print("No games downloaded.", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in got) / 1e6
    print(f"\n{len(got)} games -> {RAW_DIR}  ({total:.0f} MB uncompressed JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
