# Spatiotemporal tracking study — raw SportVU player movement

Study addressing the single largest gap in `../requirements.md`: requirement
#5, *"large and complex basketball datasets, including tracking/spatiotemporal
data."*

**What this uses:** raw optical tracking — x, y (and z for the ball) at **25 frames
per second** for all ten players and the ball. 10 games, **7.5 million de-duplicated
frames**.

---

## 1. Is public NBA tracking data actually available?

Short answer: **raw tracking, yes — but only a frozen 2015-16 archive. Nothing
current.**

| Source | Granularity | Public? | Used here |
|---|---|---|---|
| **SportVU raw game logs, 2015-16** | 25 Hz x/y/z, 11 entities | Yes, via GitHub mirrors | ✅ **primary** |
| `LeagueDashPtStats` (Second Spectrum) | Season aggregates per player | Yes, live | ✅ validation only |
| `PlayByPlayV3` `xLegacy`/`yLegacy` | Shot location per event | Yes, live | see `../playbyplay-study` |
| `ShotChartDetail` | Shot x/y + clock, ~102k shots/season | Yes, live | see `../playbyplay-study` |
| Second Spectrum raw feed (2017–present) | 25 Hz | **No** — teams/licensees only | ✗ |
| Hawk-Eye skeletal tracking (2023–present) | 29-point pose, 60 Hz | **No** — teams only | ✗ |

### Provenance of the raw data

The NBA installed SportVU optical tracking (six cameras per arena, 25 fps) league-wide
from 2013-14. During 2015-16 the raw game logs were briefly served from
`stats.nba.com`. The league then withdrew public access and later switched vendors to
Second Spectrum. Before access closed, the logs were archived on GitHub:

- **Primary mirror used:** [`linouk23/NBA-Player-Movements`](https://github.com/linouk23/NBA-Player-Movements) — `data/2016.NBA.Raw.SportVU.Game.Logs/`, **636 games**, ~6 MB each (7z)
- **Upstream archive:** [`neilmj/BasketballData`](https://github.com/neilmj/BasketballData)
- **Backup mirror:** [`sealneaward/nba-movement-data`](https://github.com/sealneaward/nba-movement-data) — explicitly created as a backup after the NBA closed public access

**This means the raw data is nine seasons stale.** Findings describe 2015-16
basketball, not today's. That is a hard limitation of what is public, not a design
choice — and it is exactly why teams' internal tracking feeds are valuable.

### Raw JSON schema

```
gameid, gamedate
events[]                    one per play-by-play event
  eventId
  home / visitor            team metadata + roster (name, playerid, jersey, position)
  moments[]                 ~25 per second
    [ period,
      utc_ms,
      game_clock_sec,       seconds remaining in period
      shot_clock_sec,       may be null
      None,
      positions[] ]         11 entries: ball first, then 10 players
                            each [team_id, player_id, x, y, z]
```

Court coordinates are feet on a 94 × 50 court. Ball rows carry `team_id = -1,
player_id = -1` and `z` = height in feet; player `z` is 0.

---

## 2. Pipeline

```
python/01_download_sportvu.py   GitHub mirror -> data/raw_sportvu/*.json   (~100 MB/game)
python/02_parse_moments.py      JSON          -> data/moments/*.parquet    (de-duplicated)
python/03_analysis.py           workload + spacing -> output/, figures/
python/04_validate.py           check against the NBA's own published aggregates
python/05_harvest_pbp.py        play-by-play for the same 10 games
python/06_possession_join.py    tracking <-> pbp join: heuristic validation + spacing-at-shot
```

```bash
pip install nba_api polars plotly kaleido py7zr requests scipy numpy pyarrow
python python/01_download_sportvu.py --games 10   # ~1 GB of JSON
python python/02_parse_moments.py
python python/03_analysis.py
python python/04_validate.py
```

### The de-duplication trap — the most important step in this repo

SportVU logs are organised **by play-by-play event**, and consecutive events
**re-report overlapping windows of frames**. The same instant in the game appears in
several events.

Measured across the 10 games: **3.07× duplication**.

Counting raw event frames would inflate every distance, duration, and average by
roughly a factor of three. Frames are keyed on `(period, game_clock, player_id)` and
de-duplicated on that key; `02_parse_moments.py` prints the factor it removed for
every game rather than doing it silently.

This is the single most common error in public analyses of this dataset, and
Section 4 shows how it was caught.

### Other documented decisions

| Decision | Value | Why |
|---|---|---|
| `dt` from the **game clock**, not wall clock | — | Frames where the clock is stopped get `dt = 0` and drop out. Distance is then movement during **live play**, which is the meaningful denominator |
| Max plausible speed | 25 ft/s | Above this is a tracking glitch, not a human. **See the caveat in Section 4** |
| Possession = team of nearest player to ball | ≤ 4 ft, ball below 10 ft | The height condition removes shots in flight and lobs, when the nearest player is not the one in control |
| Spacing subsample | every 5th frame (5 Hz) | Spacing does not change meaningfully at 25 Hz; this makes ~117k convex hulls tractable |

---

## 3. Findings

### A. Workload — 2.00 miles per player-game during live play

Across 210 player-games, for the 120 with 20+ live minutes:

| Metric | Median |
|---|---|
| Distance covered | **2.00 miles** |
| Live minutes | **27.6** |
| Average speed | **6.22 ft/s** (4.24 mph) |
| 95th-percentile speed | **14.8 ft/s** (10.1 mph) |

### B. Spacing — the offense occupies a median 529 sq ft

Convex hull area of the five offensive players, over 116,801 sampled frames:

| Statistic | Value |
|---|---|
| Median | **529 sq ft** |
| IQR | **370 – 708 sq ft** |

The distribution is strongly right-skewed. For scale, a half court is 47 × 50 =
2,350 sq ft, so the offense typically occupies only **about a fifth of the half
court**. The long right tail is transition, where the five stretch across far more
floor before the defense sets.

### C. The play-by-play join — and the clock-latency trap inside it

The join both README limitations asked for (3 and 5) is now done, and it
contained the most instructive artifact in the study.

**Validating the possession heuristic.** Play-by-play labels possession at
discrete moments: the shooting team had the ball before every shot, the
charged team before every turnover. Naively sampling the heuristic ~1s before
each event's clock produced **67% agreement — and 43% in one game, worse than
a coin flip.** The cause was not the heuristic: **play-by-play clocks lag the
tracking clock by a per-game scorer latency of 2.5–6.0 seconds**, so a fixed
small lead lands inside the *next* possession. Sweeping the lead made
agreement climb from 24% to 96% on the worst game. After calibrating the lead
per game **on shots only**, then scoring **turnovers as a held-out check**:

| Check | Agreement |
|---|---|
| Shots (calibration metric) | **97.5%** (1,623 events) |
| Turnovers (held out) | **93.5%** (232 events) |

The heuristic is good; the naive join was broken. Per-game latencies are in
`output/possession_validation.csv`.

**Spacing at the moment of the shot.** With calibrated windows, eFG% across
spacing quartiles is nearly flat (50.0% / 45.0% / 45.7% / 47.2% from tightest
to widest), while the three-point-attempt share rises monotonically
(23.5% → 29.8%): in this sample, spacing shapes the *shot profile* more than
raw shot efficiency. The instructive part: **the uncalibrated join produced a
strong, clean, monotone "wider = better shots" gradient (42% → 53%) that was
entirely an artifact of sampling spacing in the wrong window.** A junior
analyst ships that chart; it survives review because it confirms what
everyone already believes. `figures/fig3_spacing_vs_efg.png` shows the
calibrated version and says so in the subtitle.

---

## 4. Validation

### The check that matters: reproducing the league's own numbers

The strongest available check is not a rule of thumb — it is the NBA's own published
aggregates, computed by the vendor from the same SportVU feed and served through
`LeagueDashPtStats(pt_measure_type="SpeedDistance")` for 2015-16.

Independently flattening, de-duplicating, and differencing 25 Hz frames should
reproduce them. It does:

| Metric | This pipeline | NBA official | Difference |
|---|---|---|---|
| Median distance per game | **2.00 mi** | **2.00 mi** | −0.00 |
| Median minutes per game | **27.64** | **28.05** | −0.41 |
| Median average speed | **4.24 mph** | **4.23 mph** | +0.01 |

`04_validate.py` exits non-zero if any of these drift outside tolerance.

**This is also how the de-duplication bug would have been caught.** Skipping that
step inflates distance ~3×, which would have shown up here as 6 miles per game
against the league's 2.0 — impossible to miss, and impossible to detect without an
external reference.

### A statistic that did not survive: top speed

The first version of this analysis reported median top speed as 24.8 ft/s. That
number is meaningless.

**73% of players' observed maximum sits within 0.5 ft/s of the 25 ft/s glitch
filter.** The statistic was measuring the threshold, not the athlete — optical
tracking produces occasional single-frame position jumps, and a maximum over ~40,000
frames will find them every time.

Top speed is therefore **not reported as a finding**. The 95th percentile
(14.8 ft/s) is stable across players and games and is used instead. `max_speed`
stays in the output CSV, flagged as diagnostic only.

The general lesson: **an extreme-value statistic on noisy sensor data measures the
noise.** Ranking players by tracked top speed would have produced a leaderboard of
whoever got the worst camera frame.

---

## 5. Limitations

1. **The data is 2015-16.** Nine seasons stale, pre-dating the pace-and-space peak.
   Nothing here should be read as describing current NBA basketball. No public raw
   tracking exists for recent seasons.
2. **Ten games, early January 2016.** Chosen deterministically (alphabetically first
   in the archive) for reproducibility. Not a random sample of the season, so
   team-level numbers are not league-representative.
3. **Possession is a heuristic** — now validated: 97.5% agreement with the
   shooting team at shots and 93.5% on held-out turnovers after per-game
   clock-latency calibration (see Finding C). It still misassigns during loose
   balls, steals, and contested rebounds, and the latency calibration itself
   assumes the offset is constant within a game.
4. **Spacing is convex hull area only.** It treats a well-spaced five-out set and a
   set with one player stranded in a corner as similar if the hulls match. It says
   nothing about whether the spacing was *useful* — no defender positions, no
   shot-quality outcome.
5. **Shot-outcome join** — now done (Finding C): spacing-at-shot vs eFG%, with
   the clock-latency artifact documented. Ten games is still a small sample
   for the flat-gradient conclusion; it rules out a large effect in this data,
   not a small one.
6. **Listed positions come from the game log**, not from where players actually
   stood.

---

## 6. What this does and does not demonstrate

**Does:** handling genuinely large spatiotemporal data (7.5M frames), finding and
quantifying a 3× structural duplication trap, computational geometry per frame,
validating against an independent external source, discarding a statistic that
turned out to measure sensor noise, a validated cross-source join (tracking ↔
play-by-play) including per-game clock-latency calibration, and a
possession-heuristic accuracy estimate with a held-out check.

**Does not:** defender-distance shot quality, player tracking through occlusion,
or any work on a current-season feed (none is public).
