"""Write every analysis number in README.md from the committed output CSVs.

Same discipline as the rest of the family (see shot-quality-study's
04_findings.py): numbers in the prose are read from output/, never typed.
This study's README keeps hand-written prose OUTSIDE marker-delimited
regions; this script owns only the text between

    <!-- gen:NAME --> ... <!-- /gen:NAME -->

for NAME in: scale, findings, validation, limitations, modern.

Numbers that are properties of the source rather than computed results
(25 Hz, 2015-16, season labels, camera counts, thresholds like the 25 ft/s
glitch filter or the 2-second label window) stay literal prose. So do a few
one-off diagnostics from the calibration sweep (the naive-join 67%/43%/24%
agreement rates and the half-game robustness figures) that 06 prints but
does not commit to output/.

The generator writes PLAIN text. <abbr> hover annotations inside the
regions are owned by ../basketball-analysis-tools/glossary.py; run
`glossary.py --sync` after this script. `--check` strips those tags before
comparing, so the round trip is idempotent.

Run:   python python/08_findings.py           rewrite the README regions
       python python/08_findings.py --check   verify README matches output/
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
README = ROOT / "README.md"

sys.path.insert(0, str(ROOT.parent / "basketball-analysis-tools"))
try:
    from glossary import strip_abbr
except ImportError:  # tools repo not checked out next door: replicate
    _ABBR = re.compile(r'<abbr title="[^"]*">([^<]*)</abbr>')

    def strip_abbr(text: str) -> str:
        return _ABBR.sub(r"\1", text)

MPH_PER_FTS = 3600 / 5280
MINUS = "−"  # the README's tables use a true minus sign
MIN_LIVE_MINUTES = 20


def signed(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}".replace("-", MINUS)


# ---------------------------------------------------------------- sections


def scale_section() -> str:
    w = pl.read_csv(OUT / "player_workload.csv")
    n_games = w["game"].n_unique()
    # workload counts frames for the 10 player rows of each moment; the
    # ball is the 11th tracked entity in every de-duplicated frame
    total_m = w["frames"].sum() * 11 / 10 / 1e6
    return f"""\
**What this uses:** raw optical tracking: x, y (and z for the ball) at **25 frames
per second** for all ten players and the ball. {n_games} games, **{total_m:.1f} million de-duplicated
frames**."""


def possession_numbers() -> dict[str, str]:
    p = pl.read_csv(OUT / "possession_validation.csv")
    fga_rate = p["fga_agree"].sum() / p["fga_checked"].sum()
    to_rate = p["to_agree"].sum() / p["to_checked"].sum()
    return {
        "lead_range": f"{p['calibrated_lead_s'].min():.1f}–"
                      f"{p['calibrated_lead_s'].max():.1f}",
        "worst_cal": f"{p['fga_rate'].min():.0%}",
        "fga_rate": f"{fga_rate:.1%}",
        "fga_n": f"{p['fga_checked'].sum():,}",
        "to_rate": f"{to_rate:.1%}",
        "to_n": f"{p['to_checked'].sum():,}",
    }


def findings_section() -> str:
    w = pl.read_csv(OUT / "player_workload.csv")
    sub = w.filter(pl.col("live_minutes") >= MIN_LIVE_MINUTES)
    med_miles = f"{sub['dist_miles'].median():.2f}"
    med_min = f"{sub['live_minutes'].median():.1f}"
    med_fts = sub["mean_speed"].median()
    p95_fts = sub["p95_speed"].median()

    s = pl.read_csv(OUT / "spacing_frames.csv")
    med_hull = f"{s['hull_area'].median():.0f}"
    q25, q75 = s["hull_area"].quantile(0.25), s["hull_area"].quantile(0.75)

    pv = possession_numbers()

    q = pl.read_csv(OUT / "spacing_efg_quartiles.csv").sort("quartile")
    efg_list = " / ".join(f"{v:.1%}" for v in q["efg"])
    share_lo = f"{q['share_3pt'][0]:.1%}"
    share_hi = f"{q['share_3pt'][q.height - 1]:.1%}"

    return f"""\
## 3. Findings

### A. Workload: {med_miles} miles per player-game during live play

Across {w.height} player-games, for the {sub.height} with 20+ live minutes:

| Metric | Median |
|---|---|
| Distance covered | **{med_miles} miles** |
| Live minutes | **{med_min}** |
| Average speed | **{med_fts:.2f} ft/s** ({med_fts * MPH_PER_FTS:.2f} mph) |
| 95th-percentile speed | **{p95_fts:.1f} ft/s** ({p95_fts * MPH_PER_FTS:.1f} mph) |

### B. Spacing: the offense occupies a median {med_hull} sq ft

Convex hull area of the five offensive players, over {s.height:,} sampled frames:

| Statistic | Value |
|---|---|
| Median | **{med_hull} sq ft** |
| IQR | **{q25:.0f} – {q75:.0f} sq ft** |

The distribution is strongly right-skewed. For scale, a half court is 47 × 50 =
2,350 sq ft, so the offense typically occupies only **about a fifth of the half
court**. The long right tail is transition, where the five stretch across far more
floor before the defense sets.

### C. The play-by-play join, and the clock-latency trap inside it

The join both README limitations asked for (3 and 5) is now done, and it
contained the most instructive artifact in the study.

**Validating the possession heuristic.** Play-by-play labels possession at
discrete moments: the shooting team had the ball before every shot, the
charged team before every turnover. Naively sampling the heuristic ~1s before
each event's clock produced **67% agreement, and 43% in one game, worse than
a coin flip.** The cause was not the heuristic: **play-by-play clocks lag the
tracking clock by a per-game scorer latency of {pv["lead_range"]} seconds**, so a fixed
small lead lands inside the *next* possession. Sweeping the lead made
agreement climb from 24% to {pv["worst_cal"]} on the worst game. After calibrating the lead
per game **on shots only**, then scoring **turnovers as a held-out check**:

| Check | Agreement |
|---|---|
| Shots (calibration metric) | **{pv["fga_rate"]}** ({pv["fga_n"]} events) |
| Turnovers (held out) | **{pv["to_rate"]}** ({pv["to_n"]} events) |

The heuristic is good; the naive join was broken. Per-game latencies are in
`output/possession_validation.csv`.

Robustness of the constant-latency assumption: calibrating each half of each
game separately moves the chosen lead by up to 2.5 s in a few games, but
agreement at either half's optimum stays between 94% and 100%. The 2-second
label window makes the calibration tolerant to that much drift. The
assumption is approximate, and approximately harmless here.

**Spacing at the moment of the shot.** With calibrated windows, eFG% across
spacing quartiles is nearly flat ({efg_list} from tightest
to widest), while the three-point-attempt share rises monotonically
({share_lo} → {share_hi}): in this sample, spacing shapes the *shot profile* more than
raw shot efficiency. The instructive part: **the uncalibrated join produced a
strong, clean, monotone "wider = better shots" gradient (42% → 53%) that was
entirely an artifact of sampling spacing in the wrong window.** A junior
analyst ships that chart; it survives review because it confirms what
everyone already believes. `figures/fig3_spacing_vs_efg.png` shows the
calibrated version and says so in the subtitle."""


def validation_section() -> str:
    v = pl.read_csv(OUT / "validation_vs_official.csv")
    assert (v["status"] == "PASS").all(), "validation_vs_official has a FAIL"
    row = {r["metric"]: r for r in v.iter_rows(named=True)}
    dist = row["median distance per game (miles)"]
    mins = row["median minutes per game"]
    speed = row["median average speed (mph)"]

    w = pl.read_csv(OUT / "player_workload.csv")
    sub = w.filter(pl.col("live_minutes") >= MIN_LIVE_MINUTES)
    med_max = f"{sub['max_speed'].median():.1f}"
    near_cap = f"{(sub['max_speed'] >= 24.5).mean():.0%}"
    frames_scale = f"~{int(round(sub['frames'].median() / 10000)) * 10000:,}"
    p95 = f"{sub['p95_speed'].median():.1f}"

    return f"""\
## 4. Validation

### The check that matters: reproducing the league's own numbers

The strongest available check is not a rule of thumb. It is the NBA's own published
aggregates, computed by the vendor from the same SportVU feed and served through
`LeagueDashPtStats(pt_measure_type="SpeedDistance")` for 2015-16.

Independently flattening, de-duplicating, and differencing 25 Hz frames should
reproduce them. It does:

| Metric | This pipeline | NBA official | Difference |
|---|---|---|---|
| Median distance per game | **{dist["local"]:.2f} mi** | **{dist["official"]:.2f} mi** | {signed(dist["diff"])} |
| Median minutes per game | **{mins["local"]:.2f}** | **{mins["official"]:.2f}** | {signed(mins["diff"])} |
| Median average speed | **{speed["local"]:.2f} mph** | **{speed["official"]:.2f} mph** | {signed(speed["diff"])} |

`04_validate.py` exits non-zero if any of these drift outside tolerance.

**This is also how the de-duplication bug would have been caught.** Skipping that
step inflates distance ~3×, which would have shown up here as {dist["official"] * 3:.0f} miles per game
against the league's {dist["official"]:.1f}: impossible to miss, and impossible to detect without an
external reference.

### A statistic that did not survive: top speed

The first version of this analysis reported median top speed as {med_max} ft/s. That
number is meaningless.

**{near_cap} of players' observed maximum sits within 0.5 ft/s of the 25 ft/s glitch
filter.** The statistic was measuring the threshold, not the athlete. Optical
tracking produces occasional single-frame position jumps, and a maximum over {frames_scale}
frames will find them every time.

Top speed is therefore **not reported as a finding**. The 95th percentile
({p95} ft/s) is stable across players and games and is used instead. `max_speed`
stays in the output CSV, flagged as diagnostic only.

The general lesson: **an extreme-value statistic on noisy sensor data measures the
noise.** Ranking players by tracked top speed would have produced a leaderboard of
whoever got the worst camera frame."""


def limitations_section() -> str:
    pv = possession_numbers()
    return f"""\
## 5. Limitations

1. **The data is 2015-16.** Nine seasons stale, pre-dating the pace-and-space peak.
   Nothing here should be read as describing current NBA basketball. No public raw
   tracking exists for recent seasons.
2. **Ten games, early January 2016.** Chosen deterministically (alphabetically first
   in the archive) for reproducibility. Not a random sample of the season, so
   team-level numbers are not league-representative.
3. **Possession is a heuristic**, now validated: {pv["fga_rate"]} agreement with the
   shooting team at shots and {pv["to_rate"]} on held-out turnovers after per-game
   clock-latency calibration (see Finding C). It still misassigns during loose
   balls, steals, and contested rebounds, and the latency calibration itself
   assumes the offset is constant within a game.
4. **Spacing is convex hull area only.** It treats a well-spaced five-out set and a
   set with one player stranded in a corner as similar if the hulls match. It says
   nothing about whether the spacing was *useful*: no defender positions, no
   shot-quality outcome.
5. **Shot-outcome join**, now done (Finding C): spacing-at-shot vs eFG%, with
   the clock-latency artifact documented. Ten games is still a small sample
   for the flat-gradient conclusion; it rules out a large effect in this data,
   not a small one.
6. **Listed positions come from the game log**, not from where players actually
   stood."""


def modern_section() -> str:
    t = pl.read_csv(OUT / "modern_movement_trend.csv").sort("season")
    v = pl.read_csv(OUT / "modern_validation.csv")
    assert v["passed"].all(), "modern_validation has a failing season"
    assert v.height == t.height, "trend and validation season counts differ"
    recon_exp = round(math.log10(v["value"].median()))

    recent = t.tail(3)["miles_per_team_game"]
    lo, hi = recent.min(), recent.max()
    first = t.filter(pl.col("season") == "2013-14")["miles_per_team_game"][0]
    y16 = t.filter(pl.col("season") == "2015-16")["miles_per_team_game"][0]
    gap = (1 - y16 / ((lo + hi) / 2)) * 100
    fast = t.sort("avg_speed_mph_min_weighted").row(-1, named=True)

    return f"""\
## 7. Modern coverage: the aggregate half is current

The raw-frames half of this study is frozen in 2015-16 because the league
has published no raw tracking since. The aggregate half is not frozen.
`python/07_modern_aggregates.py` harvests
`LeagueDashPtStats(pt_measure_type="SpeedDistance")` for every season of
the tracking era, 2013-14 through 2025-26, at both player and team level,
and gates itself the family way: the player table's league totals must
reconcile with the independently queried team table, every season
(observed agreement ~1e{recon_exp}; `output/modern_validation.csv`).

The longitudinal read (`output/modern_movement_trend.csv`,
`figures/fig4_modern_movement.html`): teams now cover about {lo:.1f} to {hi:.1f}
miles per game, up from {first:.1f} in 2013-14. The raw-frames season analyzed
above (2015-16, {y16:.2f}) sits at the low end of the modern range, roughly {gap:.0f}%
below current movement volume. {fast["season"]} posts the fastest minutes-weighted
average speed of the tracking era ({fast["avg_speed_mph_min_weighted"]:.2f} mph). Context worth carrying into
any workload claim built on the 2015-16 sample."""


SECTIONS = {
    "scale": scale_section,
    "findings": findings_section,
    "validation": validation_section,
    "limitations": limitations_section,
    "modern": modern_section,
}


# ------------------------------------------------------------------ splice


def _pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- gen:{name} -->\n)(.*?)(\n<!-- /gen:{name} -->)", re.S)


def splice(text: str, name: str, body: str) -> str:
    pat = _pattern(name)
    if not pat.search(text):
        raise SystemExit(f"README is missing markers for gen:{name}")
    return pat.sub(lambda m: m.group(1) + body + m.group(3), text)


def committed_region(text: str, name: str) -> str:
    m = _pattern(name).search(text)
    if not m:
        raise SystemExit(f"README is missing markers for gen:{name}")
    return m.group(2)


def main(argv: list[str]) -> int:
    check = "--check" in argv
    text = README.read_text(encoding="utf-8")
    generated = {name: fn() for name, fn in SECTIONS.items()}

    if check:
        bad = [name for name, body in generated.items()
               if strip_abbr(committed_region(text, name)) != body]
        if bad:
            for name in bad:
                print(f"gen:{name}: committed README region does not match "
                      f"regeneration from output/")
            print(f"TRACKING FINDINGS CHECK FAILED: {len(bad)} of "
                  f"{len(generated)} regions stale (rerun 08_findings.py "
                  f"then glossary.py --sync)")
            return 1
        print(f"TRACKING FINDINGS CHECK PASSED: all {len(generated)} README "
              f"regions regenerate from output/")
        return 0

    for name, body in generated.items():
        text = splice(text, name, body)
    README.write_text(text, encoding="utf-8")
    print(f"Updated {len(generated)} generated regions in {README}")
    print("Now run: python ../basketball-analysis-tools/glossary.py --sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
