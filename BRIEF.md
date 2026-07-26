# One finding, three audiences: raw tracking (SportVU)

Same analysis, three registers. Numbers from `output/`.

---

## For the performance staff (one minute)

From 25-frames-per-second camera data across 10 games:

- A rotation player covers about **2 miles per game while the clock is
  running**, at an average moving speed around 4.2 mph, matching the NBA's
  own published numbers, which is how we know our processing is right.
- Don't trust "top speed" leaderboards from tracking data. The cameras
  occasionally glitch a player's position between frames, and a maximum over
  40,000 frames will always find the glitch. We use the 95th-percentile speed
  instead (~10 mph); it's stable from game to game.
- Offensive spacing: the five offensive players typically occupy about
  **530 sq ft, roughly a fifth of the half court**. Wider setups shift the
  shot mix toward threes; in these 10 games they did *not* clearly produce
  more efficient shots. Ten games is a small sample. Treat that as "no large
  effect," not "no effect."

## For analytics peers

7.5M de-duplicated frames (raw archive re-reports overlapping windows per
event: 3.07× duplication measured, the classic trap in this dataset).
Clock-derived dt so distance counts live play only; 25 ft/s glitch filter;
external validation against `LeagueDashPtStats` medians (2.00 vs 2.00 mi).
New in this pass: a validated join to play-by-play. Pbp event clocks lag the
tracking clock by a **per-game scorer latency of 2.5–6.0 s**; naive joining
gives 67% possession-heuristic agreement (one game below coin-flip), and
per-game calibration on shots with turnovers held out gives **97.5% / 93.5%**.
The uncalibrated join also manufactures a strong spurious spacing→eFG
gradient (42→53%); calibrated, the gradient is flat-ish with 3PA share rising
in spacing. Caveats: 2015-16 data (nothing newer is public), 10 games,
convex-hull spacing only, no defender positioning.

## For the executive summary (three bullets)

- We process raw NBA tracking data at scale and reproduce the league's own
  published numbers before reporting anything of our own.
- Two silent data traps found and neutralized: 3× frame duplication, and a
  2.5–6 s per-game clock misalignment that fabricates plausible-looking
  findings if unhandled.
- Possession detection from raw coordinates validated at 94–98% against
  play-by-play ground truth: the building block for possession-level
  tracking work.
