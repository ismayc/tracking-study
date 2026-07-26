# Seven and a half million frames, and the six-second trap hiding inside them

Optical tracking is the richest basketball data there is: x and y for all
ten players (and z for the ball), twenty-five times a second. It is also
the least forgiving. This study works with the raw 2015-16 SportVU feed
(7.5 million de-duplicated frames across ten games), and its most valuable
finding is not a basketball result at all. It is a data trap that would
have quietly fabricated a basketball result, and the calibration that
defused it.

Start with the part that builds trust. Before asking the tracking data
anything new, the pipeline asks it something the NBA has already answered:
how far do players run? Distance covered, recomputed frame-by-frame from
raw coordinates, reproduces the league's published player-tracking
aggregates. That check has to come first, because every claim downstream
rides on the coordinates being what they say they are.

{{fig:fig1_distance_covered|Distance covered per game, recomputed from raw frames against the NBA's published aggregates. This figure earns the study its license to say anything else: if the reconstruction disagreed with the league's own numbers, nothing downstream would be worth reading.}}

The interesting work begins when tracking meets play-by-play, because
that join is what turns anonymous coordinates into *possessions*: who had
the ball, which direction they were attacking, what happened next. The
naive join matches each play-by-play event to the tracking frame with the
same game clock. On the worst of the ten games, that naive join agreed
with a possession-direction heuristic just 24% of the time: worse than a
coin flip, on a binary label.

The cause was not the heuristic. **Play-by-play clocks lag the tracking
clock by a per-game scorer latency of 2.5 to 6.0 seconds**: a human at
the scorer's table, typing at human speed, in different arenas with
different reflexes. A fixed correction cannot fix it because the lag
differs by game; an uncorrected join lands events inside the *next*
possession. Calibrating the latency per game on shots only, then scoring
turnovers as a held-out check, agreement rises to 97.5% on 1,623 shot
events and 93.5% on the 232 held-out turnovers.

Here is why that matters beyond hygiene, and why the third figure is the
one to linger on. With calibrated windows, effective field-goal percentage
across floor-spacing quartiles is nearly flat (50.0%, 45.0%, 45.7%, 47.2%
from tightest to widest), while the share of threes rises steadily (23.5%
to 29.8%). In this sample, spacing changes *what shots are taken* more than
it changes raw efficiency. But run the same analysis with the naive,
uncalibrated join and you get a clean, monotone, publication-ready
"wider spacing = better shooting" gradient from 42% to 53%: **entirely an
artifact of sampling the floor at the wrong moment**. A junior analyst
would have shipped it; a coaching staff might have acted on it.

{{fig:fig3_spacing_vs_efg|Shooting efficiency by spacing quartile, measured at the calibrated moment of the shot. This is the study's cautionary centerpiece: the honest version is nearly flat, while the uncalibrated version of the same chart showed a strong clean gradient that was pure artifact. One figure, two joins, opposite conclusions: the argument for clock calibration in a single image.}}

{{fig:fig2_spacing_distribution|The distribution of team spacing across all frames: the raw material behind the quartile analysis, and the context for how much (or little) spacing actually varies possession to possession.}}

The study's honest boundary: ten games is a workload-and-methods sample,
not a league conclusion, and 2015-16 SportVU is not the current Hawk-Eye
feed. What transfers is the discipline: validate against published
aggregates before trusting coordinates, calibrate clocks before joining
sources, and hold out an event type the calibration never saw. The
spacing-gradient artifact is this family's standing exhibit for why.
