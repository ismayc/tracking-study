"""Explainer diagrams for the story page, drawn with the family's famdiagram
(../basketball-analysis-tools/famdiagram.py) and saved to figures/<ref>.svg
(plus a <ref>_narrow.svg phone layout). report_builder.py inlines them.

Every number drawn is read from a committed output (11_join_diagnostics.py
writes the example and sweep tables) or from the study's own constants.

Diagrams:
  dg1_one_shot      a timeline of one real made three: who the tracking says
                    has the ball, the ball's height, the shot's milestones,
                    and the window each join samples
                    (output/example_shot_join.csv)
  dg2_lead_sweep    shot agreement against how far before the play-by-play
                    stamp the window sits, every game, with each game's
                    chosen lead (output/lead_sweep.csv,
                    output/possession_validation.csv)
  dg3_spacing_court what the spacing number measures: two real frames on a
                    half court, the offense's convex hull drawn and sized
                    (output/example_spacing_frames.csv,
                    output/spacing_efg_quartiles.csv)

Run:     python python/12_diagrams.py
--check: rebuild every SVG in memory and fail if a committed one differs.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT.parent / "basketball-analysis-tools"))
import famdiagram as fd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tracking_analysis", Path(__file__).with_name("03_analysis.py"))
a3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a3)

COURT_L, COURT_W = 94.0, 50.0       # ft, the court
RIM_X, ARC_R, CORNER = 5.25, 23.75, 22.0   # ft, NBA geometry (drawn only)


def wrap(s: str, width: float, size: float) -> list[str]:
    per = size * 0.55
    out, cur = [], ""
    for w in s.split():
        t = f"{cur} {w}".strip()
        if cur and len(t) * per > width:
            out.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        out.append(cur)
    return out


def lines(d, x, y, s, width, size, **kw) -> float:
    gap = size * 1.3
    ls = wrap(s, width, size)
    for i, ln in enumerate(ls):
        d.text(x, y + i * gap, ln, size=size, **kw)
    return y + (len(ls) - 1) * gap


def mmss(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def game_name(g: str) -> str:
    """'01.01.2016.NYK.at.CHI' -> ('NYK at CHI', 'January 1, 2016')."""
    m, dd, y, away, _, home = g.split(".")
    month = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"][int(m) - 1]
    return f"{away} at {home}", f"{month} {int(dd)}, {y}"


# ---------------------------------------------------------------------------
# 1. One shot on a timeline
# ---------------------------------------------------------------------------

def majority(fr: pl.DataFrame, lo: float, hi: float):
    w = fr.filter(pl.col("t_before_stamp").is_between(lo, hi)
                  & pl.col("off_team_id").is_not_null())
    if not w.height:
        return None
    c = (w.group_by("off_team_id").agg(n=pl.len())
         .sort(["n", "off_team_id"], descending=[True, False]))
    return int(c["off_team_id"][0])


def dg_one_shot(layout: str) -> fd.Diagram:
    ex = pl.read_csv(OUT / "example_shot_join.csv")
    ev = ex.filter(pl.col("kind") == "event").row(0, named=True)
    fr = ex.filter(pl.col("kind") == "frame").sort("t_before_stamp", descending=True)
    shooter_id, other_id = ev["shooter_team_id"], ev["other_team_id"]
    teams = ev["game"].split(".")
    tri = {shooter_id: ev["shooter_team"],
           other_id: [t for t in (teams[3], teams[5]) if t != ev["shooter_team"]][0]}
    lead, naive, lb = ev["calibrated_lead_s"], ev["naive_lead_s"], ev["lookback_s"]
    matchup, when = game_name(ev["game"])
    t_hi = float(fr["t_before_stamp"].max())
    t_lo = float(fr["t_before_stamp"].min())
    peak = fr.sort("z", descending=True).row(0, named=True)
    landed = fr.filter((pl.col("off_team_id") == other_id)
                       & (pl.col("t_before_stamp") < peak["t_before_stamp"])).row(
        0, named=True)
    wide = layout == "wide"
    W = 760 if wide else 360
    s = 14 if wide else 12
    pad = 20 if wide else 14
    title = (f"One made three, {ev['shooter']} for {ev['shooter_team']}, {matchup}, "
             f"{when}: the ball is in the air {peak['t_before_stamp']:.1f} s before the "
             f"play-by-play stamp")
    d = fd.Diagram(W, 0, title)
    y = lines(d, pad, 26, f"{ev['shooter']}'s three for {ev['shooter_team']}, {matchup}, "
              f"{when}, Q{ev['period']} {mmss(ev['game_clock'])}", W - 2 * pad, s + 2,
              weight="600")
    y = lines(d, pad, y + s + 8, "Time runs left to right, in seconds before the "
              "moment the play-by-play logs the shot.", W - 2 * pad, s, fill="ink3")
    x0, x1 = pad + (6 if wide else 4), W - pad - 6
    X = lambda t: x0 + (t_hi - t) / (t_hi - t_lo) * (x1 - x0)  # noqa: E731
    # possession ribbon
    ry = y + 40
    d.text(x0, ry - 8, "Who the tracking says has the ball", size=s, fill="ink2",
           weight="600")
    ts = fr["t_before_stamp"].to_list()
    labs = fr["off_team_id"].to_list()
    runs = []   # contiguous runs of one label: [label, start t, end t]
    for i, (t, lab) in enumerate(zip(ts, labs)):
        t_next = ts[i + 1] if i + 1 < len(ts) else t_lo
        if runs and runs[-1][0] == lab:
            runs[-1][2] = t_next
        else:
            runs.append([lab, t, t_next])
    for lab, ta, tb in runs:
        if lab is not None:
            d.rect(X(ta), ry, X(tb) - X(ta), 16,
                   fill="nba" if lab == shooter_id else "c4")
    lx = x0
    for lab, col in ((tri[shooter_id], "nba"), (tri[other_id], "c4")):
        d.rect(lx, ry + 24, 12, 12, fill=col, rx=2)
        d.text(lx + 18, ry + 35, lab, size=s, fill="ink2")
        lx += 80
    blank = "blank: no one within reach of the ball"
    if wide:
        d.text(lx, ry + 35, blank, size=s, fill="ink3")
        hy = ry + 72
    else:
        d.text(x0, ry + 35 + s + 8, blank, size=s, fill="ink3")
        hy = ry + 72 + s + 8
    # ball height
    hh = 110 if wide else 100
    zmax = float(fr["z"].max())
    Y = lambda z: hy + hh - z / zmax * hh  # noqa: E731
    d.text(x0, hy - 8, "Ball height", size=s, fill="ink2", weight="600")
    d.line(x0, Y(0), x1, Y(0), stroke="rule", sw=1)
    d.line(x0, Y(a3.POSSESSION_MAX_BALL_Z), x1, Y(a3.POSSESSION_MAX_BALL_Z),
           stroke="rule", sw=1, dash="3 3")
    d.text(x1, Y(a3.POSSESSION_MAX_BALL_Z) - 4, f"{a3.POSSESSION_MAX_BALL_Z:g} ft",
           size=s, fill="ink3", anchor="end")
    d.polyline([(X(t), Y(z)) for t, z in fr.select("t_before_stamp", "z").iter_rows()],
               stroke="ink2", sw=1.8)
    # milestones
    my = hy + hh + 18
    marks = [(peak["t_before_stamp"], "shot at its peak", "ink"),
             (landed["t_before_stamp"], f"{tri[other_id]} has it", "c4"),
             (0.0, "play-by-play stamp", "c8")]
    for k, (t, lab, col) in enumerate(marks):
        d.line(X(t), ry - 2, X(t), my + 2, stroke=col, sw=1.4,
               dash=None if col == "c8" else "4 3")
        d.circle(X(t), my + 6, 4, fill=col)
        tx = X(t)
        anchor = "end" if tx + len(lab) * s * 0.3 > x1 else "middle"
        d.text(tx + (4 if anchor == "end" else 0), my + 24 + k * (s + 6), lab, size=s,
               fill=col, anchor=anchor, weight="600")
    # windows
    wy = my + 24 + 3 * (s + 6) + (30 if wide else 40)
    d.text(x0, wy - 10, f"What each join samples: a {lb:g} s window, named for the "
           f"team with the ball most", size=s, fill="ink2", weight="600") if wide else \
        lines(d, x0, wy - 26, f"What each join samples: a {lb:g} s window, named for "
              f"the team with the ball most", W - 2 * pad, s, fill="ink2", weight="600")
    wins = [(0.0, "at the stamp"), (naive, f"{naive:g} s before"),
            (lead, f"calibrated, {lead:g} s before")]
    for k, (ld, name) in enumerate(wins):
        who = majority(fr, ld, ld + lb)
        ok = who == shooter_id
        col = "nba" if ok else "c4"
        by = wy + k * (s + 22)
        d.rect(X(ld + lb), by, X(ld) - X(ld + lb), 14, fill=col, opacity=0.85, rx=3)
        txt = f"{name}: {tri.get(who, 'no one')}" + (" (right)" if ok else " (wrong)")
        if X(ld + lb) - x0 > len(txt) * s * 0.55 + 10:
            d.text(X(ld + lb) - 8, by + 12, txt, size=s, fill=col, anchor="end",
                   weight="600")
        else:
            d.text(X(ld) + 8, by + 12, txt, size=s, fill=col, weight="600")
    ay = wy + 3 * (s + 22) + 6
    d.line(x0, ay, x1, ay, stroke="rule", sw=1)
    t = int(t_hi)
    while t >= t_lo:
        d.line(X(t), ay, X(t), ay + 4, stroke="rule", sw=1)
        d.text(X(t), ay + s + 6, f"{t}", size=s, fill="ink3", anchor="middle")
        t -= 1 if wide else 2
    end = lines(d, pad, ay + 2 * s + 18, "seconds before the stamp", W - 2 * pad, s,
                fill="ink3")
    d.h = end + 14
    return d


# ---------------------------------------------------------------------------
# 2. The lead sweep
# ---------------------------------------------------------------------------

def dg_lead_sweep(layout: str) -> fd.Diagram:
    sw = pl.read_csv(OUT / "lead_sweep.csv")
    val = pl.read_csv(OUT / "possession_validation.csv")
    ex = pl.read_csv(OUT / "example_shot_join.csv").filter(pl.col("kind") == "event")
    ex_game = ex["game"][0]
    games = val["game"].to_list()
    leads = sorted(sw["lead_s"].unique().to_list())
    at0 = sw.filter(pl.col("lead_s") == leads[0])
    lo_r, hi_r = at0["rate"].min(), at0["rate"].max()
    chosen = val["calibrated_lead_s"]
    wide = layout == "wide"
    W = 760 if wide else 360
    s = 14 if wide else 12
    pad = 20 if wide else 14
    title = (f"Shot agreement against how far before the stamp the window sits: "
             f"{lo_r:.0%} to {hi_r:.0%} at the stamp, each game's best between "
             f"{chosen.min():g} and {chosen.max():g} s")
    d = fd.Diagram(W, 0, title)
    y = lines(d, pad, 26, "Move the window earlier, one half second at a time",
              W - 2 * pad, s + 2, weight="600")
    y = lines(d, pad, y + s + 8, "Each line is one game: the share of its shots whose "
              "window names the shooting team. Dots: the lead the calibration chose for that game.",
              W - 2 * pad, s, fill="ink3")
    gx0, gx1 = pad + (44 if wide else 38), W - pad - 8
    gy0, gh = y + 30, 280 if wide else 250
    X = lambda v: gx0 + (v - leads[0]) / (leads[-1] - leads[0]) * (gx1 - gx0)  # noqa: E731
    Y = lambda r: gy0 + gh - r * gh  # noqa: E731
    d.rect(X(chosen.min()), gy0, X(chosen.max()) - X(chosen.min()), gh, fill="nba",
           opacity=0.08)
    d.text((X(chosen.min()) + X(chosen.max())) / 2, gy0 + gh - 10,
           "range of chosen leads", size=s, fill="nba", anchor="middle")
    for r in (0, 0.25, 0.5, 0.75, 1.0):
        d.line(gx0, Y(r), gx1, Y(r), stroke="rule", sw=1 if r else 1.4)
        d.text(gx0 - 6, Y(r) + 4, f"{r:.0%}", size=s, fill="ink3", anchor="end")
    for g in games:
        gr = sw.filter(pl.col("game") == g).sort("lead_s")
        pts = [(X(a), Y(b)) for a, b in gr.select("lead_s", "rate").iter_rows()]
        hl = g == ex_game
        d.polyline(pts, stroke="c4" if hl else "ink3", sw=2.2 if hl else 1.1,
                   opacity=None if hl else 0.7)
    for g, ld in val.select("game", "calibrated_lead_s").iter_rows():
        r = sw.filter((pl.col("game") == g) & (pl.col("lead_s") == ld))["rate"][0]
        d.circle(X(ld), Y(r), 4, fill="nba")
    exr = sw.filter((pl.col("game") == ex_game) & (pl.col("lead_s") == leads[0]))["rate"][0]
    m, _ = game_name(ex_game)
    d.text(X(leads[0]) + 8, Y(exr) + 4, f"{m}: {exr:.1%}", size=s, fill="c4",
           weight="600")
    for v in range(int(leads[0]), int(leads[-1]) + 1, 1 if wide else 2):
        d.text(X(v), gy0 + gh + s + 6, f"{v}", size=s, fill="ink3", anchor="middle")
    step = sw.sort("lead_s")["lead_s"].unique().diff().drop_nulls().max()
    end = lines(d, pad, gy0 + gh + 2 * s + 18, "seconds before the stamp where the "
                f"window starts, in {step:g} s steps", W - 2 * pad, s, fill="ink3")
    d.h = end + 14
    return d


# ---------------------------------------------------------------------------
# 3. What spacing measures
# ---------------------------------------------------------------------------

def court(d, x0, y0, k, depth):
    """Half court, baseline at the bottom, `depth` ft deep; k px per ft."""
    X = lambda a: x0 + a * k  # noqa: E731  a: across, 0..50
    Y = lambda b: y0 + (depth - b) * k  # noqa: E731  b: from the baseline
    d.rect(X(0), Y(depth), COURT_W * k, depth * k, fill="surface2", stroke="rule", sw=1)
    d.rect(X(COURT_W / 2 - 8), Y(19), 16 * k, 19 * k, fill="none", stroke="rule", sw=1)
    d.line(X(COURT_W / 2 - 3), Y(4), X(COURT_W / 2 + 3), Y(4), stroke="ink2", sw=2)
    d.circle(X(COURT_W / 2), Y(RIM_X), 0.75 * k, fill="none", stroke="c8", sw=2)
    cy = RIM_X + (ARC_R ** 2 - CORNER ** 2) ** 0.5
    c0, c1 = COURT_W / 2 - CORNER, COURT_W / 2 + CORNER
    d.path(f"M{X(c0):.1f},{Y(0):.1f} L{X(c0):.1f},{Y(cy):.1f} "
           f"A{ARC_R * k:.1f},{ARC_R * k:.1f} 0 0 1 {X(c1):.1f},{Y(cy):.1f} "
           f"L{X(c1):.1f},{Y(0):.1f}", stroke="ink3", sw=1.2, opacity=0.6)
    return X, Y


def dg_spacing_court(layout: str) -> fd.Diagram:
    sp = pl.read_csv(OUT / "example_spacing_frames.csv")
    q = pl.read_csv(OUT / "spacing_efg_quartiles.csv").sort("mean_hull")
    wide = layout == "wide"
    W = 760 if wide else 360
    s = 14 if wide else 12
    pad = 20 if wide else 14
    tight = sp.filter(pl.col("example") == "tight")
    widef = sp.filter(pl.col("example") == "wide")
    title = (f"Spacing is the area of the offense's five-player hull: "
             f"{tight['frame_hull'][0]:.0f} sq ft against {widef['frame_hull'][0]:.0f} "
             f"sq ft in two real frames")
    d = fd.Diagram(W, 0, title)
    y = lines(d, pad, 26, "Spacing: the area inside the five attackers", W - 2 * pad,
              s + 2, weight="600")
    y = lines(d, pad, y + s + 8, "Stretch a band around the five players on offense; "
              "spacing is the floor area inside it, in square feet.", W - 2 * pad, s,
              fill="ink3")
    depth = 40.0
    k = 6.4 if wide else 6.2
    pw = COURT_W * k
    end = y
    for i, (name, grp, qi) in enumerate((("tight", tight, 0), ("wide", widef, -1))):
        if wide:
            x0, y0 = pad + i * (pw + 50), y + 58
        else:
            x0, y0 = (W - pw) / 2, end + 58
        r = grp.row(0, named=True)
        qr = q.row(qi, named=True)
        X, Y = court(d, x0, y0, k, depth)
        off = r["offense_team_id"]
        players = grp.filter(pl.col("entity") == "player")
        ball = grp.filter(pl.col("entity") == "ball").row(0, named=True)
        ox = players.filter(pl.col("team_id") == off)["x"].mean()
        flip = ox > COURT_L / 2

        def tf(x, yv):
            b = COURT_L - x if flip else x
            a = COURT_W - yv if flip else yv
            return X(a), Y(min(b, depth))
        o = players.filter(pl.col("team_id") == off)
        pts = np.array([tf(x, yv) for x, yv in o.select("x", "y").iter_rows()])
        hull = pts[ConvexHull(pts).vertices]
        d.polyline([tuple(p) for p in hull] + [tuple(hull[0])], stroke="nba", sw=2,
                   fill="nba", opacity=0.22)
        d.polyline([tuple(p) for p in hull] + [tuple(hull[0])], stroke="nba", sw=2)
        for x, yv in players.filter(pl.col("team_id") != off).select("x", "y").iter_rows():
            cx, cy = tf(x, yv)
            d.circle(cx, cy, 5, fill="none", stroke="ink3", sw=1.6)
        for cx, cy in pts:
            d.circle(cx, cy, 5.5, fill="nba")
        bx, by = tf(ball["x"], ball["y"])
        d.circle(bx, by, 3.5, fill="c4")
        head = "Tight" if name == "tight" else "Wide"
        d.text(x0, y0 - 24, f"{head}: {r['frame_hull']:.0f} sq ft", size=s + 1,
               weight="600", fill="nba")
        d.text(x0, y0 - 8, f"{qr['quartile']} shots average {qr['mean_hull']:.0f} sq ft",
               size=s, fill="ink3")
        end = Y(0) + (8 if wide else 14)
    ly = end + s + 10
    d.circle(pad + 6, ly - 4, 5.5, fill="nba")
    d.text(pad + 16, ly, "offense", size=s, fill="ink2")
    d.circle(pad + 96, ly - 4, 5, fill="none", stroke="ink3", sw=1.6)
    d.text(pad + 106, ly, "defense", size=s, fill="ink2")
    d.circle(pad + 166, ly - 4, 3.5, fill="c4")
    d.text(pad + 176, ly, "ball", size=s, fill="ink2")
    d.circle(pad + 226, ly - 4, 4, fill="none", stroke="c8", sw=2)
    d.text(pad + 236, ly, "rim", size=s, fill="ink2")
    end = lines(d, pad, ly + s + 12, "Each frame is drawn with the basket the offense "
                "attacks at the bottom.", W - 2 * pad, s, fill="ink3")
    d.h = end + 14
    return d


BUILDERS = {"dg1_one_shot": dg_one_shot, "dg2_lead_sweep": dg_lead_sweep,
            "dg3_spacing_court": dg_spacing_court}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fails = []
    for ref, build in BUILDERS.items():
        wide, narrow = build("wide"), build("narrow")
        if args.check:
            for dg, name in ((wide, f"{ref}.svg"), (narrow, f"{ref}_narrow.svg")):
                p = FIG / name
                if not p.exists() or p.read_text() != dg.svg():
                    fails.append(name)
        else:
            for p in fd.save(FIG, ref, wide, narrow):
                print("wrote", p.relative_to(ROOT))
    if args.check:
        for f in fails:
            print("FAIL", f, "differs from a fresh build")
        print("DIAGRAMS", "FAILED" if fails else f"PASSED: {len(BUILDERS)} diagrams")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
