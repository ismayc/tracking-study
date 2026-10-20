"""Commit the possession-join diagnostics the story and its diagrams draw on.

06_possession_join.py picks each game's sampling lead and reports only the
winner. This script keeps the evidence behind that choice, so the page can
show it and every number quoted about it is read from a committed table:

  output/lead_sweep.csv
      shot agreement at every lead on 06's grid, per game. The best lead
      per game reproduces 06's calibrated_lead_s (the --check gate).
  output/spacing_efg_quartiles_naive.csv
      eFG% by spacing quartile with spacing sampled at the naive fixed lead
      (NAIVE_LEAD, the "about 1 s before each event's clock" join the README
      describes), the same computation 06 runs at the calibrated lead.
  output/example_shot_join.csv
      one illustrative shot from the game whose same-clock join agrees least
      (rows of kind "frame": possession label and ball height every
      FRAME_STEP-th frame; one row of kind "event": the shot).
  output/example_spacing_frames.csv
      two single frames from that game: the shot whose spacing is closest
      to the tightest quartile's mean, and the one closest to the widest
      quartile's mean, with every player's and the ball's position.

It also draws figures/fig3_spacing_vs_efg (calibrated and naive joins side
by side), which 06 used to draw with the calibrated join alone.

The example rows are raw tracking extracts from data/moments/ and data/pbp/,
which this public repo already commits; the other tables are aggregates.

Run:     python python/11_join_diagnostics.py          (about two minutes)
--figure: redraw fig3 from the committed tables only.
--check: verify the committed tables agree with each other and with 06's
         outputs, without recomputing from frames.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = ROOT / "figures"
MOM_DIR = ROOT / "data" / "moments"
PBP_DIR = ROOT / "data" / "pbp"

_spec = importlib.util.spec_from_file_location(
    "possession_join", Path(__file__).with_name("06_possession_join.py"))
j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(j)
a3 = j._a3

NAIVE_LEAD = 1.0       # s; the README's "about 1 s before each event's clock"
FRAME_STEP = 5         # keep every 5th frame (5 Hz) in the example strip
BEFORE_S = 9.0         # strip starts this long before the stamp
AFTER_S = 1.0          # and ends this long after it
MIN_ARC_Z = 12.0       # ft; the ball clears this on a jump shot's arc
MAX_ARC_S = 2.0        # s; longer above MIN_ARC_Z means more than one shot


def quartiles(shots: pl.DataFrame) -> pl.DataFrame:
    """06's eFG%-by-spacing-quartile table, for any set of joined shots."""
    qs = [shots["hull_area"].quantile(q) for q in (0.25, 0.5, 0.75)]
    shots = shots.with_columns(
        quartile=pl.when(pl.col("hull_area") <= qs[0]).then(pl.lit("Q1 (tightest)"))
        .when(pl.col("hull_area") <= qs[1]).then(pl.lit("Q2"))
        .when(pl.col("hull_area") <= qs[2]).then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4 (widest)")))
    return (shots.group_by("quartile")
            .agg(n=pl.len(),
                 mean_hull=pl.col("hull_area").mean(),
                 efg=(pl.col("made").sum()
                      + 0.5 * (pl.col("made") & (pl.col("shot_value") == 3)).sum())
                     / pl.len(),
                 share_3pt=(pl.col("shot_value") == 3).mean())
            .sort("mean_hull"))


def joined_shots(pbp: pl.DataFrame, hulls: pl.DataFrame, lead: float,
                 game: str) -> list[dict]:
    """06's spacing-at-the-shot join at a given lead."""
    rows = []
    for ev in pbp.filter(pl.col("is_fga")).iter_rows(named=True):
        if ev["team_id"] == 0:
            continue
        lo = ev["secs_left"] + lead
        h = hulls.filter((pl.col("period") == ev["period"])
                         & (pl.col("game_clock") >= lo)
                         & (pl.col("game_clock") <= lo + j.LOOKBACK_S)
                         & (pl.col("team_id") == ev["team_id"]))
        if h.height >= j.MIN_FRAMES:
            rows.append({"game": game, "period": ev["period"],
                         "secs_left": ev["secs_left"],
                         "hull_area": float(h["hull_area"].mean()),
                         "made": bool(ev["made"]),
                         "shot_value": float(ev["shot_value"] or 0),
                         "team_id": ev["team_id"]})
    return rows


def window_counts(poss: pl.DataFrame, period: int, clock: float,
                  lead: float) -> pl.DataFrame | None:
    """06's label window, as (off_team_id, n) sorted by n, then team id."""
    lo, hi = clock + lead, clock + lead + j.LOOKBACK_S
    win = poss.filter((pl.col("period") == period)
                      & (pl.col("game_clock") >= lo)
                      & (pl.col("game_clock") <= hi))
    if win.height < j.MIN_FRAMES:
        return None
    return (win.group_by("off_team_id").agg(n=pl.len())
            .sort(["n", "off_team_id"], descending=[True, False]))


def label(poss, period, clock, lead) -> int | None:
    """06's window_label with ties broken by team id. 06 sorts on the count
    alone, so a window split evenly between the teams gets whichever team
    polars' group_by returns first, which varies run to run; this copy is
    deterministic so the committed sweep reproduces."""
    c = window_counts(poss, period, clock, lead)
    return None if c is None else int(c["off_team_id"][0])


def sweep_point(poss, pbp, lead) -> tuple[int, int, int]:
    """Shots checked, shots agreeing, and windows split evenly between the
    teams (where 06's own label is a coin flip) at one lead."""
    n = agree = tied = 0
    for ev in pbp.filter(pl.col("is_fga")).iter_rows(named=True):
        if ev["team_id"] == 0:
            continue
        c = window_counts(poss, ev["period"], ev["secs_left"], lead)
        if c is None:
            continue
        n += 1
        agree += int(c["off_team_id"][0]) == ev["team_id"]
        tied += c.height > 1 and c["n"][0] == c["n"][1]
    return n, agree, tied


def pick_example(mom, pbp, poss, lead):
    """First made three where the same-clock window names the other team and
    the calibrated window names the shooter's, with one clear shot arc and an
    unbroken possession: every labeled frame from the strip's start to the
    arc is the shooter's team, every one from the arc's landing to the stamp
    is the other team's, and the clock runs throughout (no stopped frames)."""
    ball = (mom.filter(pl.col("entity") == "ball")
            .unique(subset=["period", "game_clock"], keep="first"))
    for ev in pbp.filter(pl.col("is_fga") & pl.col("made")
                         & (pl.col("shot_value") == 3)).iter_rows(named=True):
        if ev["team_id"] == 0:
            continue
        s0, p = ev["secs_left"], ev["period"]
        at_stamp = label(poss, p, s0, 0.0)
        cal = label(poss, p, s0, lead)
        if at_stamp is None or at_stamp == ev["team_id"] or cal != ev["team_id"]:
            continue
        arc = (ball.filter((pl.col("period") == p)
                           & pl.col("game_clock").is_between(s0 - AFTER_S, s0 + BEFORE_S))
               .sort("game_clock", descending=True))
        if not arc.height or arc["z"].max() < MIN_ARC_Z:
            continue
        gaps = arc["game_clock"].diff().abs().drop_nulls()
        if gaps.max() > a3.MAX_DT:
            continue
        air = arc.filter(pl.col("z") >= MIN_ARC_Z)["game_clock"]
        up, down = air.max(), air.min()
        if up - down > MAX_ARC_S or up < s0:
            continue   # one arc, before the stamp
        lab = poss.filter((pl.col("period") == p)
                          & pl.col("game_clock").is_between(s0 - AFTER_S, s0 + BEFORE_S))
        pre = lab.filter(pl.col("game_clock") > up)["off_team_id"].unique().to_list()
        post = lab.filter(pl.col("game_clock") < down)["off_team_id"].unique().to_list()
        if pre == [ev["team_id"]] and post == [at_stamp]:
            return ev, ball
    raise SystemExit("no example shot meets the rule")


def example_rows(game, ev, ball, poss, lead) -> pl.DataFrame:
    s0 = ev["secs_left"]
    p = ev["period"]
    frames = (ball.filter((pl.col("period") == p)
                          & pl.col("game_clock").is_between(s0 - AFTER_S, s0 + BEFORE_S))
              .sort("game_clock", descending=True)
              .with_row_index("i").filter(pl.col("i") % FRAME_STEP == 0)
              .select("game_clock", "z")
              .join(poss.filter(pl.col("period") == p).select("game_clock", "off_team_id"),
                    on="game_clock", how="left")
              .with_columns(t_before_stamp=pl.col("game_clock") - s0))
    other = [t for t in frames["off_team_id"].drop_nulls().unique().to_list()
             if t != ev["team_id"]]
    event = pl.DataFrame([{
        "kind": "event", "game": game, "period": p, "game_clock": s0,
        "t_before_stamp": 0.0, "z": None, "off_team_id": ev["team_id"],
        "shooter_team": ev["teamTricode"], "shooter": ev["playerName"],
        "shot_value": float(ev["shot_value"]), "made": bool(ev["made"]),
        "shooter_team_id": ev["team_id"],
        "other_team_id": other[0] if other else None,
        "calibrated_lead_s": lead, "naive_lead_s": NAIVE_LEAD,
        "lookback_s": j.LOOKBACK_S}])
    fr = frames.with_columns(kind=pl.lit("frame"), game=pl.lit(game),
                             period=pl.lit(p))
    return pl.concat([event, fr], how="diagonal_relaxed").select(event.columns)


def spacing_frames(game, mom, pbp, poss, hulls, lead, targets) -> pl.DataFrame:
    shots = pl.DataFrame(joined_shots(pbp, hulls, lead, game))
    out = []
    for label, target in targets:
        s = shots.with_columns(d=(pl.col("hull_area") - target).abs()).sort("d").row(
            0, named=True)
        lo = s["secs_left"] + lead
        win = hulls.filter((pl.col("period") == s["period"])
                           & pl.col("game_clock").is_between(lo, lo + j.LOOKBACK_S)
                           & (pl.col("team_id") == s["team_id"]))
        fr = win.with_columns(d=(pl.col("hull_area") - s["hull_area"]).abs()).sort(
            "d").row(0, named=True)
        pts = mom.filter((pl.col("period") == s["period"])
                         & (pl.col("game_clock") == fr["game_clock"]))
        out.append(pts.select("entity", "team_id", "x", "y").with_columns(
            example=pl.lit(label), game=pl.lit(game), period=pl.lit(s["period"]),
            game_clock=pl.lit(fr["game_clock"]), stamp_secs=pl.lit(s["secs_left"]),
            offense_team_id=pl.lit(s["team_id"]), made=pl.lit(s["made"]),
            shot_value=pl.lit(s["shot_value"]),
            frame_hull=pl.lit(fr["hull_area"]),
            window_mean_hull=pl.lit(s["hull_area"])))
    return pl.concat(out).select(
        "example", "game", "period", "game_clock", "stamp_secs", "offense_team_id",
        "made", "shot_value", "frame_hull", "window_mean_hull", "entity", "team_id",
        "x", "y")


def fig3(cal: pl.DataFrame, naive: pl.DataFrame, n_games: int) -> None:
    n_cal = int(cal["n"].sum())
    n_nv = int(naive["n"].sum())
    lab = cal["quartile"].to_list()
    fig = go.Figure(layout=a3.base_layout(
        "Sampled at the right moment, spacing barely moves eFG%",
        f"eFG% by offensive spacing quartile, spacing in the {j.LOOKBACK_S:g} s window "
        f"before each shot · calibrated join {n_cal:,} shots, naive {NAIVE_LEAD:g} s join "
        f"{n_nv:,} shots · {n_games} games, 2015-16",
        "Offensive convex-hull area before the shot (quartiles)", "Effective FG%"))
    fig.add_trace(go.Bar(
        name="calibrated join", x=lab, y=cal["efg"].to_list(),
        marker=dict(color=a3.BLUE, line=dict(width=0)),
        text=[f"{v:.1%}" for v in cal["efg"]], textposition="outside",
        textfont=dict(color=a3.MUTED, size=11),
        hovertemplate="calibrated: eFG %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(
        name=f"naive {NAIVE_LEAD:g} s join", x=lab, y=naive["efg"].to_list(),
        marker=dict(color=a3.MUTED, line=dict(width=0)),
        text=[f"{v:.1%}" for v in naive["efg"]], textposition="outside",
        textfont=dict(color=a3.MUTED, size=11),
        hovertemplate="naive: eFG %{y:.3f}<extra></extra>"))
    top = float(max(cal["efg"].max(), naive["efg"].max()))
    fig.update_layout(barmode="group", showlegend=True,
                      legend=dict(orientation="h", x=0.01, y=0.99, xanchor="left",
                                  yanchor="top", bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(tickformat=".0%", range=[0, top * 1.3])
    fig.add_annotation(
        x=lab[-1], y=float(naive["efg"][-1]), yshift=34, showarrow=False,
        text=f"naive: {naive['efg'][0]:.0%} to {naive['efg'][-1]:.0%}",
        font=dict(color=a3.INK2, size=12), xanchor="center")
    a3.save(fig, "fig3_spacing_vs_efg")


def compute() -> int:
    val = pl.read_csv(OUT / "possession_validation.csv")
    leads = dict(val.select("game", "calibrated_lead_s").iter_rows())
    games = sorted(MOM_DIR.glob("*.parquet"))
    sweep, naive_rows, cache = [], [], {}
    for path in games:
        g = path.stem
        mom = pl.read_parquet(path)
        pbp = j.load_pbp(PBP_DIR / path.name)
        poss = a3.possession_frames(mom)
        hulls = j.hulls_by_clock(mom, poss)
        for lead in j.LEAD_GRID:
            n, agree, tied = sweep_point(poss, pbp, lead)
            sweep.append({"game": g, "lead_s": lead, "shots_checked": n,
                          "shots_agree": agree, "tied_windows": tied,
                          "rate": agree / n if n else None})
        naive_rows += joined_shots(pbp, hulls, NAIVE_LEAD, g)
        print(f"  {g}", flush=True)
        cache[g] = (mom, pbp, poss, hulls)
    sw = pl.DataFrame(sweep)
    sw.write_csv(OUT / "lead_sweep.csv")
    naive = quartiles(pl.DataFrame(naive_rows))
    naive.write_csv(OUT / "spacing_efg_quartiles_naive.csv")

    worst = (sw.filter(pl.col("lead_s") == 0.0).sort("rate").row(0, named=True)["game"])
    mom, pbp, poss, hulls = cache[worst]
    ev, ball = pick_example(mom, pbp, poss, leads[worst])
    example_rows(worst, ev, ball, poss, leads[worst]).write_csv(OUT / "example_shot_join.csv")
    cal = pl.read_csv(OUT / "spacing_efg_quartiles.csv")
    targets = [("tight", float(cal["mean_hull"][0])), ("wide", float(cal["mean_hull"][-1]))]
    spacing_frames(worst, mom, pbp, poss, hulls, leads[worst], targets).write_csv(
        OUT / "example_spacing_frames.csv")
    fig3(cal, naive, val.height)
    print("wrote output/lead_sweep.csv, output/spacing_efg_quartiles_naive.csv, "
          "output/example_shot_join.csv, output/example_spacing_frames.csv, "
          "figures/fig3_spacing_vs_efg")
    return 0


def check() -> int:
    fails = []
    val = pl.read_csv(OUT / "possession_validation.csv")
    sw = pl.read_csv(OUT / "lead_sweep.csv")
    best = (sw.sort(["game", "rate", "lead_s"], descending=[False, True, False])
            .group_by("game", maintain_order=True).first())
    m = best.join(val, on="game", how="full", coalesce=True)
    # 06 breaks a window split evenly between the teams arbitrarily (see
    # label()), so its counts can differ from this sweep's by at most the
    # number of such windows, and its chosen lead must be best up to them.
    for r in m.iter_rows(named=True):
        at = sw.filter((pl.col("game") == r["game"])
                       & (pl.col("lead_s") == r["calibrated_lead_s"])).row(0, named=True)
        if at["shots_checked"] != r["fga_checked"]:
            fails.append(f"{r['game']}: shots checked at 06's lead differ")
        if abs(at["shots_agree"] - r["fga_agree"]) > at["tied_windows"]:
            fails.append(f"{r['game']}: agreement at 06's lead differs by more than "
                         f"its {at['tied_windows']} evenly split windows")
        best_lo = (r["shots_agree"] - r["tied_windows"]) / r["shots_checked"]
        at_hi = (at["shots_agree"] + at["tied_windows"]) / at["shots_checked"]
        if best_lo > at_hi:
            fails.append(f"{r['game']}: 06's lead {r['calibrated_lead_s']} is not "
                         f"the best up to evenly split windows")
    if sorted(sw["lead_s"].unique().to_list()) != j.LEAD_GRID:
        fails.append("sweep leads differ from 06's LEAD_GRID")
    naive = pl.read_csv(OUT / "spacing_efg_quartiles_naive.csv")
    if naive.height != 4 or naive["n"].min() < naive["n"].max() - 1:
        fails.append("naive quartile table is not four near-equal quartiles")
    ex = pl.read_csv(OUT / "example_shot_join.csv")
    e = ex.filter(pl.col("kind") == "event").row(0, named=True)
    worst = sw.filter(pl.col("lead_s") == 0.0).sort("rate").row(0, named=True)["game"]
    if e["game"] != worst:
        fails.append("example shot is not from the worst same-clock game")
    if e["calibrated_lead_s"] != dict(val.select("game", "calibrated_lead_s")
                                      .iter_rows())[e["game"]]:
        fails.append("example shot's lead differs from 06's")
    sp = pl.read_csv(OUT / "example_spacing_frames.csv")
    for lab, grp in sp.group_by("example"):
        off = grp.filter((pl.col("entity") == "player")
                         & (pl.col("team_id") == pl.col("offense_team_id")))
        area = a3.hull_area(off["x"].to_numpy(), off["y"].to_numpy())
        if off.height != 5 or not np.isclose(area, grp["frame_hull"][0], atol=1e-9):
            fails.append(f"spacing frame {lab[0]}: hull does not recompute")
    for f in fails:
        print("FAIL", f)
    print("JOIN DIAGNOSTICS", "FAILED" if fails else
          f"PASSED: 06's lead is the sweep's best in all {best.height} games, up to "
          f"evenly split windows; examples consistent")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--figure", action="store_true",
                    help="redraw fig3 from the committed tables only")
    args = ap.parse_args()
    if args.figure:
        fig3(pl.read_csv(OUT / "spacing_efg_quartiles.csv"),
             pl.read_csv(OUT / "spacing_efg_quartiles_naive.csv"),
             pl.read_csv(OUT / "possession_validation.csv").height)
        return 0
    return check() if args.check else compute()


if __name__ == "__main__":
    sys.exit(main())
