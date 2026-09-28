"""One hard map, one flight, shown in full.

Map 14 of `cluttered_world`: 45 obstacles, target 27 m away at a bearing of
about 170 deg -- almost directly behind the start heading -- and invisible
from the start.  It is the map the connectome solved and every other arm
failed, including the full-information planner, and where both hand-written
references flew due south to y = -38, the opposite direction.

This runs that single flight and shows what the circuit actually had to work
with: where it went, what its eyes saw at four moments along the way, when
the target first entered the frame, and what the descending readout was
commanding throughout.

Not a measurement.  A demonstration of one case that the aggregates cannot
show.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from environment.target_world import Agent, cluttered_world, shortest_path
from sensors.attraction_cue import AttractionCueSensor
from sim.episode import DT, V_CRUISE

from cx_search import GAIN, MAX_STEPS, oracle_cmd, planner_cmd
from cx_vision import VisionSub, build

MAP = 14


def fly(cmd, world, heading, sensor, inp=None, snaps=()):
    """One episode, keeping everything worth plotting."""
    ag = Agent(world=world, start=(0.0, 0.0, 2.0), heading=heading)
    rec = {"cmd": [], "bearing": [], "seen": [], "dist": [], "psi": []}
    views = {}
    for k in range(MAX_STEPS):
        cue = sensor.sense(ag.p, ag.heading, world.target, world.obstacles)
        if inp is not None and k in snaps:
            views[k] = (inp._lamina_luminance(world, ag.p, ag.heading).copy(),
                        ag.p.copy(), ag.heading)
        u = float(cmd(cue, ag.heading, ag))
        rec["cmd"].append(u)
        rec["bearing"].append(cue.bearing_deg if cue.valid else np.nan)
        rec["seen"].append(cue.valid)
        rec["dist"].append(float(np.linalg.norm(world.target - ag.p)))
        rec["psi"].append(ag.heading)
        ag.step(u, V_CRUISE, 0.0, DT)
        if ag.collided or world.reached(ag.p):
            break
    rec = {k: np.array(v) for k, v in rec.items()}
    return ag, rec, views


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net, ann, sub, rho, nl, nc = build()
    vs = VisionSub(net, ann)
    sen = AttractionCueSensor()
    world, heading = cluttered_world(seed=MAP)

    v = world.target - np.array([0.0, 0.0, 2.0])
    b0 = (math.degrees(math.atan2(v[1], v[0])) - math.degrees(heading)
          + 180) % 360 - 180
    print("map %d: target %.1f m away, bearing %+.0f deg from the start "
          "heading, %d obstacles"
          % (MAP, np.linalg.norm(v), b0, len(world.obstacles)))

    # first pass: no snapshots, just to find out how long the flight is
    vs.reset()
    ag, rec, _ = fly(lambda c, h, a: GAIN * vs.turn(world, a.p, h),
                     world, heading, sen)
    n = len(rec["cmd"])
    first_seen = int(np.argmax(rec["seen"])) if rec["seen"].any() else -1
    print("flight: %d steps, reached %s, target first seen at step %d "
          "(%.0f%% of steps in view)"
          % (n, world.reached(ag.p) and not ag.collided, first_seen,
             100 * rec["seen"].mean()))

    snaps = sorted({0, max(first_seen - 10, 1), first_seen, n - 1})[:4]
    vs.reset()
    ag, rec, views = fly(lambda c, h, a: GAIN * vs.turn(world, a.p, h),
                         world, heading, sen, inp=vs.inp, snaps=set(snaps))
    path = ag.path_array()

    others = {}
    for nm, f in (("P blind", lambda w: (lambda c, h, a:
                                         2.0 * math.radians(c.bearing_deg)
                                         if c.valid else 0.0)),
                  ("P + oracle", oracle_cmd),
                  ("planner", planner_cmd)):
        a2, r2, _ = fly(f(world), world, heading, sen)
        others[nm] = (a2.path_array(),
                      world.reached(a2.p) and not a2.collided)

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.5, 1, 1, 1],
                          height_ratios=[1, 1, 1], hspace=0.42, wspace=0.32)

    # --- the flight -------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    for o in world.obstacles:
        ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="0.87", zorder=1))
    sp = shortest_path(world)
    if sp:
        a = np.array(sp)
        ax.plot(a[:, 0], a[:, 1], ":", color="#27ae60", lw=1.1,
                label="grid shortest", zorder=2)
    for nm, (p2, ok) in others.items():
        ax.plot(p2[:, 0], p2[:, 1], lw=1.0, alpha=0.55, zorder=3,
                label="%s%s" % (nm, "" if ok else " (failed)"))
    ax.plot(path[:, 0], path[:, 1], color="#c0392b", lw=2.2,
            label="connectome", zorder=4)
    m = rec["seen"][:len(path) - 1]
    if m.any():
        ax.scatter(path[:-1][m][:, 0], path[:-1][m][:, 1], s=7,
                   color="#f39c12", zorder=5, label="target in view")
    for k in snaps:
        if k in views:
            _, pk, _ = views[k]
            ax.annotate("t=%d" % k, pk[:2], fontsize=8, zorder=7,
                        bbox=dict(fc="w", ec="0.6", alpha=0.85, pad=1.2))
    ax.add_patch(plt.Circle(tuple(world.target[:2]), world.reach_radius,
                            fill=False, ls="--", color="0.4", zorder=6))
    ax.plot(0, 0, "s", ms=8, color="k", zorder=7)
    ax.plot(*world.target[:2], marker="*", ms=18, color="#f39c12", zorder=8)
    ax.arrow(0, 0, 4 * math.cos(heading), 4 * math.sin(heading),
             head_width=1.0, color="k", zorder=7)
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="best")
    ax.set_title("map %d: target %.0f m at %+.0f deg, unseen at t=0"
                 % (MAP, np.linalg.norm(v), b0), fontsize=10)

    # --- what the eyes saw ------------------------------------------
    az = vs.inp.eye_az
    el = vs.inp.eye_el
    band = np.abs(el) < 25
    edges = np.arange(-140, 141, 6.0)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for j, k in enumerate(snaps):
        axv = fig.add_subplot(gs[0, 1 + j] if j < 3 else gs[1, 3])
        lum, pk, hk = views[k]
        prof = np.array([lum[band & (az >= a) & (az < b)].mean()
                         if (band & (az >= a) & (az < b)).any() else np.nan
                         for a, b in zip(edges[:-1], edges[1:])])
        axv.fill_between(ctr, prof, color="#34495e", alpha=0.85)
        axv.axvline(0, color="k", lw=0.7, ls=":")
        tv = world.target - pk
        tb = (math.degrees(math.atan2(tv[1], tv[0])) - math.degrees(hk)
              + 180) % 360 - 180
        if abs(tb) < 140:
            axv.axvline(tb, color="#f39c12", lw=1.8)
        axv.set_title("t=%d  %s" % (k, "target visible"
                                    if rec["seen"][k] else "target not seen"),
                      fontsize=8.5)
        axv.set_xlabel("bearing, deg (+ = left)", fontsize=7)
        axv.tick_params(labelsize=7)
        if j == 0:
            axv.set_ylabel("lamina luminance", fontsize=7)

    # --- the traces --------------------------------------------------
    t = np.arange(len(rec["cmd"])) * DT
    ax1 = fig.add_subplot(gs[1, 1:3])
    ax1.plot(t, rec["bearing"], color="#f39c12", lw=1.4)
    ax1.axhline(0, color="k", lw=0.6, ls=":")
    ax1.fill_between(t, -180, 180, where=rec["seen"], color="#f39c12",
                     alpha=0.13, step="mid")
    ax1.set_ylabel("target bearing, deg", fontsize=8)
    ax1.set_ylim(-60, 60)
    ax1.tick_params(labelsize=7)
    ax1.set_title("shaded = target in view; it is found at t=%.1f s"
                  % (first_seen * DT), fontsize=9)

    ax2 = fig.add_subplot(gs[2, 1:3])
    ax2.plot(t, rec["cmd"], color="#c0392b", lw=0.9)
    ax2.axhline(0, color="k", lw=0.6, ls=":")
    for lv in (math.radians(90), -math.radians(90)):
        ax2.axhline(lv, color="0.5", lw=0.7, ls="--")
    ax2.fill_between(t, ax2.get_ylim()[0], ax2.get_ylim()[1],
                     where=rec["seen"], color="#f39c12", alpha=0.13,
                     step="mid")
    ax2.set_ylabel("yaw command, rad/s", fontsize=8)
    ax2.set_xlabel("time, s", fontsize=8)
    ax2.tick_params(labelsize=7)

    ax3 = fig.add_subplot(gs[2, 3])
    ax3.plot(t, rec["dist"], color="#2c3e50", lw=1.5)
    ax3.axhline(world.reach_radius, color="#27ae60", ls="--", lw=1)
    ax3.fill_between(t, 0, max(rec["dist"]), where=rec["seen"],
                     color="#f39c12", alpha=0.13, step="mid")
    ax3.set_ylabel("distance to target, m", fontsize=8)
    ax3.set_xlabel("time, s", fontsize=8)
    ax3.tick_params(labelsize=7)

    fig.suptitle("One flight: the frozen 66k connectome finds a target it "
                 "could not see at the start", fontsize=12)
    out = REPO / ("results/cx_demo_map%d.png" % MAP)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
