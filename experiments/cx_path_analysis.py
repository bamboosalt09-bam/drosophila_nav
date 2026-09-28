"""What do the paths actually look like, and why is the connectome's shorter?

Everything so far has been aggregates: reach, collisions, mean final
distance.  Those say the connectome arrives more often and ends closer, but
"경로의 효율성 분석" is the deliverable and an aggregate cannot say WHY one
path is better than another.

What the aggregates leave unexplained:

  - the connectome ends 0.57 m closer yet sits at the yaw ACCELERATION limit
    26.9% of the time against 1.1% for the P reference.  A controller that
    saturates a quarter of the time and still wins is doing something the
    mean does not show.
  - clipping a quarter of its command costs it nothing (reach 0.875 ->
    0.879), so whatever is being clipped carries no information.
  - tighten the acceleration to 1 rad/s^2 and the advantage inverts.

The obvious hypothesis, and the one this file is written to test: the
connectome rides a high-frequency dither on top of a correct low-frequency
command.  The dither is what saturates the body; the body filters it out for
free; and squeezing the acceleration limit far enough finally eats into the
low-frequency part too.

So decompose the command in frequency, and measure the path properties an
aggregate hides: detour against the shortest legal path, total heading swept,
how early the bearing error collapses, and how close each arm shaves past an
obstacle.

ponytail: a 1 Hz split, not a designed filter bank.  Control runs at 10 Hz so
Nyquist is 5 Hz; 1 Hz separates "steering" from "chatter" well enough to tell
the hypothesis apart from its negation, and a real filter bank is only worth
building if the split turns out to be where the story lives.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from sensors.attraction_cue import AttractionCueSensor
from sim.episode import DT

from cx_validate import tasks_for
from cx_vision import VisionSub, build
from cx_vision_loop import oracle_cmd

GAIN = -3e6
SPLIT_HZ = 1.0


def run(cmd_fn, world, heading, sensor, max_steps=120, yaw_params=None):
    """One episode, keeping the whole trace rather than a summary of it."""
    from environment.target_world import Agent
    from sim.episode import V_CRUISE

    agent = Agent(world=world, start=(0.0, 0.0, 2.0), heading=heading,
                  yaw_params=yaw_params)
    cmds, psis, bearings, clear = [], [], [], []
    for _ in range(max_steps):
        cue = sensor.sense(agent.p, agent.heading, world.target,
                           world.obstacles)
        u = float(cmd_fn(cue, agent.heading, agent))
        cmds.append(u)
        psis.append(agent.heading)
        bearings.append(cue.bearing_deg if cue.valid else np.nan)
        if world.obstacles:
            clear.append(min(o.distance_to(agent.p) for o in world.obstacles))
        agent.step(u, V_CRUISE, 0.0, DT)
        if agent.collided or world.reached(agent.p):
            break
    return {"path": agent.path_array(), "cmd": np.array(cmds),
            "psi": np.array(psis), "bearing": np.array(bearings),
            "clearance": np.array(clear) if clear else np.array([np.nan]),
            "reached": bool(world.reached(agent.p)) and not agent.collided,
            "collided": bool(agent.collided), "world": world}


def band_power(cmd, dt=DT, split=SPLIT_HZ):
    """Fraction of command variance above `split` Hz."""
    x = cmd - cmd.mean()
    if len(x) < 8 or x.std() == 0:
        return float("nan")
    f = np.fft.rfftfreq(len(x), dt)
    P = np.abs(np.fft.rfft(x)) ** 2
    tot = P.sum()
    return float(P[f > split].sum() / tot) if tot > 0 else float("nan")


def metrics(r):
    p, w = r["path"], r["world"]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    length = float(seg.sum())
    floor = float(np.linalg.norm(w.target - p[0]) - w.reach_radius)
    dpsi = np.abs(np.diff(np.unwrap(r["psi"])))
    b = np.abs(r["bearing"])
    ok = ~np.isnan(b)
    # how many steps until the bearing error first drops under 10 deg and
    # stays there: "does it commit early or wander"
    settle = np.nan
    if ok.any():
        under = np.where(ok & (b < 10.0))[0]
        for i in under:
            tail = b[i:][ok[i:]]
            if len(tail) and np.all(tail < 20.0):
                settle = int(i)
                break
    return {
        "reached": r["reached"], "collided": r["collided"],
        "steps": len(r["cmd"]),
        "path_len": length,
        # 1.0 is the shortest path that counts as arriving; NaN if it never did
        "detour": length / floor if (r["reached"] and floor > 0) else np.nan,
        "turned_rad": float(dpsi.sum()),
        "cmd_rms": float(np.sqrt(np.mean(r["cmd"] ** 2))),
        # RMS of the step-to-step CHANGE: this is what an acceleration limit
        # actually fights, and it is the number the aggregates never showed
        "cmd_jerk": float(np.sqrt(np.mean(np.diff(r["cmd"]) ** 2)))
        if len(r["cmd"]) > 1 else np.nan,
        "hf_fraction": band_power(r["cmd"]),
        "settle_step": settle,
        "min_clearance": float(np.nanmin(r["clearance"])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--plot", type=int, default=6)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    net, ann, sub, rho, n_lam, n_cx = build()
    vs = VisionSub(net, ann)
    print("%d neurons  (%.0f s)\n" % (net.n, time.perf_counter() - t0),
          flush=True)
    sensor = AttractionCueSensor()

    def conn(w):
        vs.reset()
        return lambda cue, h, ag: GAIN * vs.turn(w, ag.p, h)

    def pblind(w):
        return lambda cue, h, ag: (2.0 * math.radians(cue.bearing_deg)
                                   if cue.valid else 0.0)

    def porc(w):
        f = oracle_cmd([None], w)
        def g(cue, h, ag):
            f.__globals__  # keep the closure honest about what it reads
            u = 2.0 * math.radians(cue.bearing_deg) if cue.valid else 0.0
            for o in w.obstacles:
                d = np.array([o.x, o.y]) - ag.p[:2]
                rr = float(np.linalg.norm(d))
                if rr > 4.0 or rr < 1e-6:
                    continue
                rel = (math.atan2(d[1], d[0]) - h + math.pi) % (2*math.pi) - math.pi
                if abs(rel) > math.radians(70):
                    continue
                u -= math.copysign(1.0, rel) * 3.0 * (4.0 - rr) / 4.0
            return u
        return g

    arms = [("connectome", conn), ("P blind", pblind), ("P + oracle", porc)]
    rows, keep = [], {}
    for seed in range(2, 2 + args.seeds):
        for i, (w, h0) in enumerate(tasks_for(seed, args.episodes)):
            for name, fac in arms:
                r = run(fac(w), w, h0, sensor)
                m = metrics(r)
                m.update({"arm": name, "seed": seed, "episode": i})
                rows.append(m)
                if seed == 2 and i < args.plot:
                    keep.setdefault(i, {})[name] = r
        print("seed %d done" % seed, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/cx_path_analysis.csv", index=False)

    print("\n--- path shape " + "-" * 56)
    print("%-12s %8s %8s %9s %9s %8s %8s %9s"
          % ("arm", "reached", "detour", "path_len", "turned", "settle",
             "clear", "collided"))
    for name, _ in arms:
        s = df[df.arm == name]
        print("%-12s %8.3f %8.3f %9.2f %9.2f %8.1f %8.2f %9.3f"
              % (name, s.reached.mean(), s.detour.mean(), s.path_len.mean(),
                 s.turned_rad.mean(), s.settle_step.mean(),
                 s.min_clearance.mean(), s.collided.mean()))

    print("\n--- what the command looks like " + "-" * 39)
    print("%-12s %10s %10s %14s"
          % ("arm", "cmd_rms", "cmd_jerk", "power > 1 Hz"))
    for name, _ in arms:
        s = df[df.arm == name]
        print("%-12s %10.3f %10.3f %13.1f%%"
              % (name, s.cmd_rms.mean(), s.cmd_jerk.mean(),
                 100 * s.hf_fraction.mean()))

    # ---- picture of the actual trajectories -------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(keep)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    col = {"connectome": "#c0392b", "P blind": "#2980b9",
           "P + oracle": "#7f8c8d"}
    for ax, (i, d) in zip(np.ravel(axes), sorted(keep.items())):
        w = next(iter(d.values()))["world"]
        for o in w.obstacles:
            ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="0.85"))
        ax.add_patch(plt.Circle(tuple(w.target[:2]), w.reach_radius,
                                fill=False, ls="--", color="0.4"))
        ax.plot(*w.target[:2], marker="*", ms=14, color="#f39c12", zorder=5)
        for name, r in d.items():
            p = r["path"]
            ax.plot(p[:, 0], p[:, 1], color=col[name], lw=1.6, label=name,
                    alpha=0.9)
            ax.plot(p[-1, 0], p[-1, 1], "o", ms=4, color=col[name])
        ax.set_aspect("equal")
        ax.set_title("episode %d" % i, fontsize=9)
        ax.tick_params(labelsize=7)
    np.ravel(axes)[0].legend(fontsize=7, loc="best")
    fig.suptitle("Paths: connectome vs references (seed 2)", fontsize=11)
    fig.tight_layout()
    out = REPO / "results/cx_paths.png"
    fig.savefig(out, dpi=130)
    print("\nwrote %s and results/cx_path_analysis.csv" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
