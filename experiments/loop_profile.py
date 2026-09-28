"""Where do the 76 seconds per biological second actually go?

One brain step is: one render, one 166,700-neuron update, then fifty times
(build a controller observation, step the controller, step MuJoCo).  Those have
wildly different call counts -- 200/s, 200/s, and 10,000/s -- so a component
that looks cheap per call can still dominate.

Timed separately rather than reasoned about, because the last time this project
guessed at a bottleneck it was wrong: the 20x slowdown blamed on the sparse
matmul turned out to be a Python string map, and the 1,590 s/s figure for the
joint-level loop turned out to be `joint_commands` scanning 166,700 cell-type
strings every step, not the brain at all.

Usage:
    python experiments/loop_profile.py --steps 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "reference"))

import numpy as np
import torch

from flygym.vision.retina import Retina
from flygym_demo.complex_terrain import (
    HybridControllerObservation, PreprogrammedSteps, apply_locomotion_action)

import sensors.flywire_eye as eye
import sensors.flygym_bridge as bridge
from core.flywire_rate import FlyWireRate, activity
from core.malecns import load_malecns
from sim.flygym_walk_loop import (
    BRAIN_DT_MS, STEPS_PER_BRAIN_STEP, ConnectomeWalkLoop,
    make_sim_and_controller)


class Timer:
    def __init__(self):
        self.t = {}
        self.n = {}
        self.all = {}

    def add(self, key, dt):
        self.t[key] = self.t.get(key, 0.0) + dt
        self.n[key] = self.n.get(key, 0) + 1
        self.all.setdefault(key, []).append(dt)

    def report(self, total, bio_s):
        import numpy as _np
        print("%-26s %7s %8s %10s %10s %10s %6s"
              % ("component", "calls", "total s", "mean ms", "median ms",
                 "max ms", "%"))
        for k in sorted(self.t, key=lambda x: -self.t[x]):
            a = _np.array(self.all[k])
            print("%-26s %7d %8.2f %10.2f %10.2f %10.2f %5.1f%%"
                  % (k, self.n[k], self.t[k], 1e3 * a.mean(),
                     1e3 * _np.median(a), 1e3 * a.max(),
                     100 * self.t[k] / total))
        acc = sum(self.t.values())
        print("%-26s %9s %9.2f %12s %6.1f%%"
              % ("(unaccounted)", "", total - acc, "",
                 100 * (total - acc) / total))
        print("\ntotal %.1f s for %.3f biological s -> %.0f s per biological s"
              % (total, bio_s, total / bio_s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
    lat = eye.load_malecns_eye(ann)
    omm, _ = bridge.join(lat, Retina())
    print("load %.1f s; %d neurons, %d edges"
          % (time.perf_counter() - t0, net.n, net.n_edges), flush=True)

    sim, fly, ctrl = make_sim_and_controller(PreprogrammedSteps())
    loop = ConnectomeWalkLoop(net, ann, lat, omm, sim, fly, ctrl,
                              descending_gain=150.0)

    # warm up: the FIRST get_ommatidia_readouts lazily builds mj.Renderer and
    # a GL context, which costs seconds.  Averaging that into ten calls made an
    # earlier version of this script report 824 ms/call for vision when the
    # steady-state cost is ~33 ms -- a single outlier hidden by a mean.
    t = time.perf_counter()
    sim.get_ommatidia_readouts(fly.name)
    print("first render (renderer + GL context construction): %.2f s"
          % (time.perf_counter() - t), flush=True)

    tm = Timer()
    t_start = time.perf_counter()
    for _ in range(args.steps):
        t = time.perf_counter()
        readouts = sim.get_ommatidia_readouts(fly.name)
        tm.add("vision: render+readout", time.perf_counter() - t)

        t = time.perf_counter()
        per_eye = readouts.sum(axis=-1)
        lum = np.where(loop.eye_is_left, per_eye[0][loop.omm],
                       per_eye[1][loop.omm])
        drive = torch.zeros(net.n, 1)
        sel = loop.valid
        drive[loop.rows[sel], 0] = torch.from_numpy(
            (lum[sel] * loop.drive_gain).astype(np.float32))
        tm.add("vision: scatter to drive", time.perf_counter() - t)

        t = time.perf_counter()
        with torch.no_grad():
            loop.v = net.step(loop.v, drive, BRAIN_DT_MS)
        tm.add("brain: 166k sparse step", time.perf_counter() - t)

        t = time.perf_counter()
        rates = activity(loop.v).numpy().ravel()
        delta = loop.descending_signal(rates)
        tm.add("readout: descending", time.perf_counter() - t)

        for _ in range(STEPS_PER_BRAIN_STEP):
            t = time.perf_counter()
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            tm.add("body: build observation", time.perf_counter() - t)

            t = time.perf_counter()
            action = ctrl.step(delta, obs)
            tm.add("body: CPG controller", time.perf_counter() - t)

            t = time.perf_counter()
            apply_locomotion_action(sim, fly.name, action)
            tm.add("body: apply action", time.perf_counter() - t)

            t = time.perf_counter()
            sim.step()
            tm.add("body: mujoco step", time.perf_counter() - t)

    total = time.perf_counter() - t_start
    tm.report(total, args.steps * BRAIN_DT_MS / 1000.0)

    # the component that dominates should be obvious, not a coin flip
    top = max(tm.t, key=lambda k: tm.t[k])
    assert tm.t[top] > 0, top
    print("\ndominant component: %s (%.0f%%)"
          % (top, 100 * tm.t[top] / total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
