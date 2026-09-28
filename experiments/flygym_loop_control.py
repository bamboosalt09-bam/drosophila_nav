"""Does the connectome actually move the body, or is that gravity?

The closed loop reports 0.381 mm of displacement in 100 ms.  A fly dropped
from the spawn height settles by about that much on its own, so the number
alone says nothing.  This is the paired comparison: identical body, identical
spawn, identical physics, the ONLY difference being whether the motor readout
reaches the actuators (`actuator_gain`).

Three conditions:
    blocked     gain 0     -- neutral pose held, the body's own dynamics
    connected   gain 0.3   -- the readout as a deviation from neutral
    amplified   gain 30    -- the same readout, 100x

Amplified is the informative one.  Our readout is ~1e-3 and the joint ranges
are radians, so at gain 0.3 the brain is asking for 5e-4 rad: even a perfect
command would be invisible.  If amplified separates from blocked and
connected does not, the loop works and the gain is simply mis-scaled, which
is a calibration problem rather than a wiring one.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym.vision.retina import Retina

import sensors.flywire_eye as eye
import sensors.flygym_bridge as bridge
from core.flywire_rate import FlyWireRate
from core.malecns import load_malecns
from sim.flygym_loop import ConnectomeFlyLoop, LoopTrace, build_fly

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--steps", type=int, default=20)
_ap.add_argument("--gains", type=float, nargs="+",
                 default=[0.0, 0.3, 30.0])
_ARGS = _ap.parse_args()
N_STEPS = _ARGS.steps
CONDITIONS = {("gain%g" % g): g for g in _ARGS.gains}


def run(net, ann, lat, omm, actuator_gain: float) -> LoopTrace:
    """One loop from a fresh body, so nothing carries over between gains."""
    world = FlatGroundWorld()
    fly = build_fly()
    world.add_fly(fly, spawn_position=(0.0, 0.0, 0.5),
                  spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    loop = ConnectomeFlyLoop(net, ann, lat, omm, sim, fly,
                             actuator_gain=actuator_gain)
    trace = LoopTrace()
    for _ in range(N_STEPS):
        loop.step(trace)
    return trace


def main() -> None:
    t0 = time.perf_counter()
    ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
    lat = eye.load_malecns_eye(ann)
    omm, _ = bridge.join(lat, Retina())
    print("brain ready (%.1f s)" % (time.perf_counter() - t0), flush=True)

    traces = {}
    for name, gain in CONDITIONS.items():
        t0 = time.perf_counter()
        traces[name] = run(net, ann, lat, omm, gain)
        p = traces[name].as_arrays()["position"]
        print("%-10s gain %5.1f  moved %.4f mm  final z %.4f  (%.0f s)"
              % (name, gain, np.linalg.norm(p[-1] - p[0]), p[-1][2],
                 time.perf_counter() - t0), flush=True)

    names = list(CONDITIONS)
    base = traces[names[0]].as_arrays()["position"]
    print()
    for name in names[1:]:
        p = traces[name].as_arrays()["position"]
        d = np.linalg.norm(p - base, axis=1)
        print("%-10s vs blocked: final separation %.5f mm, max %.5f mm"
              % (name, d[-1], d.max()))

    # the loop is real only if the strongest condition leaves the body
    # somewhere the blocked one does not
    amp = traces[names[-1]].as_arrays()["position"]
    sep = np.linalg.norm(amp[-1] - base[-1])
    assert sep > 1e-6, "the motor readout does not reach the body at all"
    print("\nthe readout reaches the body: %.5f mm of separation" % sep)


if __name__ == "__main__":
    main()
