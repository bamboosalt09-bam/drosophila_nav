"""Does the circuit now see MOTION?  Two checks, no flight, ~30 s.

Written before running, with the expected numbers stated, so the run
confirms a prediction instead of exploring.

  1. CALIBRATION UNCHANGED.  The standing sweep resets between headings, so
     each heading is one frame with no predecessor, the interpolated
     sequence is constant, and the readout must be bit-identical to the
     held-column version.  Expected room 0: gain -7.1e+03, sweep -0.72.
     Anything else means the sequence changed what a STILL frame does,
     which would be a bug, not a feature.

  2. APPROACH != RECEDE.  Same pose, same distance, but arrived at by
     closing versus opening.  The still-photograph version cannot tell
     these apart: identical image -> identical steady state -> identical
     readout, difference exactly 0.  If the difference is now non-zero,
     the time axis is reaching the circuit.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np

from core.vp_subnet import build_vp_subnet
from environment.target_world import Obstacle, TargetWorld, room_world
from sensors.vp_input import VPInput
from cx_vp_room import START, calibrate, step_circuit


def readout(net, info, r):
    return float(r[info["dn_r"]].mean() - r[info["dn_l"]].mean())


def sweep_through(net, inp, info, world, poses, heading=0.0):
    """Feed a sequence of positions, return the final readout."""
    v = net.init_state(1)
    inp.reset()
    for p in poses:
        d, _ = inp.drive(world, np.array(p), heading)
        v, r = step_circuit(net, v, d)
    return readout(net, info, r)


def main() -> int:
    net, ann, sub, info = build_vp_subnet()
    inp = VPInput(net, ann, info)

    w, h, _ = room_world(seed=0)
    gain, cc, zero = calibrate(net, inp, info, w, np.array(START))
    print("1. calibration on room 0")
    print("   expected  gain -7.1e+03   sweep -0.72")
    print("   got       gain %+.1e   sweep %+.2f" % (gain, cc))
    print("   %s\n" % ("UNCHANGED, as predicted" if abs(cc + 0.72) < 0.02
                       else "CHANGED -- the sequence altered a still frame"))

    # a wall dead ahead, approached from 6 m or backed away from 2 m,
    # both ending at 4 m
    FAR = np.array([300.0, 0.0, 2.0])
    wall = TargetWorld(target=FAR,
                       obstacles=[Obstacle(x=6.0, y=0.0, radius=1.5,
                                           z=0.0, height=8.0)])
    closing = [(x, 0.0, 2.0) for x in (0.0, 0.5, 1.0, 1.5, 2.0)]
    opening = [(x, 0.0, 2.0) for x in (4.0, 3.5, 3.0, 2.5, 2.0)]

    a = sweep_through(net, inp, info, wall, closing)
    b = sweep_through(net, inp, info, wall, opening)
    print("2. same final pose (2.0 m along x), different history")
    print("   approaching  %+.6e" % a)
    print("   receding     %+.6e" % b)
    print("   difference   %+.6e" % (a - b))
    print("   %s" % ("MOTION REACHES THE CIRCUIT"
                     if abs(a - b) > 1e-12 else
                     "IDENTICAL -- still a photograph"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
