"""Does the connectome signal a turn as a wall comes closer?  Open loop.

The drone flies straight at 2 m/s toward a wall (a row of pillars) from 7 m
away, heading fixed -- no drone layer, nothing steers -- and the intended
turn u is logged every cycle.  The wall is square to the path (0 deg) or
rotated so that its LEFT end is nearer (+15, +30) or its right end (-15);
"away" is then a right or a left turn.  12 beams.  Run with the mirror
twin on and off.  u > 0 is a LEFT turn.
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np

import cx_vp_room as C
from core.vp_subnet import build_vp_subnet
from environment.target_world import Obstacle, TargetWorld
from sensors.rangefinder import LAYOUTS, Rangefinder
from sensors.vp_input import VPInput

net, ann, sub, info = build_vp_subnet()
inp = VPInput(net, ann, info)
net.freeze_params(5.0)
p0 = np.array(C.START)
far = np.array([300.0, 0.0, 2.0])
front = list(LAYOUTS[12]).index(0)


def wall(dist, ang_deg):
    a = math.radians(ang_deg)
    return [Obstacle(dist - s * math.sin(a), s * math.cos(a), 0.45, height=6.0)
            for s in np.arange(-8.0, 8.01, 0.8)]


for twin in (True, False):
    inp.twin = twin
    C.wire_cue(net, inp, info, p0)
    gain, cc, zero = C.calibrate(net, inp, info, None, p0)
    inp.cam.rangefinder = Rangefinder(inp.cam.a, LAYOUTS[12])
    inp.speed = 2.0
    print("\n=== mirror twin %s" % ("ON" if twin else "OFF"))
    for ang in (0, 15, 30, -15):
        v = net.init_state(2 if twin else 1)
        inp.reset()
        x, rows = 0.0, []
        for k in range(40):
            wld = TargetWorld(target=far, obstacles=wall(7.0, ang))
            d, tr = inp.drive(wld, np.array([x, 0.0, 2.0]), 0.0)
            v, r = C.step_circuit(net, v, C.twin(inp, d))
            rows.append((tr["beams"][front],
                         math.degrees(gain * (C.steer(inp, r) - zero))))
            x += 0.2
            if rows[-1][0] < 0.8:
                break
        away = {0: "either", 15: "RIGHT(-)", 30: "RIGHT(-)", -15: "LEFT(+)"}[ang]
        print("wall %+3d deg, away = %-8s " % (ang, away)
              + "  ".join("%.1fm:%+.0f" % (a, u) for a, u in rows[2::3]))
