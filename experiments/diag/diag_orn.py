"""How strong must the odour be before it steers?  Static, twin on.

A beacon 60 deg left at 10 m in an empty world; vision and threat cells
silenced, so only ORN drives.  Strength = the scent mass the input uses
(orn_gain x strength split over the ORNs of a side).  For scale: the mass
of a beacon at 25 m is ~0.0007, at 10 m ~0.003, at 5 m ~0.013, at 2 m ~0.06.
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
from environment.target_world import TargetWorld
from sensors.vp_input import VPInput

net, ann, sub, info = build_vp_subnet()
inp = VPInput(net, ann, info)
inp.twin = True
net.freeze_params(5.0)
p0 = np.array(C.START)
gain, cc, zero = C.calibrate(net, inp, info, None, p0)
a = math.radians(60)
wld = TargetWorld(target=np.array([p0[0] + 10 * math.cos(a), p0[1] + 10 * math.sin(a), 2.0]),
                  obstacles=[])
vis = inp.vp_rows


def u(strength):
    inp.fixed_strength = strength
    v = net.init_state(2)
    inp.reset()
    for _ in range(3):
        d, tr = inp.drive(wld, p0, 0.0)
        d[vis] = 0.0
        inp.d_mirror[vis] = 0.0
        v, r = C.step_circuit(net, v, C.twin(inp, d))
    return math.degrees(gain * (C.steer(inp, r) - zero))


print("beacon 60 deg left, odour only; u in deg/s (+ = toward it)")
for s in (0.0007, 0.003, 0.013, 0.03, 0.06, 0.1, 0.2, 0.5, 1.0, 2.0):
    print("  strength %-7g u = %+7.1f" % (s, u(s)))
