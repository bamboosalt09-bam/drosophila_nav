"""Can the optomotor route steer?  HS/H2 cells driven on one side, static.

HSE, HSN, HSS and H2 (1 per side each) are the lobula plate's horizontal
wide-field motion cells -- the fly's own optomotor channel to the central
brain.  They are in the subnet but only get pixel contrast today.  Drive
one side's four cells alone on the usual baseline and read the intended
turn (deg/s, + = left) through the runner's calibration.
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
net.freeze_params(5.0)
p0 = np.array(C.START)
C.wire_cue(net, inp, info, p0)
gain, cc, zero = C.calibrate(net, inp, info, None, p0)
ct = ann["cell_type"].astype(str).to_numpy()
side = ann["side"].astype(str).to_numpy()
hs = np.isin(ct, ["HSE", "HSN", "HSS", "H2"])
hs_l, hs_r = np.flatnonzero(hs & (side == "left")), np.flatnonzero(hs & (side == "right"))
empty = TargetWorld(target=np.array([300.0, 0.0, 2.0]), obstacles=[])
inp.reset()
inp.drive(empty, p0, 0.0)
base, _ = inp.drive(empty, p0, 0.0)


def u(rows, val):
    d = base.clone()
    d[rows] += val
    v = net.init_state(1)
    for _ in range(3):
        v, r = C.step_circuit(net, v, d)
    return math.degrees(gain * (C.steer(inp, r) - zero))


u0 = u([], 0.0)
print("HS/H2 cells: left %s, right %s" % (list(ct[hs_l]), list(ct[hs_r])))
print("drive per cell   left side alone   right side alone   (deg/s, + = left)")
for val in (0.05, 0.1, 0.2, 0.5, 1.0):
    print("  %-6g %16.1f %18.1f" % (val, u(hs_l, val) - u0, u(hs_r, val) - u0))
