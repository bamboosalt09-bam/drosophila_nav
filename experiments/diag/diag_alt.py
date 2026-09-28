"""Room 6, 12 beams, v9: alternation fires -- does it change the turn?"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np

import cx_vp_room as C
import sim.drone_layer as DL
from core.vp_subnet import build_vp_subnet
from environment.room_file import ensure as room
from sensors.rangefinder import LAYOUTS, Rangefinder
from sensors.vp_input import VPInput

net, ann, sub, info = build_vp_subnet()
net.freeze_params(5.0)
inp = VPInput(net, ann, info)
w, h, reach = room(6)
p0 = np.array(C.START)
gain, cc, zero = C.calibrate(net, inp, info, w, p0)
print("PFL3 calibration (drive, du):", C.calibrate_alt(net, inp, gain, zero, p0))
inp.cam.rangefinder = Rangefinder(inp.cam.a, LAYOUTS[12])
base, peak = C.calibrate_threat(net, inp, p0)

log = []
_f = DL.follow
def spy(u, pts, rear, mem=None, pose=None):
    out = _f(u, pts, rear, mem, pose)
    log.append((u, out[0], out[1], inp.alt_left > 0, inp.alt_dir,
                sum(inp._yaw_hist)))
    return out
C.follow = spy
res = C.fly(net, inp, info, w, h, gain, zero, base, peak)
print("12 beams:", {k: res[k] for k in ("found", "alternations", "dist")})
u = np.degrees(np.array([x[0] for x in log]))
yaw = np.degrees(np.array([x[1] for x in log]))
alt = np.array([x[3] for x in log])
adir = np.array([x[4] for x in log])
starts = [i for i in range(1, len(alt)) if alt[i] and not alt[i - 1]]
for s in starts:
    pre = slice(max(s - 20, 0), s)
    dur = slice(s, s + 30)
    post = slice(s + 30, s + 60)
    print("\nalternation at t=%.1f s, pushing %s (the loop was %s)"
          % (s * 0.1, "RIGHT" if adir[s] < 0 else "LEFT",
             "LEFT" if adir[s] < 0 else "RIGHT"))
    for nm, sl in (("2 s before", pre), ("during 3 s", dur), ("3 s after", post)):
        print("   %-11s intended turn mean %+6.0f deg/s   actual yaw mean %+6.0f deg/s"
              % (nm, u[sl].mean(), yaw[sl].mean()))
