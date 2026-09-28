"""(1) Does the drone hang around beacons it already took?  From saved paths.
(2) Why does the connectome stall in corners?  Re-fly room 5, 3 beams, log
the drone layer's inputs and outputs every step."""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np
import pandas as pd

from environment.room_file import load

# ---- (1) lingering at taken beacons ---------------------------------------
print("(1) steps spent within 3 m of a beacon AFTER taking it (120 s = 1200)")
tot = {}
for rm in (0, 5, 6, 7):
    w, h, reach = load(rm)
    P = np.load(REPO / "results" / ("paths_room%d.npz" % rm))
    tg = np.array(w.all_targets())[:, :2]
    for key in P.files:
        path = P[key]
        after = 0
        for t in tg:
            d = np.linalg.norm(path - t, axis=1)
            hit = np.flatnonzero(d < w.reach_radius)
            if len(hit):
                after += int((d[hit[0] + 20:] < 3.0).sum())   # skip 2 s after
        arm = key.split("_", 1)[1]
        tot.setdefault(arm, []).append((after, len(path)))
for arm, v in tot.items():
    a = np.array(v)
    print("  %-10s %5d of %6d steps (%.1f%%) near an already-taken beacon"
          % (arm, a[:, 0].sum(), a[:, 1].sum(), 100 * a[:, 0].sum() / a[:, 1].sum()))

# ---- (2) the corner trap ---------------------------------------------------
import cx_vp_room as C
from core.vp_subnet import build_vp_subnet
from environment.room_file import ensure as room
from sensors.rangefinder import LAYOUTS, Rangefinder
from sensors.vp_input import VPInput
import sim.drone_layer as DL

log = []
_follow = DL.follow
def spy(u, tl, tr, front, rear):
    out = _follow(u, tl, tr, front, rear)
    log.append((u, tl, tr, front, rear, out[0], out[1]))
    return out
C.follow = spy

net, ann, sub, info = build_vp_subnet()
net.freeze_params(5.0)
inp = VPInput(net, ann, info)
w, h, reach = room(5)
p0 = np.array(C.START)
gain, cc, zero = C.calibrate(net, inp, info, w, p0)
inp.cam.rangefinder = Rangefinder(inp.cam.a, LAYOUTS[3])
base, peak = C.calibrate_threat(net, inp, p0)
res = C.fly(net, inp, info, w, h, gain, zero, base, peak)
print("\n(2) room 5, 3 beams:", {k: res[k] for k in ("found", "stopped", "dist")})
L = pd.DataFrame(log, columns=["u", "thr_l", "thr_r", "front", "rear",
                               "r_cmd", "v_cmd"])
L["u_deg"] = np.degrees(L.u)
L["r_deg"] = np.degrees(L.r_cmd)
st = L[L.v_cmd <= 0]
mv = L[L.v_cmd > 0]
print("  stopped steps %d of %d" % (len(st), len(L)))
print("  while STOPPED: |intended turn| median %.0f deg/s, |actual yaw| "
      "median %.1f deg/s  (ratio %.2f)"
      % (st.u_deg.abs().median(), st.r_deg.abs().median(),
         st.r_deg.abs().median() / max(st.u_deg.abs().median(), 1e-9)))
print("  while MOVING : |intended turn| median %.0f deg/s, |actual yaw| "
      "median %.1f deg/s" % (mv.u_deg.abs().median(), mv.r_deg.abs().median()))
print("  while STOPPED: front median %.2f m, rear %s, threat L %.2f R %.2f"
      % (st.front.median(), "all 0 (no rear beam)" if (st.rear == 0).all()
         else "median %.2f" % st.rear.median(), st.thr_l.median(), st.thr_r.median()))
sgn = np.sign(st.u.values)
flips = int((sgn[1:] != sgn[:-1]).sum())
print("  intended-turn sign flips while stopped: %d over %d steps" % (flips, len(st)))
# heading swept while stopped, in one long stop
runs, cur = [], []
for i, s in enumerate((L.v_cmd <= 0).values):
    if s: cur.append(i)
    elif cur: runs.append(cur); cur = []
if cur: runs.append(cur)
lr = max(runs, key=len) if runs else []
if lr:
    yaw = np.degrees(np.sum(L.r_cmd.values[lr]) * 0.1)
    print("  longest stop: %d steps (%.1f s), net heading change %.0f deg, "
          "intended turn over it: mean %+.0f deg/s"
          % (len(lr), len(lr) * 0.1, yaw, L.u_deg.values[lr].mean()))
