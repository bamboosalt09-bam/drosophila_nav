"""Room 6: why does 24 beams fly a messier path than 12?"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np
import pandas as pd

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

paths = []
_row = C._row
def row(ag, *a):
    paths.append(ag.path_array())
    return _row(ag, *a)
C._row = row

logs = {}
for n in (12, 24):
    rf = Rangefinder(inp.cam.a, LAYOUTS[n])
    inp.cam.rangefinder = rf
    covered = int((rf._beam_of_col[inp.thr_col] >= 0).sum())
    base, peak = C.calibrate_threat(net, inp, p0)
    log = []
    _f = DL.follow
    def spy(u, tl, tr, pts, rear, mem=None, _f=_f, log=log):
        k = u / DL.V_CRUISE - DL.K_THREAT * (tl - tr)
        d = DL.hit_distance(pts, k)
        out = _f(u, tl, tr, pts, rear, mem)
        log.append((u, tl, tr, d, "dir" in (mem or {}), out[0], out[1]))
        return out
    C.follow = spy
    res = C.fly(net, inp, info, w, h, gain, zero, base, peak)
    C.follow = _f
    L = pd.DataFrame(log, columns=["u", "thr_l", "thr_r", "d_path", "sacc",
                                   "r", "v"])
    logs[n] = (L, paths[-1])
    print("%2d beams: threat cells covered %d of %d | found %d, dist %.0f m"
          % (n, covered, len(inp.thr_col), res["found"], res["dist"]))
    print("   |intent| median %.0f deg/s, |thr_l - thr_r| mean %.3f, "
          "max(thr) mean %.3f, saccade steps %d, v median %.2f"
          % (np.degrees(L.u.abs().median()), (L.thr_l - L.thr_r).abs().mean(),
             np.maximum(L.thr_l, L.thr_r).mean(), int(L.sacc.sum()),
             L.v.median()))

a, b = logs[12][1], logs[24][1]
m = min(len(a), len(b))
gap = np.linalg.norm(a[:m] - b[:m], axis=1)
k0 = int(np.argmax(gap > 0.5))
print("\npaths first split by >0.5 m at step %d (t = %.1f s), position %s"
      % (k0, k0 * 0.1, np.round(b[k0], 1)))
for n in (12, 24):
    L = logs[n][0]
    s = L.iloc[max(k0 - 15, 0):k0 + 5]
    print("  %2d beams, steps %d..%d: intent %s deg/s" % (
        n, max(k0 - 15, 0), k0 + 4,
        np.round(np.degrees(s.u.values[::4]), 0).astype(int).tolist()))
    print("            thr_l %s" % np.round(s.thr_l.values[::4], 2).tolist())
    print("            thr_r %s" % np.round(s.thr_r.values[::4], 2).tolist())
    print("            d_path %s  saccade %s" % (
        np.round(np.minimum(s.d_path.values[::4], 9.9), 1).tolist(),
        s.sacc.values[::4].astype(int).tolist()))
