"""Room 6, v11, 12 beams: the odour cue along the flight, approach vs loop.

Re-renders the saved path (no connectome) and records, per step, which
beacon the odour points at, its mass and its distance.  The question for
ORN adaptation: what separates "pulled toward a beacon it is reaching" from
"pulled toward one it cannot reach" (the corridor loop, t > ~80 s)?
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from environment.room_file import load
from environment.target_world import TargetWorld
from sensors.fisheye import FisheyeCamera

VER = sys.argv[1] if len(sys.argv) > 1 else "v11"
BEAMS = int(sys.argv[2]) if len(sys.argv) > 2 else 12

w, h, reach = load(6)
path = np.load(REPO / "results" / ("paths_room6_%s.npz" % VER))["b%d_connectome" % BEAMS]
tg = np.array(w.all_targets())
got = [int(np.argmax(np.linalg.norm(path - t[:2], axis=1) < w.reach_radius))
       if (np.linalg.norm(path - t[:2], axis=1) < w.reach_radius).any() else 10**9
       for t in tg]
cam = FisheyeCamera()
rec = []                                   # t, target idx, mass, distance
for k in range(len(path) - 1):
    rem = [i for i, g in enumerate(got) if g > k]
    if not rem:
        break
    world = TargetWorld(target=tg[rem[0]], extra_targets=[tg[i] for i in rem[1:]],
                        obstacles=w.obstacles, bounds=w.bounds)
    d = path[k + 1] - path[k]
    hd = math.atan2(d[1], d[0]) if np.hypot(*d) > 1e-6 else h
    p = np.array([path[k, 0], path[k, 1], 2.0])
    r = cam.sense(world, p, hd)
    if r["scent_mass"] <= 0:
        rec.append((k * 0.1, -1, 0.0, np.nan))
        continue
    wb = hd + math.radians(r["scent"])
    errs = [abs((math.atan2(tg[i][1] - p[1], tg[i][0] - p[0]) - wb + math.pi)
                % (2 * math.pi) - math.pi) for i in rem]
    i = rem[int(np.argmin(errs))]
    rec.append((k * 0.1, i, r["scent_mass"],
                float(np.hypot(tg[i][0] - p[0], tg[i][1] - p[1]))))

rec = np.array(rec)
print("beacons reached at t =", {i: g * 0.1 for i, g in enumerate(got) if g < 10**9},
      " never:", [i for i, g in enumerate(got) if g >= 10**9])
print("\n  t(s)  target  mass     dist(m)   (1 s means)")
for s in range(0, int(rec[-1, 0]) + 1, 2):
    m = (rec[:, 0] >= s) & (rec[:, 0] < s + 1)
    if not m.any():
        continue
    tgt = rec[m, 1].astype(int)
    vals, cnt = np.unique(tgt, return_counts=True)
    main = vals[np.argmax(cnt)]
    sel = m & (rec[:, 1] == main)
    print("  %4d   %3s    %.5f  %6.1f   (%d%% of the second on it)"
          % (s, main if main >= 0 else "-", rec[sel, 2].mean(),
             np.nanmean(rec[sel, 3]) if main >= 0 else np.nan,
             100 * cnt.max() / cnt.sum()))
np.save(REPO / "results" / ("odour_trace_room6_%s_b%d.npy" % (VER, BEAMS)), rec)
