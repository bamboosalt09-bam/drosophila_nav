"""Along the v7 12-beam connectome flight in room 6: where does the odour
bearing point, relative to the individual beacons in view?"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from environment.room_file import load
from environment.target_world import TargetWorld
from sensors.fisheye import FisheyeCamera

w, h, reach = load(6)
path = np.load(REPO / "results" / "paths_room6_v7.npz")["b12_connectome"]
tg = np.array(w.all_targets())
got = [int(np.argmax(np.linalg.norm(path - t[:2], axis=1) < w.reach_radius))
       if (np.linalg.norm(path - t[:2], axis=1) < w.reach_radius).any() else 10**9
       for t in tg]
cam = FisheyeCamera()
rows = []
for k in range(0, len(path) - 1, 10):
    rem = [t for t, g in zip(tg, got) if g > k]
    if not rem:
        break
    world = TargetWorld(target=rem[0], extra_targets=rem[1:],
                        obstacles=w.obstacles, bounds=w.bounds)
    d = path[k + 1] - path[k]
    hd = math.atan2(d[1], d[0]) if np.hypot(*d) > 1e-6 else h
    p = np.array([path[k, 0], path[k, 1], 2.0])
    r = cam.sense(world, p, hd)
    bl = [b for b in cam.blobs(r["unlit"])]
    if len(bl) < 2:
        continue
    # match each blob to a beacon by bearing, to know its distance
    info = []
    for b in bl:
        best = None
        for t in rem:
            brg = math.degrees((math.atan2(t[1] - p[1], t[0] - p[0]) - hd
                                + math.pi) % (2 * math.pi) - math.pi)
            e = abs(brg - b.bearing_deg)
            if best is None or e < best[0]:
                best = (e, float(np.hypot(t[0] - p[0], t[1] - p[1])))
        info.append((b.bearing_deg, b.intensity, best[1]))
    info.sort(key=lambda x: x[2])
    near_brg = info[0][0]
    scent = r["scent"]
    to_near = abs(scent - near_brg)
    to_any = min(abs(scent - x[0]) for x in info)
    rows.append((k, len(info), scent, info, to_near, to_any))

n = len(rows)
at_near = sum(1 for r in rows if r[4] < 15)
at_other = sum(1 for r in rows if r[4] >= 15 and r[5] < 15)
between = sum(1 for r in rows if r[5] >= 15)
print("%d sampled moments with 2+ beacons in view" % n)
print("  odour bearing within 15 deg of the NEAREST beacon : %3d (%.0f%%)" % (at_near, 100 * at_near / n))
print("  ... of a FARTHER beacon instead                    : %3d (%.0f%%)" % (at_other, 100 * at_other / n))
print("  ... of NO beacon (pointing between them)           : %3d (%.0f%%)" % (between, 100 * between / n))
print("\nfirst few moments (bearing deg, pull = pixel mass, distance m):")
for k, nb, scent, info, tn, ta in rows[:6]:
    print("  t=%4.1fs  odour %+6.1f | %s" % (k * 0.1, scent, "  ".join(
        "%+.0f (pull %.0f, %.1f m)" % (b, s, dd) for b, s, dd in info)))
