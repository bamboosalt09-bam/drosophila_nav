"""Room 6: after the last visible beacon, does the drone bounce off walls?

Geometry only.  An "event" is a stretch where the drone is slower than
0.5 m/s (stopped / turning in place / backing off: the drone layer's
blocked branch).  For each: where, how close the nearest obstacle was, and
the heading before vs after -- a reversal is ~180 deg.

usage: diag_bounce.py VER BEAMS T0
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from environment.room_file import load

VER, BEAMS, T0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
w, h, reach = load(6)
obs = np.array([(o.x, o.y, o.radius) for o in w.obstacles])
P = np.load(REPO / "results" / ("paths_room6_%s.npz" % VER))["b%d_connectome" % BEAMS]
seg = np.diff(P, axis=0)
spd = np.linalg.norm(seg, axis=1) / 0.1
hd = np.arctan2(seg[:, 1], seg[:, 0])
slow = spd < 0.5
k = int(T0 / 0.1)
n = 0
print("%s %d beams, t >= %.0f s" % (VER, BEAMS, T0))
while k < len(slow):
    if not slow[k]:
        k += 1
        continue
    e = k
    while e < len(slow) and slow[e]:
        e += 1
    a, b = max(k - 3, 0), min(e + 3, len(hd) - 1)
    turn = math.degrees((hd[b] - hd[a] + math.pi) % (2 * math.pi) - math.pi)
    near = (np.hypot(*(obs[:, :2] - P[k]).T) - obs[:, 2]).min()
    n += 1
    print("  t=%5.1f-%5.1f s at (%5.1f,%5.1f), wall %.2f m: heading %+4.0f -> %+4.0f "
          "(turned %+4.0f)" % (k * 0.1, e * 0.1, P[k, 0], P[k, 1], near,
                               math.degrees(hd[a]), math.degrees(hd[b]), turn))
    k = e
straight = spd[int(T0 / 0.1):] >= 0.5
print("%d events; %.0f%% of the time moving at >= 0.5 m/s" % (n, 100 * straight.mean()))

# U-turns at speed: >= 150 deg of heading change within 3 s
print("\nU-turns (>=150 deg within 3 s):")
hu = np.unwrap(hd)
k, W = int(T0 / 0.1), 30
while k + W < len(hu):
    dturn = math.degrees(hu[k + W] - hu[k])
    if abs(dturn) >= 150:
        # free distance straight ahead when the turn began
        u = np.array([math.cos(hd[k]), math.sin(hd[k])])
        dd = obs[:, :2] - P[k]
        along, across = dd @ u, np.abs(dd @ np.array([-u[1], u[0]]))
        hit = (along > 0) & (across < obs[:, 2] + 0.4)
        ahead = along[hit].min() if hit.any() else np.inf
        print("  t=%5.1f s at (%5.1f,%5.1f): %+4.0f deg in 3 s, speed %.1f m/s, "
              "wall straight ahead %.1f m"
              % (k * 0.1, P[k, 0], P[k, 1], dturn, spd[k:k + W].mean(), ahead))
        k += W
    else:
        k += 1
