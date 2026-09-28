"""Room 6, the corridor loop: what is ahead of the drone when it turns?

Geometry only, from the saved path: speed, yaw rate, and the free distance
straight ahead (a 0.4 m-wide corridor, like the drone layer's swept path)
against the room's obstacles.  No circuit, no drone layer.
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from environment.room_file import load

VER = sys.argv[1] if len(sys.argv) > 1 else "v11"
BEAMS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
T0 = float(sys.argv[3]) if len(sys.argv) > 3 else 65.0
DT, HALF_W = 0.1, 0.40

w, h, reach = load(6)
obs = np.array([(o.x, o.y, o.radius) for o in w.obstacles])
P = np.load(REPO / "results" / ("paths_room6_%s.npz" % VER))["b%d_connectome" % BEAMS]


def free_ahead(p, hd, horizon=6.0):
    """Distance along heading to the first obstacle within HALF_W of the line."""
    d = obs[:, :2] - p
    u = np.array([math.cos(hd), math.sin(hd)])
    along = d @ u
    across = np.abs(d @ np.array([-u[1], u[0]]))
    hit = (along > 0) & (across < obs[:, 2] + HALF_W)
    if not hit.any():
        return horizon
    return float(min(horizon, (along[hit] - np.sqrt(np.maximum(
        (obs[hit, 2] + HALF_W) ** 2 - across[hit] ** 2, 0))).min()))


seg = np.diff(P, axis=0)
spd = np.linalg.norm(seg, axis=1) / DT
# heading from motion; hold the last one while (nearly) stationary
hd = np.zeros(len(seg))
last = h
for i, s in enumerate(seg):
    if np.hypot(*s) > 0.02:
        last = math.atan2(s[1], s[0])
    hd[i] = last
yaw = np.degrees(np.diff(np.unwrap(hd))) / DT          # deg/s, + = left
k0 = int(T0 / DT)
rows = []
for i in range(k0, len(yaw)):
    rows.append((i * DT, spd[i], yaw[i], free_ahead(P[i], hd[i]),
                 float((np.hypot(*(obs[:, :2] - P[i]).T) - obs[:, 2]).min())))
R = np.array(rows)
t, v, y, fa, near = R.T

print("%s, %d beams, t >= %.0f s: %d steps" % (VER, BEAMS, T0, len(R)))
print("x range %.1f..%.1f m, y range %.1f..%.1f m"
      % (P[k0:, 0].min(), P[k0:, 0].max(), P[k0:, 1].min(), P[k0:, 1].max()))
turning = np.abs(y) > 30
print("\nturning (|yaw| > 30 deg/s): %d%% of steps, %d%% of those LEFT"
      % (100 * turning.mean(), 100 * (y[turning] > 0).mean()))
for lo, hi in ((0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99)):
    m = (fa >= lo) & (fa < hi)
    if m.any():
        print("  free ahead %.0f-%s m: %4d steps, turning %3d%%, mean speed %.2f m/s, "
              "mean yaw %+5.0f deg/s"
              % (lo, "%.0f" % hi if hi < 99 else "+", m.sum(),
                 100 * turning[m].mean(), v[m].mean(), y[m].mean()))
print("\nstraight-ahead free distance when a turn STARTS:")
starts = np.flatnonzero(turning[1:] & ~turning[:-1]) + 1
for s in starts[:25]:
    e = s
    while e < len(y) and turning[e]:
        e += 1
    print("  t=%5.1f s  free %.1f m  speed %.2f m/s  -> turns %+4.0f deg over %.1f s "
          "(mean %+4.0f deg/s), nearest obstacle %.2f m"
          % (t[s], fa[s], v[s], y[s:e].sum() * DT, (e - s) * DT, y[s:e].mean(), near[s]))
