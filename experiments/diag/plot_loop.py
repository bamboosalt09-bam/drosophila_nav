"""Zoom on the room-6 corridor loop, path coloured by yaw rate."""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment.room_file import load

VER, BEAMS, T0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
FULL = "full" in sys.argv[4:]      # whole room instead of the left pocket
w, h, reach = load(6)
P = np.load(REPO / "results" / ("paths_room6_%s.npz" % VER))["b%d_connectome" % BEAMS]
seg = np.diff(P, axis=0)
hd, last = np.zeros(len(seg)), h
for i, s in enumerate(seg):
    if np.hypot(*s) > 0.02:
        last = math.atan2(s[1], s[0])
    hd[i] = last
yaw = np.degrees(np.diff(np.unwrap(hd))) / 0.1
k0 = int(T0 / 0.1)
fig, ax = plt.subplots(figsize=(8, 8) if FULL else (6, 8))
for o in w.obstacles:
    ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="0.6"))
for t in w.all_targets():
    ax.plot(t[0], t[1], "*", color="green", ms=12)
sc = ax.scatter(P[k0:-2, 0], P[k0:-2, 1], c=yaw[k0:], cmap="coolwarm_r",
                vmin=-90, vmax=90, s=8)
ax.plot(P[k0, 0], P[k0, 1], "ko")
ax.set_xlim(*((-19, 19) if FULL else (-18.5, -4)))
ax.set_ylim(*((-19, 19) if FULL else (-18.5, 3)))
ax.set_aspect("equal")
fig.colorbar(sc, ax=ax, label="yaw rate, deg/s  (red = RIGHT turn, blue = left)")
ax.set_title("room 6, %s, %d beams, t >= %.0f s (black dot = start of window)"
             % (VER, BEAMS, T0), fontsize=9)
out = REPO / "results" / ("loop_%s_b%d_room6.png" % (VER, BEAMS))
fig.savefig(out, dpi=80, bbox_inches="tight")
print(out)
