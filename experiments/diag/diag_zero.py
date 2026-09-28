"""Why did 0 beams do best in room 6 up to v8?  From saved paths only."""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from environment.room_file import load

w, h, reach = load(6)
tg = np.array(w.all_targets())[:, :2]
obs = [(o.x, o.y, o.radius) for o in w.obstacles]


def clearance(p):
    return min(math.hypot(p[0] - x, p[1] - y) - r for x, y, r in obs)


def stats(path):
    seg = np.diff(path, axis=0)
    step = np.linalg.norm(seg, axis=1)
    moving = step > 1e-4
    hd = np.arctan2(seg[moving, 1], seg[moving, 0])
    dh = np.abs((np.diff(hd) + np.pi) % (2 * np.pi) - np.pi)
    dist = step.sum()
    reached = sorted(int(np.argmax(np.linalg.norm(path - t, axis=1) < w.reach_radius))
                     for t in tg if (np.linalg.norm(path - t, axis=1) < w.reach_radius).any())
    clr = np.array([clearance(p) for p in path[::2]])
    return {"speed": dist / (len(path) * 0.1), "turn_per_m": math.degrees(dh.sum()) / max(dist, 1e-9),
            "t5": (reached[4] * 0.1) if len(reached) >= 5 else None,
            "min_clear": clr.min(), "near_frac": float((clr < 0.8).mean())}


for ver in ("v8", "v9"):
    P = np.load(REPO / "results" / ("paths_room6_%s.npz" % ver))
    print("=== %s, connectome, room 6" % ver)
    print("%5s %9s %12s %10s %12s %14s"
          % ("beams", "speed", "turn deg/m", "5th at", "min clear", "time < 0.8 m"))
    for n in (0, 1, 3, 6, 12, 24):
        s = stats(P["b%d_connectome" % n])
        print("%5d %7.2f m/s %10.1f %9s %10.2f m %12.0f%%"
              % (n, s["speed"], s["turn_per_m"],
                 ("%.0f s" % s["t5"]) if s["t5"] else "-", s["min_clear"],
                 100 * s["near_frac"]))
