"""Spread (max distance from centre) of each completed full turn along a
path: separate the corridor loop from the room-scale sweeps."""
import math
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "results"
WIN = 200                      # 20 s of 0.1 s steps


def full_turns(path):
    seg = np.diff(path, axis=0)
    hd = np.arctan2(seg[:, 1], seg[:, 0])
    dh = (np.diff(hd) + np.pi) % (2 * np.pi) - np.pi      # yaw per step
    out = []
    k = 1
    while k < len(dh):
        acc, start = 0.0, None
        for i in range(k, max(k - WIN, -1), -1):
            acc += dh[i]
            if abs(acc) > 2 * math.pi:
                start = i
                break
        if start is not None:
            pts = path[start:k + 2]
            spread = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()
            gap = float(np.linalg.norm(path[k + 1] - path[start]))
            out.append((k * 0.1, spread, gap))
            k += 30                  # skip ahead, like the trigger does
        else:
            k += 1
    return out


for ver, beams, label in (("v10", 6, "sweep that mis-fired (v10)"),
                          ("v10", 24, "sweep that mis-fired (v10)"),
                          ("v11", 12, "corridor loop, not caught (v11)"),
                          ("v11", 3, "clean route (v11)")):
    P = np.load(R / ("paths_room6_%s.npz" % ver))["b%d_connectome" % beams]
    ft = full_turns(P)
    print("%-3s %2d beams  %-32s full turns: %s"
          % (ver, beams, label,
             ", ".join("t=%.0fs close %.1fm" % (t, g) for t, s, g in ft) or "none"))
