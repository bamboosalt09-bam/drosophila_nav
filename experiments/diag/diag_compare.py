"""Room 6, connectome: path statistics of two versions side by side.

usage: diag_compare.py v13 v14
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from environment.room_file import load

w, h, reach = load(6)
obs = np.array([(o.x, o.y, o.radius) for o in w.obstacles])


def stats(path):
    seg = np.diff(path, axis=0)
    step = np.linalg.norm(seg, axis=1)
    mv = step > 0.02
    hd = np.unwrap(np.arctan2(seg[mv, 1], seg[mv, 0]))
    dh = np.diff(hd)
    dist = step.sum()
    clr = np.array([(np.hypot(*(obs[:, :2] - p).T) - obs[:, 2]).min() for p in path[::2]])
    # sign changes of the turn: zig-zag
    turning = np.abs(dh) > math.radians(3)
    sgn = np.sign(dh[turning])
    flips = int((sgn[1:] != sgn[:-1]).sum())
    return (dist / (len(path) * 0.1), math.degrees(np.abs(dh).sum()) / max(dist, 1e-9),
            100 * (dh > 0).sum() / max(len(dh), 1), flips / max(dist, 1e-9) * 10,
            clr.min())


print("%4s %4s %6s %9s %11s %9s %13s %9s %7s %6s"
      % ("ver", "beam", "found", "speed", "turn deg/m", "left %", "flips /10 m",
         "min clr", "cue", "corr"))
for n in (0, 1, 3, 6, 12, 24):
    for ver in sys.argv[1:]:
        if "b%d_connectome" % n not in np.load(REPO / "results" / ("paths_room6_%s.npz" % ver)).files:
            continue
        P = np.load(REPO / "results" / ("paths_room6_%s.npz" % ver))["b%d_connectome" % n]
        d = pd.read_csv(REPO / "results" / ("sweep_beams_room6_%s.csv" % ver))
        row = d[(d.beams == n) & (d.arm == "connectome")].iloc[0]
        sp, tpm, left, flips, clr = stats(P)
        print("%4s %4d %6d %7.2f m/s %9.1f %9.0f %11.1f %8.2f m %7.2f %6.2f"
              % (ver, n, row.found, sp, tpm, left, flips, clr, row.cue_frac, row["corr"]))
