"""Re-create one moment of a saved flight and split the intended turn.

usage: diag_pose.py VER BEAMS T   (room 6, connectome path)
Static: the circuit settles 3 cycles from rest at that pose, with
the remaining beacons as they were.  Each input is removed in turn.
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np

import cx_vp_room as C
from core.vp_subnet import build_vp_subnet
from environment.room_file import ensure as room
from environment.target_world import TargetWorld
from sensors.rangefinder import LAYOUTS, Rangefinder
from sensors.vp_input import VPInput

VER, BEAMS, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
net, ann, sub, info = build_vp_subnet()
inp = VPInput(net, ann, info)
net.freeze_params(5.0)
w, h, reach = room(6)
C.wire_cue(net, inp, info, np.array(C.START))
gain, cc, zero = C.calibrate(net, inp, info, w, np.array(C.START))
inp.cam.rangefinder = Rangefinder(inp.cam.a, LAYOUTS[BEAMS])
P = np.load(REPO / "results" / ("paths_room6_%s.npz" % VER))["b%d_connectome" % BEAMS]
k = int(T / 0.1)
tg = np.array(w.all_targets())
got = [int(np.argmax(np.linalg.norm(P - t[:2], axis=1) < w.reach_radius))
       if (np.linalg.norm(P - t[:2], axis=1) < w.reach_radius).any() else 10**9 for t in tg]
rem = [tg[i] for i, g in enumerate(got) if g > k]
wld = TargetWorld(target=rem[0], extra_targets=rem[1:], obstacles=w.obstacles, bounds=w.bounds)
d = P[k + 1] - P[k]
hd = math.atan2(d[1], d[0])
p = np.array([P[k, 0], P[k, 1], 2.0])
inp.speed = float(np.hypot(*d) / 0.1)


def u(drop=()):
    v = net.init_state(1)
    inp.reset()
    for _ in range(3):
        dd, tr = inp.drive(wld, p, hd)
        for rows in drop:
            dd[rows] = 0.0
        v, r = C.step_circuit(net, v, dd)
    return math.degrees(gain * (C.steer(inp, r) - zero)), tr


full, tr = u()
orn = np.r_[inp.cue_left, inp.cue_right]      # the goal cue (PFL3)
vis = np.setdiff1d(inp.vp_rows, inp.thr_rows)
print("%s %d beams, t=%.1f s: pos (%.1f, %.1f) heading %+.0f deg, speed %.1f m/s"
      % (VER, BEAMS, T, p[0], p[1], math.degrees(hd), inp.speed))
print("odour: bearing %+.0f deg, mass %.4f" % (tr["scent"], tr["scent_mass"]))
print("intended turn (+ = left), deg/s:")
print("  everything              %+7.1f" % full)
print("  odour only removed      %+7.1f" % u([orn])[0])
print("  vision removed          %+7.1f" % u([vis])[0])
print("  threat removed          %+7.1f" % u([inp.thr_rows])[0])
print("  odour ONLY (vis+threat off) %+7.1f" % u([vis, inp.thr_rows])[0])
