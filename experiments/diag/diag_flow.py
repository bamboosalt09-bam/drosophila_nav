"""Does yaw_flow read the drone's rotation off real room-6 images?

(1) pure rotation: same spot, heading changed by a known amount;
(2) real flight: along the saved v16 24-beam path, image rotation against
    the heading change the path itself implies (translation adds parallax).
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from environment.room_file import load
from sensors.fisheye import FisheyeCamera
from sensors.vp_input import CONTROL_DT, yaw_flow

w, h, reach = load(6)
cam = FisheyeCamera()


def prof(p, hd):
    r = cam.sense(w, np.array([p[0], p[1], 2.0]), hd)
    return r["unlit"].reshape(cam.n_az, cam.n_el).mean(axis=1)


print("(1) pure rotation at the start and two other spots")
for p in ((0.0, 0.0), (-12.0, -8.0), (10.0, -12.0)):
    row = []
    for rate in (-90, -30, 0, 30, 90):
        a = prof(p, 0.3)
        b = prof(p, 0.3 + math.radians(rate) * CONTROL_DT)
        row.append("%+d->%+.0f" % (rate, yaw_flow(a, b, cam.da)))
    print("   at (%5.1f,%5.1f): %s" % (p[0], p[1], "  ".join(row)))

print("\n(2) along the v16 24-beam flight (deg/s, + = left)")
P = np.load(REPO / "results" / "paths_room6_v16.npz")["b24_connectome"]
seg = np.diff(P, axis=0)
hd = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
true, est = [], []
for k in range(1, len(hd) - 1, 3):
    a, b = prof(P[k], hd[k - 1]), prof(P[k + 1], hd[k])
    true.append(math.degrees(hd[k] - hd[k - 1]) / CONTROL_DT)
    est.append(yaw_flow(a, b, cam.da))
true, est = np.array(true), np.array(est)
print("   %d frames: r = %.2f, median |error| %.1f deg/s, sign agrees %.0f%% "
      "(|true| > 10)" % (len(true), np.corrcoef(true, est)[0, 1],
                         np.median(np.abs(est - true)),
                         100 * np.mean(np.sign(est[np.abs(true) > 10])
                                       == np.sign(true[np.abs(true) > 10]))))
for t0 in (40, 70, 100):
    m = slice(int(t0 / 0.3), int(t0 / 0.3) + 6)
    print("   t~%3ds  true %s" % (t0, " ".join("%+4.0f" % x for x in true[m])))
    print("          est  %s" % " ".join("%+4.0f" % x for x in est[m]))
