"""Where does the standing RIGHT turn come from?  Static probes, no flight.

Same setup as cx_vp_room.main (room 6 start, frozen params, same
calibration).  u > 0 is a LEFT turn, deg/s: the intended turn the drone
layer receives.  Each probe settles 3 control cycles from rest, like
calibrate().  Symmetric scenes should give u = 0; mirrored ones +-equal.

Measured 2026-09-28: no cue +2.9; threat on all cells -21 (0.25) / -78
(1.0), left cells -81 vs right +62; pillar dead ahead -35; mirrored pair
-23 (vision alone -17); corridor -26; pillar 45 deg L/R -56/+30; beacon
60 deg L/R +77/-61.  Gyro at 90 deg/s: +0.3.
Tried and reverted: re-weighting cells per 30 deg azimuth band so each
mirror pair turned equal and opposite.  The circuit is nonlinear, so the
weights that balance a band-wide probe do not balance a single object: the
pair went -23 -> -6 but the corridor -26 -> +18 and the beacon sum
+16 -> +56; the frontal threat band (-4.9 vs +0.2) cannot be balanced by
weights at all.
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
from environment.target_world import Obstacle, TargetWorld
from sensors.rangefinder import LAYOUTS, Rangefinder
from sensors.vp_input import VPInput

net, ann, sub, info = build_vp_subnet()
inp = VPInput(net, ann, info)
net.freeze_params(5.0)
w, h, reach = room(6)
p0 = np.array(C.START)
gain, cc, zero = C.calibrate(net, inp, info, w, p0)
inp.cam.rangefinder = Rangefinder(inp.cam.a, LAYOUTS[12])
inp.speed = 2.0
far = np.array([300.0, 0.0, 2.0])
x0, y0 = p0[0], p0[1]


def u_of(world, yaw=0.0):
    inp.yaw_rate = yaw
    v = net.init_state(2 if inp.twin else 1)
    inp.reset()
    for _ in range(3):
        d, tr = inp.drive(world, p0, 0.0)
        v, r = C.step_circuit(net, v, C.twin(inp, d))
    inp.yaw_rate = 0.0
    return math.degrees(gain * (C.steer(inp, r) - zero))


def beacon(deg, d=25.0):
    a = math.radians(deg)
    return TargetWorld(target=np.array([x0 + d * math.cos(a), y0 + d * math.sin(a), 2.0]),
                       obstacles=[])


def world(obs):
    return TargetWorld(target=far, obstacles=obs)


scenes = {
    "no cue (empty)": world([]),
    "pillar dead ahead 2.0 m": world([Obstacle(x0 + 2.0, y0, 0.8, height=6.0)]),
    "mirrored pillar pair": world([Obstacle(x0 + 2.5, y0 + 1.2, 0.5, height=6.0),
                                   Obstacle(x0 + 2.5, y0 - 1.2, 0.5, height=6.0)]),
    "corridor, walls 1.5 m each side": world(
        [Obstacle(x0 + dx, y0 + sy * 2.0, 0.5, height=6.0)
         for dx in np.arange(-3.0, 6.1, 1.0) for sy in (1, -1)]),
}
mirrored = {
    "pillar 45 deg LEFT / RIGHT at 2.5 m": [
        world([Obstacle(x0 + 1.77, y0 + s * 1.77, 0.5, height=6.0)]) for s in (1, -1)],
    "beacon 60 deg LEFT / RIGHT (odour)": [beacon(60), beacon(-60)],
}


def report(tag):
    print("\n" + tag)
    for nm, wld in scenes.items():
        print("    %-34s u = %+7.1f" % (nm, u_of(wld)))
    for nm, (a, b) in mirrored.items():
        ua, ub = u_of(a), u_of(b)
        print("    %-34s u = %+7.1f / %+7.1f   (sum %+6.1f, should be 0)"
              % (nm, ua, ub, ua + ub))


report("as flown")
print("\ngyro, turning right at 90 deg/s: change %+.1f deg/s"
      % (u_of(world([]), yaw=-math.radians(90)) - u_of(world([]))))

inp.twin = True
gain, cc, zero = C.calibrate(net, inp, info, w, p0)
print("\nmirror twin: recalibrated gain %.3g, zero %.2g" % (gain, zero))
report("mirror twin")
