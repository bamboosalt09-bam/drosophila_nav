"""One table: what each input does to the steering readout, alone.

Every input the drone gives the circuit, switched on by itself on top of
the same baseline (empty world, standing, no cue, no rotation), once with
the stimulus on the LEFT and once on the RIGHT.  u is the intended turn in
deg/s (+ = left) through the runner's own calibration.

    vision  a pillar 2.5 m away at +-45 deg, standing (no looming)
    threat  the same pillar approached at 2 m/s, 12 beams, threat part only
    cue     a beacon 10 m away at +-60 deg, the PFL3 goal cue only
    (gyro: removed 2026-09-28 -- via JO it measured -0.2 / +0.2 deg/s)
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
from environment.target_world import Obstacle, TargetWorld
from sensors.rangefinder import LAYOUTS, Rangefinder
from sensors.vp_input import CUE_GAIN, VPInput

net, ann, sub, info = build_vp_subnet()
inp = VPInput(net, ann, info)
net.freeze_params(5.0)
p0 = np.array(C.START)
ul, ur = C.wire_cue(net, inp, info, p0)
gain, cc, zero = C.calibrate(net, inp, info, None, p0)
inp.cam.rangefinder = Rangefinder(inp.cam.a, LAYOUTS[12])
far = np.array([300.0, 0.0, 2.0])
x0, y0 = p0[0], p0[1]


def drive(world, speed=0.0, cue=0.0):
    """The stationary drive for one condition (two frames, so no ramp)."""
    inp.speed, inp.cue_gain = speed, cue
    inp.reset()
    inp.drive(world, p0, 0.0)
    d, _ = inp.drive(world, p0, 0.0)
    inp.speed, inp.cue_gain = 2.0, CUE_GAIN
    return d


def u_of(d):
    v = net.init_state(1)
    for _ in range(3):
        v, r = C.step_circuit(net, v, d)
    return math.degrees(gain * (C.steer(inp, r) - zero))


empty = TargetWorld(target=far, obstacles=[])
base = drive(empty)
u0 = u_of(base)


def pillar(sign):
    a = math.radians(45 * sign)
    return TargetWorld(target=far, obstacles=[
        Obstacle(x0 + 2.5 * math.cos(a), y0 + 2.5 * math.sin(a), 0.5, height=6.0)])


def beacon(sign):
    a = math.radians(60 * sign)
    return TargetWorld(target=np.array([x0 + 10 * math.cos(a), y0 + 10 * math.sin(a), 2.0]),
                       obstacles=[])


def only(part):
    """Baseline plus one input's contribution."""
    return u_of(base + part) - u0


rows = []
for sign in (1, -1):
    wv = pillar(sign)
    vis = drive(wv)                                   # standing: vision only
    rows.append(("vision", sign, only(vis - base)))
    thr = drive(wv, speed=2.0) - vis                  # the looming part
    rows.append(("threat", sign, only(thr)))
    wb = beacon(sign)
    cue = drive(wb, cue=CUE_GAIN) - drive(wb)         # the PFL3 part
    rows.append(("cue", sign, only(cue)))

print("baseline u %+.1f deg/s; calibration gain %.3g; PFL3 wiring L %+.0f R %+.0f"
      % (u0, gain, ul, ur))
print("\n%-7s %14s %14s %10s   %s" % ("input", "stim LEFT", "stim RIGHT", "L+R", "expected"))
expect = {"vision": "away from the pillar (L-, R+)",
          "threat": "away from the pillar (L-, R+)",
          "cue": "toward the beacon (L+, R-)"}
for name in ("vision", "threat", "cue"):
    a = [u for n, s, u in rows if n == name and s == 1][0]
    b = [u for n, s, u in rows if n == name and s == -1][0]
    print("%-7s %+12.1f %+14.1f %+10.1f   %s" % (name, a, b, a + b, expect[name]))
