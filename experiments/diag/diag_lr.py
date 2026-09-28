"""Left/right symmetry of every input and readout the loop uses (no flight)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from core.vp_subnet import build_vp_subnet
from sensors.vp_input import VPInput

net, ann, sub, info = build_vp_subnet()
inp = VPInput(net, ann, info)
side = ann["side"].astype(str).to_numpy()
ct = ann["cell_type"].astype(str).to_numpy()
az = np.full(net.n, np.nan)
az[inp.vp_rows] = inp.vp_az
for name, l, r in (("PFL3 goal cue", inp.pfl3_l, inp.pfl3_r), ("JO gyro", inp.jo_l, inp.jo_r),
                   ("steer DNa01/02", inp.steer_l, inp.steer_r),
                   ("PFL3", inp.pfl3_l, inp.pfl3_r)):
    print("%-15s left %4d  right %4d" % (name, len(l), len(r)))
tl = inp.thr_rows[side[inp.thr_rows] == "left"]
tr = inp.thr_rows[side[inp.thr_rows] == "right"]
print("%-15s left %4d  right %4d   (cells with a receptive direction)" % ("LC4+LPLC2", len(tl), len(tr)))
for nm, rows in (("left", tl), ("right", tr)):
    a = az[rows]
    print("   %-5s cells: azimuth mean %+6.1f, looking left(+) %3d / right(-) %3d, "
          "within +-30 deg of ahead %3d" % (nm, a.mean(), (a > 0).sum(), (a < 0).sum(),
                                            (np.abs(a) < 30).sum()))
for t in ("LC4", "LPLC2"):
    print("   %-6s left %3d right %3d" % (t, ((ct == t) & (side == "left")).sum(),
                                        ((ct == t) & (side == "right")).sum()))
vl = inp.vp_rows[side[inp.vp_rows] == "left"]
vr = inp.vp_rows[side[inp.vp_rows] == "right"]
print("%-15s left %4d  right %4d; azimuth mean L %+5.1f R %+5.1f"
      % ("VP (vision)", len(vl), len(vr), np.nanmean(az[vl]), np.nanmean(az[vr])))
