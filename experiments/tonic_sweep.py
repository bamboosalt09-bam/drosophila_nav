"""Is the network stuck at its rectification floor?

`motor_drive_probe.py` found that past the optic lobe the median neuron's net
synaptic input is ~450x smaller than its own excitatory input and slightly
NEGATIVE, so the median rate is exactly 0.0 and ~60% of every stage sits at
the floor of `tanh(relu(v))`.

A network at its floor cannot represent a signed quantity: only the strongest
input gets through and it always has the same sign.  That predicts two things
this project has already measured and could not explain --

    `turn` correlates with bar azimuth at -0.67 but never crosses zero
    agonist-minus-antagonist is one-sided and ~1e-3

-- and it predicts the fix.  Real flies have tonic excitation setting a
non-zero baseline; this model has none.  Add a constant drive to every neuron
and the operating point lifts off the floor.

So: sweep the tonic level, and at each level sweep the bar around the fly.
The question is not "does activity increase" -- it trivially will.  The
question is whether the OUTPUT BECOMES SIGNED: does turn cross zero, and does
it stay correlated with where the bar is.  A tonic drive large enough to
saturate everything would also destroy the correlation, so both numbers have
to be read together.

Headings are batched as columns, so each tonic level is one run.

Usage:
    python experiments/tonic_sweep.py
    python experiments/tonic_sweep.py --duration 100 --headings 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch

import sensors.flywire_eye as eye
from core.flywire_rate import FlyWireRate
from core.malecns import load_malecns
from decoder.steering import joint_commands

DRIVE_LIT = 0.20          # same working point the heading probe uses
TONIC = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)


def signed(x: np.ndarray) -> float:
    """How far from crossing zero: 0 means centred, >=0.5 means one-sided."""
    rng = x.max() - x.min()
    return float("nan") if rng == 0 else abs(x.mean()) / rng


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--headings", type=int, default=12)
    ap.add_argument("--background", type=float, default=0.3)
    ap.add_argument("--out", default="results/tonic_sweep.csv")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
    lat = eye.load_malecns_eye(ann)

    idx = pd.Index(net.ids).get_indexer(lat.root_id)
    ok = idx >= 0
    inject = lat.of_type(*eye.MALECNS_INJECT_TYPES) & ok
    rows = idx[inject]
    az, el = lat.azimuth_deg[inject], lat.elevation_deg[inject]
    print("ready (%.1f s); injecting into %d columnar cells"
          % (time.perf_counter() - t0, len(rows)), flush=True)

    sc = ann["super_class"].to_numpy(dtype="<U32")
    side = ann["side"].to_numpy(dtype="<U16")
    desc = sc == "descending_neuron"
    desc_r, desc_l = desc & (side == "right"), desc & (side == "left")
    stages = ("ol_intrinsic", "cb_intrinsic", "descending_neuron",
              "vnc_intrinsic", "vnc_motor")

    headings = np.linspace(-180, 180, args.headings, endpoint=False)
    scene = eye.bar_scene(0.0, bar_width_deg=15.0, contrast=1.0,
                          background=args.background)

    # one column per heading; the visual part never changes with tonic level
    vis = torch.zeros(net.n, len(headings))
    for k, h in enumerate(headings):
        lum = np.zeros(len(az))
        for da in np.linspace(-0.5, 0.5, 5) * eye.ACCEPTANCE_DEG:
            lum += scene(az + h + da, el)
        lum /= 5.0
        vis[rows, k] = torch.from_numpy((lum * DRIVE_LIT).astype(np.float32))

    recs = []
    for tonic in TONIC:
        t0 = time.perf_counter()
        drive = vis + tonic
        with torch.no_grad():
            rates, _ = net.run(drive, args.duration, args.dt)
        r = rates.numpy()                      # (n_neurons, n_headings)

        turn_d = r[desc_r].sum(axis=0) - r[desc_l].sum(axis=0)
        turn_m, cmax = [], []
        for k in range(len(headings)):
            c = joint_commands(r[:, k], ann)
            lt = sum(v for kk, v in c.items() if kk[0] == "l")
            rt = sum(v for kk, v in c.items() if kk[0] == "r")
            turn_m.append(rt - lt)
            cmax.append(max(abs(v) for v in c.values()))
        turn_m = np.array(turn_m)

        sin_h = np.sin(np.deg2rad(headings))
        row = {"tonic": tonic,
               "desc_offset_over_range": signed(turn_d),
               "desc_corr_sin": float(np.corrcoef(sin_h, turn_d)[0, 1]),
               "motor_offset_over_range": signed(turn_m),
               "motor_corr_sin": float(np.corrcoef(sin_h, turn_m)[0, 1]),
               "motor_turn_min": turn_m.min(), "motor_turn_max": turn_m.max(),
               "max_joint_cmd": float(np.max(cmax)),
               "secs": time.perf_counter() - t0}
        for s in stages:
            row["frac_active_" + s] = float((r[sc == s] > 1e-12).mean())
        recs.append(row)
        print("tonic %-6g  active cb %.3f vnc %.3f  |  desc off/rng %.3f "
              "corr %+.3f  |  motor off/rng %.3f corr %+.3f  cmd %.2e  (%.0fs)"
              % (tonic, row["frac_active_cb_intrinsic"],
                 row["frac_active_vnc_intrinsic"],
                 row["desc_offset_over_range"], row["desc_corr_sin"],
                 row["motor_offset_over_range"], row["motor_corr_sin"],
                 row["max_joint_cmd"], row["secs"]), flush=True)

    df = pd.DataFrame(recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n--- verdict " + "-" * 50)
    base = df.iloc[0]
    best = df.loc[df["motor_offset_over_range"].idxmin()]
    print("tonic 0      motor off/rng %.3f  corr %+.3f"
          % (base["motor_offset_over_range"], base["motor_corr_sin"]))
    print("tonic %-6g motor off/rng %.3f  corr %+.3f"
          % (best["tonic"], best["motor_offset_over_range"],
             best["motor_corr_sin"]))
    crossed = df[df["motor_offset_over_range"] < 0.5]
    print("levels where motor turn CROSSES ZERO: %s"
          % (crossed["tonic"].tolist() or "none"))
    print("wrote %s" % args.out)
    # a level that crosses zero but has lost the bar is not a fix
    assert len(df) == len(TONIC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
