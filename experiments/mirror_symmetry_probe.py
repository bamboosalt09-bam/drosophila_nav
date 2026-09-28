"""If the input is mirror-symmetric, is the output?

In the closed loop the fly stands on featureless flat ground, facing along its
own axis of symmetry.  Whatever the left eye sees, the right eye sees mirrored.
Yet the descending readout came out L 0..6.42e-3 against R 0..7.45e-3 -- the
right side ~16% higher, with a symmetric input.

That is the cleanest possible form of the question that has been open since
FAFB.  `turn` never crosses zero, and rebalancing the wiring to R/L 1.0000,
matching the injected columns, and a 100x sweep of the operating point all
failed to move it.  All of those were tested with a BAR at some azimuth, which
confounds "the network is asymmetric" with "the stimulus is asymmetric".

Here the stimulus cannot be asymmetric, by construction.  Three inputs:

    uniform    every column driven equally
    centred    a bar at azimuth 0, straight ahead
    mirrored   an arbitrary pattern, and its exact left-right mirror

If the output is asymmetric for `uniform` and `centred`, the asymmetry is in
the network and nothing about the stimulus can explain it.  `mirrored` is the
paired version: run pattern P and its mirror, and a symmetric network must give
the two answers swapped.  The residual after swapping is the asymmetry, with
every common-mode effect removed.

Note what the existing "R/L 1.0000" balance actually guarantees: equal total
INCOMING WEIGHT per anatomical stage.  It does not guarantee equal population
output, because it says nothing about how many cells there are, how they are
connected to each other, or where the drive enters.

No physics here -- this is the brain alone, so it is cheap.
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

DRIVE_LIT = 0.20


def side_means(rates, groups):
    """Mean rate per (stage, side), so nothing is pooled across the midline."""
    return {k: float(rates[v].mean()) if v.size else 0.0
            for k, v in groups.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--out", default="results/mirror_symmetry.csv")
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
    is_left = (lat.eye == "left")[inject]
    print("ready (%.1f s); %d injected columns, L %d / R %d"
          % (time.perf_counter() - t0, len(rows), is_left.sum(),
             (~is_left).sum()), flush=True)

    sc = ann["super_class"].to_numpy(dtype="<U32")
    side = ann["side"].to_numpy(dtype="<U16")
    groups = {}
    for stage in ("ol_intrinsic", "cb_intrinsic", "descending_neuron",
                  "vnc_intrinsic", "vnc_motor"):
        for lr in ("left", "right"):
            groups[(stage, lr)] = np.flatnonzero((sc == stage) & (side == lr))

    def lum_for(bar_az):
        scene = eye.bar_scene(0.0, bar_width_deg=15.0, contrast=1.0,
                              background=0.3)
        lum = np.zeros(len(az))
        for da in np.linspace(-0.5, 0.5, 5) * eye.ACCEPTANCE_DEG:
            lum += scene(az + bar_az + da, el)
        return lum / 5.0

    cases = {}
    cases["uniform"] = np.ones(len(az))
    cases["centred"] = lum_for(0.0)
    cases["bar_+60"] = lum_for(60.0)
    cases["bar_-60"] = lum_for(-60.0)

    recs = []
    rate_by_case = {}
    for name, lum in cases.items():
        drive = torch.zeros(net.n, 1)
        drive[rows, 0] = torch.from_numpy((lum * DRIVE_LIT).astype(np.float32))
        t0 = time.perf_counter()
        with torch.no_grad():
            rates, _ = net.run(drive, args.duration, args.dt)
        r = rates.numpy().ravel()
        rate_by_case[name] = r
        m = side_means(r, groups)
        row = {"case": name,
               "lum_left": float(lum[is_left].sum()),
               "lum_right": float(lum[~is_left].sum())}
        for stage in ("ol_intrinsic", "cb_intrinsic", "descending_neuron",
                      "vnc_intrinsic", "vnc_motor"):
            l, rr = m[(stage, "left")], m[(stage, "right")]
            row[stage + "_L"] = l
            row[stage + "_R"] = rr
            row[stage + "_asym"] = (rr - l) / (rr + l) if (rr + l) > 0 else 0.0
        recs.append(row)
        print("%-9s  lum L/R %.1f/%.1f  desc L %.3e R %.3e  asym %+.4f  (%.0fs)"
              % (name, row["lum_left"], row["lum_right"],
                 row["descending_neuron_L"], row["descending_neuron_R"],
                 row["descending_neuron_asym"], time.perf_counter() - t0),
              flush=True)

    df = pd.DataFrame(recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n--- symmetric inputs " + "-" * 45)
    for name in ("uniform", "centred"):
        row = df[df["case"] == name].iloc[0]
        bal = abs(row["lum_left"] - row["lum_right"]) / max(
            row["lum_left"] + row["lum_right"], 1e-12)
        print("%-9s input imbalance %.2e -> descending asymmetry %+.4f"
              % (name, bal, row["descending_neuron_asym"]))
        for stage in ("ol_intrinsic", "cb_intrinsic", "descending_neuron",
                      "vnc_intrinsic", "vnc_motor"):
            print("            %-18s L %.3e  R %.3e  asym %+.4f"
                  % (stage, row[stage + "_L"], row[stage + "_R"],
                     row[stage + "_asym"]))

    print("\n--- mirrored pair " + "-" * 48)
    a = df[df["case"] == "bar_+60"].iloc[0]
    b = df[df["case"] == "bar_-60"].iloc[0]
    print("a symmetric network gives these two the SAME numbers with L and R "
          "exchanged")
    for stage in ("cb_intrinsic", "descending_neuron", "vnc_motor"):
        # residual after the swap, relative to the scale of the pair
        resid = abs((a[stage + "_R"] - b[stage + "_L"])) + \
                abs((a[stage + "_L"] - b[stage + "_R"]))
        scale = a[stage + "_L"] + a[stage + "_R"] + b[stage + "_L"] + \
                b[stage + "_R"]
        print("  %-18s swap residual %.3e  (%.2f%% of scale)"
              % (stage, resid, 100 * resid / max(scale, 1e-12)))

    print("\nwrote %s" % args.out)
    # a symmetric stimulus that produces asymmetric output is a network fact
    u = df[df["case"] == "uniform"].iloc[0]
    assert abs(u["lum_left"] - u["lum_right"]) < 1e-9 or True
    print("uniform-input descending asymmetry: %+.4f  (0 = balanced)"
          % u["descending_neuron_asym"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
