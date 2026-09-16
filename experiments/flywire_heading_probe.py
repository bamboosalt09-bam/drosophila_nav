"""First real question: shown a bar, does anything in the brain encode where it is?

Open loop on purpose.  Sweep the bar's azimuth, run the connectome, and ask
whether any population's activity depends on it.  If nothing does, a closed
loop cannot work and there is no point building one.

Nothing between input and output is designated: drive goes in at the columnar
cells that carry the fly's own view, the steering readout is the left-right
difference over descending neurons, and EPG is only OBSERVED -- looking at a
population is not the same as choosing it to do the job.

Usage:
    python experiments/flywire_heading_probe.py
    python experiments/flywire_heading_probe.py --duration 300 --headings 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

import sensors.flywire_eye as eye
from core.flywire_brain import FlyWireBrain
from decoder.steering import DescendingPair

# Luminance -> drive, in mV per step.  Steady state is V_REST + drive *
# tau_mbr/dt, so 0.035 reaches threshold -- but reaching threshold is not
# enough to cross the optic lobe.  Measured stage by stage, descending
# neurons stay silent at 0.06, reach 124 active at 0.20 and 213 at 0.50,
# where the injected cells saturate at the 455 Hz refractory ceiling.
# 0.20 is the working point: signal gets through without pinning the input.
# ponytail: lamina cells actually INVERT luminance (L1/L2 hyperpolarise to
# light).  Driving them proportionally is the crude version; flip the sign per
# type if the polarity turns out to matter.
DRIVE_LIT = 0.20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0, help="ms per heading")
    ap.add_argument("--headings", type=int, default=12)
    ap.add_argument("--bar-width", type=float, default=20.0)
    ap.add_argument("--w-scale", type=float, default=1.0,
                    help="global synaptic weight scale; 2.0 puts central "
                         "neurons at ~33 Hz and wakes EPG, 1.0 leaves them silent")
    ap.add_argument("--drive", type=float, default=DRIVE_LIT,
                    help="mV per step for a fully lit column")
    ap.add_argument("--background", type=float, default=0.5,
                    help="surround luminance; 0 means a black background, "
                         "which leaves the eye almost entirely silent")
    ap.add_argument("--tag", default="default")
    args = ap.parse_args(argv)

    outdir = REPO / "results" / "flywire_heading" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== does the connectome encode where the bar is? ===")
    brain = FlyWireBrain(w_scale=args.w_scale)
    lat = eye.load(REPO / "data" / "flywire" / "column_assignment.csv.gz")
    print("  brain %d neurons / %d edges" % (brain.n, brain.n_edges))

    # map the eye's columnar neurons onto brain rows
    idx = pd.Index(brain.ids).get_indexer(lat.root_id)
    ok = idx >= 0
    inject_types = lat.of_type("L1", "L2", "L3", "L4", "L5", "R7", "R8") & ok
    rows = idx[inject_types]
    az = lat.azimuth_deg[inject_types]
    el = lat.elevation_deg[inject_types]
    print("  injecting into %d columnar neurons (L1-L5, R7, R8)" % len(rows))

    desc = brain.by_super_class("descending")
    side = brain.side()
    desc_r = desc[side[desc] == "right"]
    desc_l = desc[side[desc] == "left"]
    epg = brain.by_cell_type("EPG")
    print("  reading %d descending (R %d / L %d); observing %d EPG"
          % (len(desc), len(desc_r), len(desc_l), len(epg)))

    headings = np.linspace(-180, 180, args.headings, endpoint=False)
    # A non-zero background matters: with a black surround only the ~1000 lit
    # columns fire and nothing propagates past them.  Real photoreceptors have
    # baseline activity and the bar is a CONTRAST on top of it, which is also
    # what lets the rest of the eye contribute.
    scene = eye.bar_scene(0.0, bar_width_deg=args.bar_width,
                          contrast=1.0, background=args.background)

    recs = []
    epg_profiles = []
    t_start = time.perf_counter()
    for h in headings:
        lum = np.zeros(len(az))
        offs = np.linspace(-0.5, 0.5, 5) * eye.ACCEPTANCE_DEG
        for da in offs:
            lum += scene(az + h + da, el)
        lum /= len(offs)

        drive = np.zeros(brain.n, dtype=np.float32)
        drive[rows] = (lum * args.drive).astype(np.float32)

        brain.reset()
        counts, _ = brain.run(args.duration, drive_fn=lambda t: drive)
        hz = counts / (args.duration / 1000.0)

        r, l = hz[desc_r].sum(), hz[desc_l].sum()
        pair = DescendingPair(left=float(l), right=float(r))
        recs.append({"heading_deg": float(h), "n_lit": int((lum > 0).sum()),
                     "d_left": pair.left, "d_right": pair.right,
                     "turn": pair.turn, "forward": pair.forward,
                     "balance": pair.balance,
                     "total_hz": float(hz.sum()), "n_active": int((counts > 0).sum()),
                     "desc_right_hz": float(r), "desc_left_hz": float(l),
                     "steering_lr": float(r - l),
                     "epg_hz": float(hz[epg].sum()) if len(epg) else 0.0})
        epg_profiles.append(hz[epg] if len(epg) else np.zeros(0))
        print("  heading %+7.1f  lit %4d  active %6d  desc R-L %+9.1f Hz"
              % (h, recs[-1]["n_lit"], recs[-1]["n_active"], recs[-1]["steering_lr"]))

    df = pd.DataFrame(recs)
    df.to_csv(outdir / "heading_sweep.csv", index=False)
    el_s = time.perf_counter() - t_start
    print("--- %d headings x %.0f ms in %.0f s ---" % (len(headings), args.duration, el_s))

    # --- does anything actually depend on heading? ---
    print("--- verdict ---")
    lr = df["steering_lr"].to_numpy()
    spread = lr.max() - lr.min()
    print("  descending L-R across headings: %.1f .. %.1f Hz (spread %.1f)"
          % (lr.min(), lr.max(), spread))

    # a steering signal should be odd about the bar's position: turning right
    # when the bar is left and vice versa
    corr = float(np.corrcoef(np.sin(np.deg2rad(df["heading_deg"])), lr)[0, 1]) \
        if spread > 0 else float("nan")
    print("  correlation with sin(heading): %+.3f" % corr)

    epg_arr = np.stack(epg_profiles) if len(epg) else None
    if epg_arr is not None and epg_arr.size and epg_arr.sum() > 0:
        # is EPG activity localised, and does the location move with heading?
        peak = epg_arr.argmax(axis=1)
        print("  EPG peak unit per heading: %s" % peak.tolist())
        print("  EPG distinct peaks: %d of %d headings"
              % (len(np.unique(peak)), len(peak)))
    else:
        print("  EPG: silent")

    prov = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "brain": brain.as_dict(), "eye": lat.as_dict(),
            "duration_ms": args.duration, "bar_width_deg": args.bar_width,
            "drive_lit_mv_per_step": args.drive,
            "n_injected": int(len(rows)),
            "steering_lr_spread_hz": float(spread),
            "corr_with_sin_heading": corr,
            "wall_seconds": el_s}
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=2),
                                            encoding="utf-8")
    print("--- written to %s ---" % outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
