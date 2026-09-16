"""Pilot: can the training loop move a connectome network toward steering?

Small on purpose.  The point is not a result -- it is to find out whether the
loop runs, whether the loss falls, and what one condition costs, before any of
this is repeated over seeds on a machine that is not this laptop.

What is trained: per-cell-type tau, bias and gain, 174 numbers.  Wiring and
synapse counts are fixed.  What is measured is the TRAJECTORY, not just the
endpoint -- loss curve, distance travelled in parameter space, and the EPG
profile before and after, because EPG forming a bump would be the sign that
the connectome specified where the computation lives and training only
supplied the missing gains.

Usage:
    python experiments/flywire_train_pilot.py --iters 30
    python experiments/flywire_train_pilot.py --wiring degree_preserving
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
import torch

import sensors.flywire_eye as eye
from core.flywire_brain import load_connectome
from core.flywire_rate import FlyWireRate
from core import null_wiring

W_SCALE = 0.01      # measured responsive regime; see flywire_rate.demo
DT_MS = 5.0         # 1.1% deviation from dt = 0.1 ms, 39x fewer steps
DURATION_MS = 200.0
GRAD_MS = 40.0      # backprop only through the last 40 ms; the readout is a
                    # steady-state rate, so the transient carries little gradient


def build_drive(net, lat, headings, background, drive_lit):
    """(n_neurons, n_headings) visual drive, one column per bar position."""
    idx = pd.Index(net.ids).get_indexer(lat.root_id)
    sel = lat.of_type("L1", "L2", "L3", "L4", "L5", "R7", "R8") & (idx >= 0)
    rows, az, el = idx[sel], lat.azimuth_deg[sel], lat.elevation_deg[sel]
    scene = eye.bar_scene(0.0, 20.0, contrast=1.0, background=background)
    d = torch.zeros(net.n, len(headings))
    for k, h in enumerate(headings):
        lum = np.zeros(len(az))
        for da in np.linspace(-0.5, 0.5, 5) * eye.ACCEPTANCE_DEG:
            lum += scene(az + h + da, el)
        d[rows, k] = torch.from_numpy((lum / 5.0 * drive_lit).astype(np.float32))
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", default="connectome",
                    choices=("connectome", "degree_preserving", "random_sparse"))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headings", type=int, default=8)
    ap.add_argument("--by-side", action="store_true",
                    help="separate parameters for left and right; without it "
                         "training cannot address a left-right asymmetry")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    tag = args.tag or "%s_seed%d" % (args.wiring, args.seed)
    outdir = REPO / "results" / "flywire_train" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    ids, out, ann, _ = load_connectome(w_scale=W_SCALE)
    if args.wiring == "degree_preserving":
        out = null_wiring.degree_preserving(out, seed=args.seed)
    elif args.wiring == "random_sparse":
        out = null_wiring.random_sparse(out, seed=args.seed)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann, by_side=args.by_side)
    print("=== pilot: %s, seed %d ===" % (args.wiring, args.seed))
    print("  %d neurons, %d edges, %d types, %d trainable"
          % (net.n, net.n_edges, len(net.types), net.n_trainable()))

    lat = eye.load(REPO / "data" / "flywire" / "column_assignment.csv.gz")
    headings = np.linspace(-150, 150, args.headings)
    drive = build_drive(net, lat, headings, background=0.5, drive_lit=0.20)

    side = ann["side"].astype(str).to_numpy()
    desc = np.flatnonzero(ann["super_class"].eq("descending").to_numpy())
    dr = torch.from_numpy(desc[side[desc] == "right"])
    dl = torch.from_numpy(desc[side[desc] == "left"])
    epg = torch.from_numpy(
        np.flatnonzero(ann["cell_type"].eq("EPG").to_numpy()))
    print("  descending R %d / L %d, observing %d EPG" % (len(dr), len(dl),
                                                          len(epg)))

    # Target: turn TOWARD the bar.  Odd in bar azimuth and zero straight
    # ahead, which is exactly what the untrained network fails to do.
    target = torch.from_numpy(
        (-np.sin(np.deg2rad(headings))).astype(np.float32))

    def steering(net):
        r, _ = net.run(drive, DURATION_MS, DT_MS, grad_ms=GRAD_MS)
        return r[dr].mean(0) - r[dl].mean(0)

    def epg_profile(net):
        with torch.no_grad():
            r, _ = net.run(drive, DURATION_MS, DT_MS)
        return r[epg].numpy()

    theta0 = torch.cat([p.detach().flatten().clone()
                        for p in net.parameters()])
    epg_before = epg_profile(net)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    hist = []
    t0 = time.perf_counter()
    for it in range(args.iters):
        opt.zero_grad()
        s = steering(net)
        # scale-free: match the SHAPE of the target, not its magnitude, so the
        # loss cannot be reduced by simply shrinking every rate
        s_n = s / (s.abs().max() + 1e-6)
        loss = torch.nn.functional.mse_loss(s_n, target)
        loss.backward()
        opt.step()
        theta = torch.cat([p.detach().flatten() for p in net.parameters()])
        dist = float((theta - theta0).norm())
        corr = float(np.corrcoef(s_n.detach().numpy(), target.numpy())[0, 1])
        hist.append({"iter": it, "loss": float(loss), "param_dist": dist,
                     "corr_with_target": corr,
                     "steering_span": float(s.max() - s.min())})
        if it % max(1, args.iters // 10) == 0 or it == args.iters - 1:
            print("  it %3d  loss %.4f  |dtheta| %.4f  corr %+.3f  (%.0f s)"
                  % (it, float(loss), dist, corr, time.perf_counter() - t0))

    epg_after = epg_profile(net)
    df = pd.DataFrame(hist)
    df.to_csv(outdir / "trajectory.csv", index=False)
    np.savez(outdir / "epg.npz", before=epg_before, after=epg_after,
             headings=headings)

    def localisation(p):
        """peak-to-mean ratio: >1 means activity is concentrated, 1 is flat."""
        m = p.mean(axis=0)
        return float(p.max(axis=0).mean() / (m.mean() + 1e-9)) if p.size else 0.0

    print("--- %d iters in %.0f s (%.1f s/iter) ---"
          % (args.iters, time.perf_counter() - t0,
             (time.perf_counter() - t0) / args.iters))
    print("  loss %.4f -> %.4f   corr %+.3f -> %+.3f   |dtheta| %.4f"
          % (hist[0]["loss"], hist[-1]["loss"], hist[0]["corr_with_target"],
             hist[-1]["corr_with_target"], hist[-1]["param_dist"]))
    print("  EPG peak/mean %.2f -> %.2f   active units %d -> %d of %d"
          % (localisation(epg_before), localisation(epg_after),
             int((epg_before.max(axis=1) > 1e-6).sum()),
             int((epg_after.max(axis=1) > 1e-6).sum()), len(epg)))

    (outdir / "provenance.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wiring": args.wiring, "seed": args.seed, "by_side": args.by_side, "iters": args.iters,
        "lr": args.lr, "w_scale": W_SCALE, "dt_ms": DT_MS,
        "duration_ms": DURATION_MS, "grad_ms": GRAD_MS, "headings_deg": headings.tolist(),
        "net": net.as_dict(),
        "loss_first": hist[0]["loss"], "loss_last": hist[-1]["loss"],
        "param_distance": hist[-1]["param_dist"],
        "seconds_per_iter": (time.perf_counter() - t0) / args.iters,
    }, indent=2), encoding="utf-8")
    print("--- written to %s ---" % outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
