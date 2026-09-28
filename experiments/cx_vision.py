"""Put the eyes back: lamina -> ... -> central complex -> PFL, in one network.

The 532-neuron subcircuit steers but is blind -- all 532 are cb_intrinsic and
not one of the 6,199 lamina cells is present.  Both it and the P reference
stall near 0.80 reach, entirely on collisions, and an oracle handed
ground-truth obstacles goes to 0.938 with zero collisions.  So the missing
15 points are the eyes, and the eyes have to go back in.

The cut is the same mechanical one as before -- forward BFS from the lamina,
backward BFS from the 10 central-complex families, keep the intersection --
which is why it lands at 66,601 neurons and 1.23M edges at 4 hops: vision is
four synapses from the central complex, and there is no shortcut.

That is 125x the subcircuit, back in the size regime that oscillated.  It is
no longer a stability problem: the oscillation was spectral radius (1135
unrescaled), and the same rho -> 0.9 rescaling that made the 532 work applies
unchanged at any size.  It is a RUNTIME problem now, so runtime is measured
first and the closed loop only runs if the number is survivable.

Inputs, both arriving at their own anatomical address:
    scene      6,199 lamina cells, obstacles as silhouettes
    cue        FB5AB left/right, magnitude pinned so only the ratio varies
Output:
    PFL right-minus-left, exactly as in the 532.
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
import scipy.sparse.linalg as spl
import torch

import sensors.flywire_eye as eye
from core.subnetwork import _reach

from cx_subcircuit import FAMILIES, READ_OUT, RHO_TARGET

CUE_TYPES = ("FB5AB",)
HOPS = 4

# Spectral radius, chosen by SETTLING TIME rather than by stability alone.
#
# A recurrent network has an effective time constant of tau/(1-rho), so the
# 0.9 that `cx_subcircuit` uses turns a 20 ms membrane into a 200 ms circuit.
# Measured step response of this network, target stepped hard left:
#
#   rho   effective tau   settles in   value at 100 ms (as % of final)
#   0.90      200 ms        695 ms          -105%      <- WRONG SIGN
#   0.70       67 ms        195 ms          +119%
#   0.50       40 ms         85 ms           +97%
#   0.30       29 ms         95 ms           +93%
#
# The control cycle is 100 ms.  At rho 0.9 the readout is sampled while the
# transient is still swinging through the opposite sign, so the command is
# the negative of what the circuit eventually computes.  That, and nothing
# else, is why the closed loop inverted: standing still the correlation with
# target bearing is +0.95, and flying it was -0.50.
#
# At rho 0.5 the circuit settles inside its own control cycle and the
# inversion disappears: +0.959 standing, +0.971 flying.
#
# ponytail: one global factor, relative wiring untouched.  The principled
# alternative is to match the control period to the circuit instead, but
# 100 ms is the fly own cycle and the Stage 0 calibration.
RHO_VISION = 0.50


def build(hops: int = HOPS, cache: bool = True):
    """Lamina-to-CX subnetwork, spectrally rescaled.  Cached: the BFS plus the
    eigenvalue is a few minutes and nothing about it changes between runs."""
    from core.flywire_rate import FlyWireRate
    from core.malecns import load_malecns

    ids, out, ann, _ = load_malecns(w_scale=1.0, symmetrise=True)
    ct = ann["cell_type"].astype(str).to_numpy()
    cx = np.flatnonzero([any(c.startswith(f) for f in FAMILIES) for c in ct])
    lat = eye.load_malecns_eye(ann)
    rows = pd.Index(ids).get_indexer(
        lat.root_id[lat.of_type(*eye.MALECNS_INJECT_TYPES)])
    lam = rows[rows >= 0]

    cf = REPO / ("results/cx_vision_idx_h%d.npy" % hops)
    if cache and cf.exists():
        idx = np.load(cf)
    else:
        t = time.perf_counter()
        keep = _reach(out, lam, hops) & _reach(out.T.tocsr(), cx, hops)
        keep[lam] = True
        keep[cx] = True
        idx = np.flatnonzero(keep)
        np.save(cf, idx)
        print("  BFS %.0f s" % (time.perf_counter() - t), flush=True)

    sub = out[idx][:, idx].tocsr()
    rf = REPO / ("results/cx_vision_rho_h%d.npy" % hops)
    if cache and rf.exists():
        rho = float(np.load(rf))
    else:
        t = time.perf_counter()
        rho = float(np.abs(spl.eigs(sub.astype(np.float64), k=1,
                                    return_eigenvectors=False)[0]))
        np.save(rf, np.array(rho))
        print("  eigs %.0f s" % (time.perf_counter() - t), flush=True)
    sub = (sub * (RHO_VISION / rho)).tocsr()

    sann = ann.iloc[idx].copy()
    # The index IS root_id and `load_malecns_eye` keys the retinotopic
    # lattice off it.  reset_index(drop=True) replaced it with 0..N-1,
    # so 6,199 lamina cells matched 1,779 unrelated neurons and every
    # viewing direction landed on the wrong cell.  That is what made
    # the two eyes report identical -137..+137 fields and the readout
    # answer with the SAME sign to left and right stimulation.
    net = FlyWireRate(out_csr=sub, ids=ids[idx], ann=sann)
    return net, sann, sub, rho, len(np.intersect1d(lam, idx)), len(
        np.intersect1d(cx, idx))


class VisionSub:
    """Scene + cue in, PFL turn out.  One 100 ms control cycle per call."""

    def __init__(self, net, ann, fixed_strength=0.1, dt_ms=5.0,
                 steps_per_cycle=20, readout='descending'):
        from sensors.connectome_input import ConnectomeInput
        self.net = net
        self.inp = ConnectomeInput(net, ann, cue_types=CUE_TYPES)
        self.inp.fixed_strength = fixed_strength
        self.dt_ms, self.steps = dt_ms, steps_per_cycle
        ct = ann["cell_type"].astype(str).to_numpy()
        side = ann["side"].to_numpy(dtype="<U16")
        # WHERE the turn is read decides whether vision is visible at all.
        # A fly does not avoid obstacles through the central complex: the
        # looming pathway runs lamina -> LC/LPLC -> descending neurons and
        # never touches PFL.  Reading only the 36 PFL cells therefore
        # cannot see an obstacle no matter how the eyes are driven -- and
        # this cut already contains LPLC2 185/185, LPLC1 134/134, LC16
        # 182/182, LC11 143/143, GF 17 and 732 descending neurons.  The
        # two streams, approach and avoidance, meet at the descending
        # neurons, so that is where both can be read at once.
        if readout == 'descending':
            m = ann["super_class"].to_numpy(dtype="<U32") == "descending_neuron"
        else:
            m = np.array([any(c.startswith(p) for p in READ_OUT)
                          for c in ct])
        self.out_l = np.flatnonzero(m & (side == "left"))
        self.out_r = np.flatnonzero(m & (side == "right"))
        self.v = None
        self.baseline = 0.0

    def reset(self):
        self.v = self.net.init_state(1)
        self.inp.reset()          # the held goal is episode state, not global

    def turn(self, world, position, heading):
        from core.flywire_rate import activity
        if self.v is None:
            self.reset()
        d, _ = self.inp.drive(world, position, heading)
        with torch.no_grad():
            for _ in range(self.steps):
                self.v = self.net.step(self.v, d, self.dt_ms)
            r = activity(self.v).numpy().ravel()
        return (float(r[self.out_r].mean() - r[self.out_l].mean())
                - self.baseline)

    def calibrate(self, steps=40):
        """Readout with the target dead ahead and nothing in the way.  The DC
        term is what killed three earlier attempts; it is measured once, on an
        empty scene, never fitted per episode."""
        from environment.target_world import TargetWorld
        w = TargetWorld(target=np.array([10.0, 0.0, 2.0]), obstacles=[])
        p = np.array([0.0, 0.0, 2.0])
        self.baseline = 0.0
        self.reset()
        t = 0.0
        for _ in range(steps):
            t = self.turn(w, p, 0.0)
        self.baseline = t
        return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=HOPS)
    ap.add_argument("--cycles", type=int, default=10)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    net, ann, sub, rho, n_lam, n_cx = build(args.hops)
    print("%d neurons, %d edges, rho %.1f -> %.2f  (%.0f s)"
          % (net.n, sub.nnz, rho, RHO_VISION, time.perf_counter() - t0))
    print("  lamina inside  %d" % n_lam)
    print("  CX families in %d" % n_cx, flush=True)
    assert n_lam > 1000, "the eyes did not survive the cut"

    vs = VisionSub(net, ann)
    print("  cue address %s -> L%d/R%d; PFL L%d/R%d"
          % (CUE_TYPES, len(vs.inp.orn_l), len(vs.inp.orn_r),
             len(vs.out_l), len(vs.out_r)), flush=True)
    assert len(vs.inp.orn_l) and len(vs.inp.orn_r), "cue has nowhere to land"
    assert len(vs.out_l) and len(vs.out_r), "no PFL to read"

    # --- runtime, before anything else ---------------------------------
    from environment.target_world import Obstacle, TargetWorld
    w = TargetWorld(target=np.array([12.0, 3.0, 2.0]),
                    obstacles=[Obstacle(x=6.0, y=1.0, radius=1.0, z=0.0,
                                        height=6.0)])
    p = np.array([0.0, 0.0, 2.0])
    vs.reset()
    vs.turn(w, p, 0.0)                       # warm up, not timed
    t = time.perf_counter()
    for _ in range(args.cycles):
        vs.turn(w, p, 0.0)
    per = (time.perf_counter() - t) / args.cycles
    print("\ncontrol cycle %.1f ms  ->  %.1f s per 120-step episode"
          % (1e3 * per, 120 * per), flush=True)

    # --- does the readout see the cue, AT SEVERAL DISTANCES -------------
    # one distance cannot reveal a DC term that scales with cue magnitude,
    # and that is exactly what wrecked three earlier attempts
    vs.calibrate()
    print("\nPFL R-L by target bearing, at three distances (empty scene):")
    print("%8s %14s %14s %14s" % ("bearing", "4 m", "8 m", "16 m"))
    tab = {}
    for b in (-40.0, -20.0, 0.0, 20.0, 40.0):
        row = []
        for dist in (4.0, 8.0, 16.0):
            th = np.radians(b)
            wt = TargetWorld(
                target=np.array([dist * np.cos(th), dist * np.sin(th), 2.0]),
                obstacles=[])
            vs.reset()
            for _ in range(40):
                val = vs.turn(wt, p, 0.0)
            row.append(val)
        tab[b] = row
        print("%+8.1f %14.6e %14.6e %14.6e" % (b, row[0], row[1], row[2]),
              flush=True)

    arr = np.array([tab[b] for b in sorted(tab)])
    for j, dist in enumerate((4.0, 8.0, 16.0)):
        col = arr[:, j]
        rng = col.max() - col.min()
        mono = bool(np.all(np.diff(col) > 0) or np.all(np.diff(col) < 0))
        print("  %4.0f m: monotonic %-5s range %.3e  offset/range %.3f"
              % (dist, mono, rng, abs(col.mean()) / rng if rng > 0 else np.nan))
    spread = np.abs(arr.mean(axis=0)).max() - np.abs(arr.mean(axis=0)).min()
    print("  DC drift across distance %.3e vs bearing range %.3e"
          % (spread, (arr.max() - arr.min())))
    print("  -> the direction signal %s the distance artefact"
          % ("survives" if (arr.max() - arr.min()) > 2 * spread else "is SWAMPED BY"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
