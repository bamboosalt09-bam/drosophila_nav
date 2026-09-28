"""Can the injected addresses even reach the readout in this subgraph?

The closed-loop sweep moved the final distance by less than 0.01 m across a
5x change in heading drive, which is not a weak effect -- it is no effect.
Before tuning anything further, check the graph: hops from each input set to
PFL, and how much drive arrives.  A path that does not exist cannot be
calibrated into existence.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from cx_subcircuit import (CUE_IN, HEAD_IN, READ_OUT, DT_MS, STEPS_PER_CYCLE,
                           Sub, build)


def hops_to(sub, src, dst, max_hops=8):
    """Fewest synapses from any neuron in src to any in dst, or None."""
    seen = np.zeros(sub.shape[0], dtype=bool)
    seen[src] = True
    frontier = seen.copy()
    for k in range(1, max_hops + 1):
        nxt = (sub.T @ frontier.astype(np.float32)) > 0
        frontier = nxt & ~seen
        if not frontier.any():
            return None
        seen |= frontier
        if seen[dst].any():
            return k
    return None


def main() -> int:
    net, cue_a, head_a, out_a, rho, sub = build()
    outs = np.concatenate(out_a)
    print("%d neurons, %d edges" % (net.n, sub.nnz))

    for name, addr in (("FB5AB (cue)", cue_a), ("PFN (heading)", head_a)):
        src = np.concatenate(addr)
        out_deg = int((sub[src] != 0).sum())
        h = hops_to(sub, src, outs)
        print("%-16s n=%4d  out-edges %5d  hops to PFL: %s"
              % (name, len(src), out_deg, h if h else "UNREACHABLE"))

    # who actually drives PFL in this subgraph?
    from core.malecns import load_malecns  # noqa: F401  (ann already in net)
    ct = net.ann["cell_type"].astype(str).to_numpy()
    in_w = np.asarray(np.abs(sub[:, outs]).sum(axis=1)).ravel()
    order = np.argsort(-in_w)[:10]
    print("\ntop presynaptic sources of PFL, by summed |weight|:")
    for i in order:
        if in_w[i] <= 0:
            break
        print("   %-12s %8.4f" % (ct[i], in_w[i]))

    # and does the readout move at all when only heading changes?
    s = Sub(net, cue_a, head_a, out_a, head_gain=1.0)
    from sensors.attraction_cue import AttractionCue
    cue = AttractionCue(0.0, 0.0, 0.0, 10.0, 0.1, True, 2)
    print("\nPFL R-L with the cue held at bearing 0, heading swept:")
    for hd in (-1.2, -0.6, 0.0, 0.6, 1.2):
        s.reset()
        for _ in range(40):
            t = s.turn(cue, hd)
        print("   heading %+5.2f rad -> %+.6e" % (hd, t))

    print("\nPFL R-L with heading held at 0, cue bearing swept:")
    s2 = Sub(net, cue_a, head_a, out_a, head_gain=0.0)
    vals = []
    for b in (-40.0, -20.0, 0.0, 20.0, 40.0):
        s2.reset()
        c = AttractionCue(b, 0.0, 0.0, 10.0, 0.1, True, 2)
        for _ in range(40):
            t = s2.turn(c, 0.0)
        vals.append(t)
        print("   bearing %+6.1f deg -> %+.6e" % (b, t))
    rng = max(vals) - min(vals)
    print("   range %.3e, offset/range %.3f"
          % (rng, abs(np.mean(vals)) / rng if rng > 0 else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
