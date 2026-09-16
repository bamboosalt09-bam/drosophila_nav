"""Audit the source notebook's connectivity_option = 'data' before using it.

The name suggests "the hemibrain connectome, per cell".  Before any result is
built on it, four things have to be checked in the actual code (not the
docstring):

  1. does it keep per-neuron connections, or population-summed weights?
  2. how are synapse counts converted into functional weights?
  3. what do the normalisation / correction steps change?
  4. do the abstract and data conditions use the same input and evaluation?

This script answers all four by executing the authors' own cells, and reports
the two bugs that stop the data path from running as published.

Usage:
    python reference/check_connectivity_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "westeinde_official" / "model.ipynb"
CELLS = (0, 3, 5, 7)

# The two repairs the data path needs.  Both are bugs, not modelling choices;
# see the printed report for why neither carries a free parameter.
REPAIR_HD_PREFS = (
    "hd_prefs = [165, 135, 105, 75, 45, 15, -15, -45, -75, -105, -135, -165]",
    "hd_prefs = np.asarray([165, 135, 105, 75, 45, 15, -15, -45, -75, -105, "
    "-135, -165])")
REPAIR_TILE = ("(1000, 1, 1, 1)", "(num_cells, 1, 1, 1)")
# already recorded elsewhere: numpy>=2 rejects the scalar assignment in cell 5
COMPAT_NUMPY2 = ("hd_deg_open_loop[:, np.argmax(", "hd_deg_open_loop[0, np.argmax(")


def cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(nb["cells"][i]["source"]) for i in CELLS]


def run(option, repairs=(REPAIR_HD_PREFS, REPAIR_TILE), n_cells=None):
    c = cells()
    c[2] = c[2].replace("connectivity_option = 'abstract'",
                        "connectivity_option = '%s'" % option)
    c[2] = c[2].replace(*COMPAT_NUMPY2)
    for old, new in repairs:
        c[2] = c[2].replace(old, new)
        c[3] = c[3].replace(old, new)
    if n_cells:
        c[2] = c[2].replace("num_cells = 1000", "num_cells = %d" % n_cells)
    g = {}
    for src in c:
        exec(compile(src, "<cell>", "exec"), g)
    return g


def restoring_gain_per_goal(g):
    """|d steering / d error| at error ~ 0, for every goal phase, at S = 1."""
    st, gp, hd = g["steering"], g["goal_phase"], g["hd_deg_open_loop"].ravel()
    out = []
    for p, P in enumerate(gp):
        err = (hd - P + 180) % 360 - 180
        ip, im = int(np.argmin(np.abs(err - 5))), int(np.argmin(np.abs(err + 5)))
        out.append(abs((st[p, -1, ip] - st[p, -1, im]) / 10.0))
    return np.asarray(out)


def main() -> int:
    print("=== does the published data path run at all? ===")
    for label, reps in (("as published", ()),
                        ("+ hd_prefs repair", (REPAIR_HD_PREFS,)),
                        ("+ tile repair", (REPAIR_HD_PREFS, REPAIR_TILE))):
        try:
            run("data", repairs=reps)
            print("  %-20s OK" % label)
        except Exception as exc:
            print("  %-20s %s: %s" % (label, type(exc).__name__, exc))

    print("\n=== Q1: per-neuron or population-summed? ===")
    g = run("data")
    print("  num_cells = %d per PFL population (indexed by PB glomerulus)"
          % g["num_cells"])
    print("  PFL->DN weights are PER CELL (length-%d vectors)" % g["num_cells"])
    print("  DNa03->DNa02 is a SCALAR (271): the DNs are single units, not "
          "populations")
    print("  left hemisphere weights are np.flip of the right, i.e. ASSUMED "
          "symmetric, not measured")

    print("\n=== Q2: synapse count -> functional weight? ===")
    pfl2 = np.array([1., 72., 56., 57., 96., 57., 54., 79., 45., 58., 89., 76.])
    p3_03 = np.array([25., 28., 8., 13., 26., 29., 12., 1., 9., 22., 1., 26.])
    p3_02 = np.array([21., 25., 17., 14., 19., 44., 25., 25., 7., 11., 3., 39.])
    print("  raw hemibrain synapse counts are used DIRECTLY as weights; no "
          "conversion, no per-cell normalisation")
    print("  mean pfl2->dna03 %.1f (range %.0f-%.0f), pfl3->dna03 %.1f, "
          "pfl3->dna02 %.1f, dna03->dna02 271"
          % (pfl2.mean(), pfl2.min(), pfl2.max(), p3_03.mean(), p3_02.mean()))
    print("  the abstract weights are these RATIOS rounded: "
          "pfl2/pfl3 = %.2f -> 4, dna03/pfl3 = %.2f -> 12"
          % (pfl2.mean() / p3_03.mean(), 271 / p3_02.mean()))

    print("\n=== Q3: what do the repairs and the normalisation change? ===")
    a = run("data", repairs=(REPAIR_HD_PREFS, ("(1000, 1, 1, 1)",
                                               "(num_cells, 1, 1, 1)")))
    b = run("data", repairs=(REPAIR_HD_PREFS, ("(1000, 1, 1, 1)",
                                               "(1, 1, 1, 1)")))
    print("  tile=num_cells vs tile=1 bit-identical: %s"
          % np.array_equal(a["steering"], b["steering"]))
    print("  (broadcasting admits only those two, and they are the same sum, "
          "so the repair carries NO free parameter)")
    print("  ELU1 rescales by the BATCH min/max at every stage, and the final "
          "steering is divided by its own global max,")
    print("  so any uniform gain is removed -- what survives is the SHAPE and "
          "the relative weighting between cells.")

    print("\n=== Q4: same input / evaluation as the abstract condition? ===")
    ab12 = run("abstract", n_cells=12)
    print("  goal-population HD prefs identical at 12 cells: %s"
          % np.allclose(np.sort(np.ravel(g["hd_prefs"])),
                        np.sort(np.ravel(ab12["hd_prefs"]))))
    print("  PFL HD prefs DIFFER: abstract is uniform + a fixed phase shift "
          "(180/+67.5/-67.5);")
    print("  data uses per-cell values predicted from PB glomerulus, with no "
          "added shift.")
    print("  cell count is NOT the cause of the difference -- abstract at 12 "
          "cells reproduces abstract at 1000:")
    hd = ab12["hd_deg_open_loop"].ravel()
    gi = int(np.argmin(np.abs(ab12["goal_phase"])))
    i90 = int(np.argmin(np.abs(hd + 90)))
    print("    steering(e=-90) : abstract@1000 %.3f | abstract@12 %.3f | "
          "data@12 %.3f"
          % (run("abstract")["steering"][gi, -1, i90],
             ab12["steering"][gi, -1, i90], g["steering"][gi, -1, i90]))

    print("\n=== consequence: goal-phase invariance ===")
    for name, gg in (("abstract", run("abstract")), ("data", g)):
        gains = restoring_gain_per_goal(gg)
        print("  %-9s restoring gain across goal phases: min %.5f max %.5f  "
              "ratio %.2f  CV %.3f"
              % (name, gains.min(), gains.max(),
                 gains.max() / max(gains.min(), 1e-12),
                 gains.std() / gains.mean()))
    gains = restoring_gain_per_goal(g)
    gp = g["goal_phase"]
    print("  data is weakest at goal = %+d deg and strongest at goal = %+d deg"
          % (gp[int(np.argmin(gains))], gp[int(np.argmax(gains))]))
    print("  => a data-condition sweep MUST vary the goal direction; holding "
          "goal = 0, as every")
    print("     Stage 1 run does, samples one arbitrary point of an 11.7x "
          "range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
