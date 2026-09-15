"""Execute the OFFICIAL Westeinde model notebook cells and dump the result.

This file runs the source authors' code, it does not reimplement it.  Cells
0 (imports), 3 (functions), 5 (setup) and 7 (computation) of

    reference/westeinde_official/Westeinde et al Model.ipynb

are exec'd verbatim, with exactly ONE controlled modification: the grid
resolution in cell 5 can be coarsened, because the full official grid
(37 goals x 6 scalars x 361 HDs x 1000 cells) needs several GB of RAM and this
machine does not have it.  Every substitution that is applied is printed, so
the deviation is auditable.

The arithmetic is untouched.  Coarsening the grid DOES change the numbers,
because the source normalises by the min/max over the whole grid -- that batch
dependence is a property of the source model, and it is precisely why our own
core has to freeze its normalisation constants.

Usage:
    python reference/run_official.py --goal-step 30 --hd-step 4 --out ref_coarse.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "westeinde_official" / "model.ipynb"

# cells to execute, in order: imports, function defs, setup, computation
CELLS = (0, 3, 5, 7)


def load_cells(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(nb["cells"][i]["source"]) for i in CELLS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-step", type=int, default=30,
                    help="official value is 10")
    ap.add_argument("--hd-step", type=int, default=4,
                    help="official value is 1")
    ap.add_argument("--num-cells", type=int, default=1000,
                    help="official value is 1000; do not change lightly, the "
                         "notebook hardcodes 1000 in the DNa02 step")
    ap.add_argument("--nonlinearity", default="ELU1")
    ap.add_argument("--out", default="ref_coarse.npz")
    args = ap.parse_args()

    sources = load_cells(NOTEBOOK)

    # ---- the single controlled modification, applied to cell 5 -----------
    subs = []
    if args.goal_step != 10:
        subs.append(("np.arange(-180, 181, 10)",
                     "np.arange(-180, 181, %d)" % args.goal_step))
    if args.hd_step != 1:
        subs.append(("np.arange(-180, 181, 1)",
                     "np.arange(-180, 181, %d)" % args.hd_step))
    if args.num_cells != 1000:
        subs.append(("num_cells = 1000", "num_cells = %d" % args.num_cells))

    # numpy>=1.25 compatibility.  The notebook assigns a shape-(1,) slice into
    # a scalar array element, which newer numpy rejects.  Taking element [0]
    # instead is mathematically identical.  These three lines only fill the
    # *_HD_prefs label arrays, which cell 7 (the model computation) never
    # reads -- they exist for later plots.
    compat = [("hd_deg_open_loop[:, np.argmax(", "hd_deg_open_loop[0, np.argmax(", 3)]

    setup = sources[2]
    print("=== numpy compatibility patches in cell 5 ===")
    for old, new, expected in compat:
        n = setup.count(old)
        if n != expected:
            raise RuntimeError(
                "expected %d occurrences of %r in cell 5, found %d"
                % (expected, old, n))
        setup = setup.replace(old, new)
        print("  %s  ->  %s   (x%d)" % (old, new, n))

    print("=== controlled substitutions in cell 5 ===")
    if not subs:
        print("  (none -- running the official grid verbatim)")
    for old, new in subs:
        n = setup.count(old)
        if n != 1:
            raise RuntimeError(
                "expected exactly one occurrence of %r in cell 5, found %d; "
                "the notebook changed and this script must be re-checked"
                % (old, n))
        setup = setup.replace(old, new)
        print("  %-28s ->  %s" % (old, new))
    sources[2] = setup

    ns: dict = {}
    for src, idx in zip(sources, CELLS):
        print("--- exec cell %d ---" % idx)
        exec(compile(src, "<notebook cell %d>" % idx, "exec"), ns)

    steering = ns["steering"]
    print("=== official model output ===")
    print("  grid: goals=%d scalars=%d HD=%d cells=%d"
          % (len(ns["goal_phase"]), ns["num_scalars"], ns["num_HD"],
             ns["num_cells"]))
    print("  steering shape:", steering.shape,
          " min=%.6f max=%.6f" % (steering.min(), steering.max()))

    gi0 = int(np.argmin(np.abs(ns["goal_phase"])))
    si1 = int(np.argmin(np.abs(ns["S_vals"] - 1.0)))

    out = HERE / args.out
    np.savez_compressed(
        out,
        steering=steering,
        steering_direct=ns["steering_direct"],
        steering_indirect=ns["steering_indirect"],
        Dna02r=ns["Dna02r"], Dna02l=ns["Dna02l"],
        Dna03r=ns["Dna03r"], Dna03l=ns["Dna03l"],
        pfl3_RL_diff=ns["pfl3_RL_diff"],
        pfl2_sum=ns["pfl2_sum"],
        pfl2_bump_amp=ns["pfl2_bump_amp"],
        # one population slice (goal = 0 deg, S = 1) for cell-level checking;
        # the full 4-D activity would make this file ~10 MB
        pfl2_slice=ns["pfl2"][gi0, si1], pfl3r_slice=ns["pfl3r"][gi0, si1],
        pfl3l_slice=ns["pfl3l"][gi0, si1],
        slice_goal_deg=float(ns["goal_phase"][gi0]),
        slice_S=float(ns["S_vals"][si1]),
        goal_phase=ns["goal_phase"],
        S_vals=ns["S_vals"],
        hd_deg=np.squeeze(ns["hd_deg_open_loop"]),
        hd_prefs=np.squeeze(ns["hd_prefs"]),
        num_cells=ns["num_cells"],
        nonlinearity_option=str(ns["nonlinearity_option"]),
    )
    print("  saved ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
