"""Stage 1, step 3: analyse the sweep (handoff doc sections 29, 30, 47).

Reads results/stage1_sweep/<tag>/sweep_trials.csv and produces the phase maps.

The one thing this script exists to get right
---------------------------------------------
Averaging success over all trials at a given r_max is WRONG, and the raw sweep
output shows why.  At small r_max the hard initial headings are excluded as
BODY_INFEASIBLE, so the surviving trials are the easy ones: at
r_max = 0.002 R_max only |e0| = 30 and 45 deg remain, against 8 distinct
magnitudes at r_max >= 0.01.  Comparing those averages compares different
tasks and manufactures a result.

So every comparison here is PAIRED (section 47): restricted to the initial
headings that are feasible in every body condition being compared, and to the
headings where the ideal body itself succeeded, so that Stage 0 finding F1 is
not being re-measured.

Usage:
    python experiments/stage1_sweep_analysis.py
    python experiments/stage1_sweep_analysis.py --tag quick
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOT_COMPARABLE = ("body_infeasible", "baseline_failure", "rescued_by_body")


def paired_subset(df: pd.DataFrame, axis: str, min_value: float | None = None):
    """Trials usable for a fair comparison along `axis`.

    Keeps only the initial headings that are feasible at EVERY level of the
    axis, and only the headings the ideal body itself handled.
    """
    d = df if min_value is None else df[df[axis] >= min_value]
    common = None
    for _, g in d.groupby(axis):
        s = set(g.loc[g["feasible"], "e0_deg"])
        common = s if common is None else (common & s)
    common = common or set()
    return d[d["e0_deg"].isin(common) & d["baseline_success"]], sorted(common)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="analyse the Stage 1 sweep")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--r-min", type=float, default=0.01,
                    help="lowest r_max ratio to include in paired comparisons")
    ap.add_argument("--compare", default=None,
                    help="another tag to diff against, cell by cell")
    args = ap.parse_args(argv)

    d = REPO / "results" / "stage1_sweep" / args.tag
    df = pd.read_csv(d / "sweep_trials.csv")
    print("=== Stage 1 step 3: sweep analysis (tag: %s) ===" % args.tag)
    print("  %d trials" % len(df))

    # ---- how severe is the selection effect? ----------------------------
    print("--- selection check: which headings survive at each r_max ---")
    sel_rows = []
    for rr, g in df.groupby("r_max_ratio"):
        att = g[~g["outcome"].isin(NOT_COMPARABLE)]
        mags = sorted(set(att["e0_deg"].abs()))
        sel_rows.append({"r_max_ratio": rr, "n_trials": len(g),
                         "n_excluded": len(g) - len(att),
                         "n_distinct_abs_e0": len(mags),
                         "max_abs_e0_deg": max(mags) if mags else np.nan})
        print("  r_max=%-7.3f excluded %4d/%4d, %d distinct |e0| up to %s deg"
              % (rr, len(g) - len(att), len(g), len(mags),
                 ("%.0f" % max(mags)) if mags else "-"))
    pd.DataFrame(sel_rows).to_csv(d / "selection_check.csv", index=False)

    # ---- paired comparison ----------------------------------------------
    sub, common = paired_subset(df, "r_max_ratio", args.r_min)
    print("--- paired sample: r_max >= %g, %d initial headings common to all ---"
          % (args.r_min, len(common)))
    print("  %s" % common)

    summary = {}
    for axis in ("r_max_ratio", "alpha_ratio", "tau_ratio"):
        agg = sub.groupby(axis)["success"].agg(["mean", "count"])
        summary[axis] = {float(k): float(v) for k, v in agg["mean"].items()}
        print("  %-12s %s" % (axis, "  ".join("%g:%.3f" % (k, v)
                                              for k, v in agg["mean"].items())))
    sub.groupby(["r_max_ratio", "alpha_ratio", "tau_ratio"])["success"].mean() \
       .rename("success_rate").reset_index().to_csv(d / "paired_grid.csv",
                                                    index=False)

    # direction of each axis: does more capability help or hurt?
    print("--- direction of each axis (section 30) ---")
    direction = {}
    for axis, more_is in (("r_max_ratio", "faster body"),
                          ("alpha_ratio", "stronger acceleration"),
                          ("tau_ratio", "more lag")):
        vals = summary[axis]
        ks = sorted(vals)
        lo, hi = vals[ks[0]], vals[ks[-1]]
        sign = "helps" if hi > lo + 0.02 else ("hurts" if hi < lo - 0.02
                                               else "no effect")
        direction[axis] = {"low": lo, "high": hi, "more_capability": sign}
        print("  %-12s %-24s %.3f -> %.3f   (%s)"
              % (axis, more_is, lo, hi, sign))

    # ---- rescue map ------------------------------------------------------
    resc = df[~df["baseline_success"].astype(bool)]
    rescue_rate = float((resc["outcome"] == "rescued_by_body").mean()) if len(resc) else np.nan
    print("--- rescue (ideal body failed, constrained body succeeded) ---")
    print("  overall: %.1f%% of %d trials" % (100 * rescue_rate, len(resc)))

    # ---- figures ---------------------------------------------------------
    def heat(ax, piv, title, xlabel, ylabel, cmap="RdYlGn"):
        im = ax.imshow(piv.values, origin="lower", vmin=0, vmax=1, cmap=cmap,
                       aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(["%g" % c for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(["%g" % r for r in piv.index])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        for (yi, xi), v in np.ndenumerate(piv.values):
            if np.isfinite(v):
                ax.text(xi, yi, "%.2f" % v, ha="center", va="center", fontsize=7)
        return im

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    specs = [("tau_ratio", "r_max_ratio", "alpha_max"),
             ("tau_ratio", "alpha_ratio", "r_max"),
             ("alpha_ratio", "r_max_ratio", "tau_r")]
    for ax, (xa, ya, marg) in zip(axes, specs):
        piv = sub.pivot_table(index=ya, columns=xa, values="success",
                              aggfunc="mean")
        im = heat(ax, piv, "marginalised over %s" % marg,
                  xa.replace("_ratio", ""), ya.replace("_ratio", ""))
    fig.colorbar(im, ax=axes, shrink=0.85, label="success rate (paired sample)")
    fig.suptitle("Stage 1: heading-recovery success of the FIXED fly core vs body limits\n"
                 "paired over %d initial headings; infeasible and baseline-failure "
                 "trials excluded" % len(common), fontsize=10)
    fig.savefig(d / "fig3_paired_phase_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for ax, (axis, label) in zip(axes, [
            ("r_max_ratio", "rate limit  r_max / R_max"),
            ("alpha_ratio", "acceleration cap  alpha_max / (R_max/T)"),
            ("tau_ratio", "response lag  tau_r / T_core")]):
        agg = sub.groupby(axis)["success"].mean()
        ax.plot(agg.index, agg.values, "o-", lw=2)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate")
    fig.suptitle("Stage 1: each axis separately (paired sample). "
                 "A FASTER body is worse; more lag is worse; "
                 "more acceleration helps up to a point.", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "fig4_axis_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if len(resc):
        piv = resc.pivot_table(index="r_max_ratio", columns="tau_ratio",
                               values="success", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        im = heat(ax, piv, "fraction rescued", "tau_r / T_core", "r_max / R_max",
                  cmap="PuBuGn")
        fig.colorbar(im, ax=ax, label="rescued fraction")
        fig.suptitle("Stage 1: body limits RESCUE the core where the ideal body fails\n"
                     "(initial headings +-90 deg, Stage 0 finding F1)", fontsize=10)
        fig.tight_layout()
        fig.savefig(d / "fig5_rescue_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- section 29: phase map by DOMINANT OUTCOME, not success rate ----
    # The success-rate maps say how often it worked; this says what happened.
    codes = ["success", "success_near_infeasible", "slow_but_stable",
             "saturation_dominated", "overshoot_dominant",
             "persistent_oscillation", "controller_instability",
             "failure_other", "rescued_by_body", "baseline_failure",
             "body_infeasible"]
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i / 20.0) for i, c in enumerate(codes)}
    dom = (df.groupby(["r_max_ratio", "tau_ratio"])["outcome"]
             .agg(lambda x: x.value_counts().idxmax()).reset_index())
    rows = sorted(df["r_max_ratio"].unique())
    cols = sorted(df["tau_ratio"].unique())
    grid_i = np.full((len(rows), len(cols)), np.nan)
    for _, r in dom.iterrows():
        grid_i[rows.index(r["r_max_ratio"]), cols.index(r["tau_ratio"])] =             codes.index(r["outcome"])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.imshow(grid_i, origin="lower", cmap=cmap, vmin=0, vmax=19, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["%g" % c for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["%g" % r for r in rows])
    ax.set_xlabel("tau_r / T_core")
    ax.set_ylabel("r_max / R_max")
    present = [c for c in codes if c in set(dom["outcome"])]
    ax.legend(handles=[plt.Line2D([0], [0], marker="s", ls="", color=colors[c],
                                  label=c) for c in present],
              fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Stage 1 / section 29: dominant outcome per body condition"
                 "  (tag: %s, over alpha_max)" % args.tag, fontsize=10)
    fig.savefig(d / "fig6_outcome_phase_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    dom.to_csv(d / "dominant_outcome.csv", index=False)
    print("--- dominant outcome per (r_max, tau) written ---")

    # ---- optional cell-by-cell comparison with another sweep ------------
    comparison = None
    if args.compare:
        other = pd.read_csv(REPO / "results" / "stage1_sweep" / args.compare
                            / "sweep_trials.csv")
        o_sub, _ = paired_subset(other, "r_max_ratio", args.r_min)
        keys = ["r_max_ratio", "alpha_ratio", "tau_ratio"]
        a = sub.groupby(keys)["success"].mean()
        b = o_sub.groupby(keys)["success"].mean()
        j = pd.concat([a.rename(args.tag), b.rename(args.compare)], axis=1).dropna()
        j["delta"] = j[args.compare] - j[args.tag]
        j.to_csv(d / ("comparison_vs_%s.csv" % args.compare))
        comparison = {"tag": args.tag, "other": args.compare,
                      "mean_self": float(j[args.tag].mean()),
                      "mean_other": float(j[args.compare].mean()),
                      "cells_better": int((j["delta"] > 0.001).sum()),
                      "cells_worse": int((j["delta"] < -0.001).sum()),
                      "cells_equal": int((j["delta"].abs() <= 0.001).sum())}
        print("--- %s vs %s (paired cells) ---" % (args.tag, args.compare))
        print("  mean success  %s %.3f   %s %.3f"
              % (args.tag, comparison["mean_self"], args.compare,
                 comparison["mean_other"]))
        print("  cells better %d / worse %d / equal %d"
              % (comparison["cells_better"], comparison["cells_worse"],
                 comparison["cells_equal"]))
        print("  by tau:")
        for tr, g in j.groupby("tau_ratio"):
            print("    tau=%5g : %s %.3f -> %s %.3f  (%+.3f)"
                  % (tr, args.tag, g[args.tag].mean(), args.compare,
                     g[args.compare].mean(), g["delta"].mean()))

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str((d / "sweep_trials.csv").relative_to(REPO)),
        "paired_sample": {"r_min": args.r_min, "initial_headings_deg": common,
                          "n_trials": int(len(sub))},
        "axis_success_rates": summary,
        "axis_direction": direction,
        "rescue_rate_where_baseline_failed": rescue_rate,
        "comparison": comparison,
        "selection_effect_note": (
            "Unpaired averages along r_max are misleading: at r_max = 0.002 "
            "only |e0| = 30 and 45 deg remain feasible, versus 8 distinct "
            "magnitudes at r_max >= 0.01. All numbers above are paired."),
    }
    (d / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("--- written ---")
    for f in ("selection_check.csv", "paired_grid.csv", "analysis.json",
              "fig3_paired_phase_maps.png", "fig4_axis_effects.png",
              "fig5_rescue_map.png"):
        if (d / f).exists():
            print("  " + f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
