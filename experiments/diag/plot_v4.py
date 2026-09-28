"""Draw the v4 sweep: one path grid per room, plus a 4-room summary."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from environment.room_file import load

ROOMS = (0, 5, 6, 7)
BEAMS = (0, 1, 3, 6, 12, 24)
ARMS = ("connectome", "planner", "centroid")
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
COL = {"connectome": "#2a78d6", "planner": "#eb6834", "centroid": "#1baf7a"}
MARK = {"connectome": "o", "planner": "s", "centroid": "^"}
outs = []

for rm in ROOMS:
    w, h, reach = load(rm)
    df = pd.read_csv(REPO / "results" / ("sweep_beams_room%d.csv" % rm))
    P = np.load(REPO / "results" / ("paths_room%d.npz" % rm))
    tg = np.array(w.all_targets())
    x0, x1, y0, y1 = w.bounds
    fig, axes = plt.subplots(3, 6, figsize=(21, 11.2))
    for i, arm in enumerate(ARMS):
        for j, n in enumerate(BEAMS):
            ax = axes[i, j]
            r = df[(df.arm == arm) & (df.beams == n)].iloc[0]
            path = P["b%d_%s" % (n, arm)]
            got = {int(v) for v in str(r.reached).split()} if str(r.reached) != "nan" else set()
            for o in w.obstacles:
                ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="0.62", lw=0))
            for k, (t, ok) in enumerate(zip(tg, reach)):
                ax.plot(t[0], t[1], marker="*" if ok else "X", ms=10,
                        color="#008300" if ok else "0.3", zorder=4)
                if k in got:
                    ax.add_patch(plt.Circle((t[0], t[1]), 1.3, fill=False,
                                            lw=1.8, color="#1c5cab", zorder=5))
            if len(path) > 1:
                seg = np.stack([path[:-1], path[1:]], axis=1)
                lc = LineCollection(seg, cmap="plasma", lw=1.5, zorder=3)
                lc.set_array(np.linspace(0, 1, len(seg)))
                ax.add_collection(lc)
            ax.plot(*path[0], "o", color=INK, ms=5, zorder=6)
            ax.plot(*path[-1], "X" if r.collided else "s",
                    color="#e34948" if r.collided else INK,
                    ms=10 if r.collided else 5, zorder=7)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("%s, %d beams\n%d/%d found%s, %s, %.0f m, stop %.0f%%"
                         % (arm, n, r.found_ok, r.reachable,
                            (" +%d walled" % r.found_walled) if r.found_walled else "",
                            "COLLIDED" if r.collided else "no collision",
                            r.dist, 100 * r.stopped),
                         fontsize=8.5, color="#e34948" if r.collided else INK)
    fig.suptitle("room %d, 120 s per flight.  green star = reachable beacon, "
                 "grey X = walled off, blue ring = reached, circle = start, "
                 "square = end, red X = collision, path colour = time "
                 "(dark -> bright)" % rm, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = REPO / "results" / ("sweep_v4_paths_room%d.png" % rm)
    fig.savefig(out, dpi=72)
    plt.close(fig)
    outs.append(out)

# ---- summary over the four rooms --------------------------------------
rows = []
for rm in ROOMS:
    d = pd.read_csv(REPO / "results" / ("sweep_beams_room%d.csv" % rm))
    rows.append(d)
d = pd.concat(rows)
g = d.groupby(["arm", "beams"]).agg(found=("found_ok", "sum"),
                                    reach=("reachable", "sum"),
                                    coll=("collided", "sum")).reset_index()
xs = np.arange(len(BEAMS))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.6, 1]})
for arm in ARMS:
    s = g[g.arm == arm].set_index("beams").loc[list(BEAMS)]
    frac = s.found / s.reach
    a1.plot(xs, frac, color=COL[arm], lw=2, marker=MARK[arm], ms=8,
            markeredgecolor="white", markeredgewidth=1.5, label=arm, zorder=3)
    a1.annotate("%s  %d/%d" % (arm, s.found.iloc[-1], s.reach.iloc[-1]),
                (xs[-1], frac.iloc[-1]), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=INK)
    a2.plot(xs, s.coll, color=COL[arm], lw=2, marker=MARK[arm], ms=8,
            markeredgecolor="white", markeredgewidth=1.5, label=arm, zorder=3)
for ax in (a1, a2):
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BEAMS])
    ax.set_xlabel("range beams", color=INK2)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2)
a1.set_ylim(0, 1.05); a1.set_xlim(-0.3, len(BEAMS) + 0.9)
a1.set_ylabel("share of reachable beacons found", color=INK2)
a1.set_title("Beacons found, rooms 0, 5, 6, 7 combined", color=INK,
             loc="left", fontsize=11)
a2.set_ylim(-0.1, 4.2); a2.set_yticks(range(5))
a2.set_ylabel("rooms with a collision (of 4)", color=INK2)
a2.set_title("Collisions", color=INK, loc="left", fontsize=11)
a2.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
out = REPO / "results" / "sweep_v4_summary.png"
fig.savefig(out, dpi=90)
outs.append(out)
print("\n".join(str(o) for o in outs))
print(g.to_string(index=False))
