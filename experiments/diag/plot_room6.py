"""Room 6 after the saccade fix: full grid, and connectome before/after."""
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

RM = 6
INK = "#1f1f1e"
w, h, reach = load(RM)
tg = np.array(w.all_targets())
x0, x1, y0, y1 = w.bounds
R = REPO / "results"
BEFORE = sys.argv[1] if len(sys.argv) > 1 else "v4"
AFTER = sys.argv[2] if len(sys.argv) > 2 else "v5"
runs = {"before": (pd.read_csv(R / ("sweep_beams_room6_%s.csv" % BEFORE)), np.load(R / ("paths_room6_%s.npz" % BEFORE))),
        "after": (pd.read_csv(R / "sweep_beams_room6.csv"), np.load(R / "paths_room6.npz"))}
# the beam layouts both runs flew
BEAMS = sorted(set(runs["after"][0].beams) & set(runs["before"][0].beams))
WIDE = 3.5 * len(BEAMS)


def panel(ax, ver, arm, n, label):
    df, P = runs[ver]
    r = df[(df.arm == arm) & (df.beams == n)].iloc[0]
    path = P["b%d_%s" % (n, arm)]
    got = {int(v) for v in str(r.reached).split()} if str(r.reached) != "nan" else set()
    for o in w.obstacles:
        ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="0.62", lw=0))
    for k, (t, ok) in enumerate(zip(tg, reach)):
        ax.plot(t[0], t[1], marker="*" if ok else "X", ms=10,
                color="#008300" if ok else "0.3", zorder=4)
        if k in got:
            ax.add_patch(plt.Circle((t[0], t[1]), 1.3, fill=False, lw=1.8,
                                    color="#1c5cab", zorder=5))
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
    ax.set_title("%s, %d beams\n%d/%d found, %s, %.0f m, stop %.0f%%"
                 % (label, n, r.found_ok, r.reachable,
                    "COLLIDED" if r.collided else "no collision",
                    r.dist, 100 * r.stopped),
                 fontsize=8.5, color="#e34948" if r.collided else INK)


legend = ("green star = reachable beacon, blue ring = reached, circle = start, "
          "square = end, red X = collision, path colour = time (dark -> bright)")

fig, axes = plt.subplots(3, len(BEAMS), figsize=(WIDE, 11.2), squeeze=False)
for i, arm in enumerate(("connectome", "planner", "centroid")):
    for j, n in enumerate(BEAMS):
        panel(axes[i, j], "after", arm, n, arm)
fig.suptitle("room 6 AFTER, 120 s per flight.  " + legend,
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(R / ("sweep_%s_paths_room6.png" % AFTER), dpi=72)
plt.close(fig)

fig, axes = plt.subplots(2, len(BEAMS), figsize=(WIDE, 7.6), squeeze=False)
for i, (ver, lab) in enumerate((("before", "connectome BEFORE"),
                                ("after", "connectome AFTER"))):
    for j, n in enumerate(BEAMS):
        panel(axes[i, j], ver, "connectome", n, lab)
fig.suptitle("room 6, connectome before / after.  "
             + legend,
             fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(R / ("sweep_%s_connectome_%s_vs_%s_room6.png" % (AFTER, BEFORE, AFTER)), dpi=72)
print("wrote both")
