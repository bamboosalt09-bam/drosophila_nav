"""Gate check: can a retinotopic map be recovered from neuron positions alone?

To give the fly its own view we must know which medulla column looks in which
direction.  The annotation table has 3D positions in the brain but no visual
direction.  The optic lobe is retinotopically organised, so position should
carry that information -- but "should" is not "does".

Mi1 is the canonical one-per-column marker, so if the columns form a clean 2D
lattice in some projection, the map is recoverable; if Mi1 positions are a
blob, this approach fails and the plan needs rethinking.
"""
import sys

import numpy as np
import pandas as pd


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


a = pd.read_csv("data/flywire/neuron_annotations_783.tsv", sep="\t",
                low_memory=False)
mi1 = a[a.cell_type.astype(str) == "Mi1"]
log("Mi1 neurons: %d" % len(mi1))
log("by side: %s" % mi1.side.value_counts().to_dict())
log("")

for side in ("left", "right"):
    s = mi1[mi1.side == side]
    pos = s[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
    ok = np.isfinite(pos).all(axis=1)
    pos = pos[ok]
    log("--- %s eye: %d Mi1 with positions ---" % (side, len(pos)))

    c = pos - pos.mean(axis=0)
    u, sv, vt = np.linalg.svd(c, full_matrices=False)
    log("  PCA singular values: %s" % np.array2string(sv, precision=0))
    log("  variance explained by first 2 axes: %.1f%%"
        % (100 * (sv[:2] ** 2).sum() / (sv ** 2).sum()))

    xy = c @ vt[:2].T          # flatten onto the sheet
    log("  sheet extent: %.0f x %.0f nm" % (np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))

    # nearest-neighbour spacing: a lattice has a tight distribution,
    # a blob has a broad one
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=7)
    nn = d[:, 1]
    log("  nearest-neighbour spacing: median %.0f nm, IQR %.0f-%.0f, CV %.2f"
        % (np.median(nn), np.percentile(nn, 25), np.percentile(nn, 75),
           nn.std() / nn.mean()))
    ring = d[:, 1:7].mean(axis=1)
    log("  mean distance to 6 nearest (hex lattice expects ~uniform): CV %.2f"
        % (ring.std() / ring.mean()))

    # how many neighbours sit within 1.5x the median spacing?
    m = np.median(nn)
    cnt = np.array([len(x) - 1 for x in cKDTree(xy).query_ball_point(xy, 1.5 * m)])
    log("  neighbours within 1.5x median spacing: mean %.1f (hex lattice -> ~6)"
        % cnt.mean())
    log("")

log("interpretation: a clean hexagonal sheet means the column lattice is")
log("recoverable from position, and a direction map can be built on it.")
