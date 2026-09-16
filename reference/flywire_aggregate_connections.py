"""Aggregate FlyWire's per-neuropil connection rows into a neuron x neuron graph.

The published table splits one neuron pair across every neuropil the pair
touches, so it has 16.8M rows for far fewer actual edges.  Summing synapse
counts per (pre, post) gives the graph a simulation needs.

Memory matters here: this machine has little free RAM, so the aggregation runs
in Arrow (C++) rather than pandas.
"""
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as ft

SRC = "data/flywire/proofread_connections_783.feather"


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


t0 = time.perf_counter()
tbl = ft.read_table(SRC, memory_map=True)
log("loaded %d rows in %.1f s" % (tbl.num_rows, time.perf_counter() - t0))

# weight each neurotransmitter probability by the synapse count, so the
# aggregate is a synapse-weighted mean rather than a mean of means
nt_cols = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
syn = tbl.column("syn_count")
weighted = {c: pc.multiply(pc.cast(tbl.column(c), pa.float64()),
                           pc.cast(syn, pa.float64())) for c in nt_cols}
agg_in = pa.table(
    {"pre": tbl.column("pre_pt_root_id"), "post": tbl.column("post_pt_root_id"),
     "syn_count": syn, **weighted})
del tbl, weighted

t0 = time.perf_counter()
g = agg_in.group_by(["pre", "post"]).aggregate(
    [("syn_count", "sum")] + [(c, "sum") for c in nt_cols])
log("aggregated to %d neuron-pair edges in %.1f s"
    % (g.num_rows, time.perf_counter() - t0))
del agg_in

syn_sum = g.column("syn_count_sum").to_numpy()
log("")
log("synapse-count threshold decides both the model and the runtime:")
log("  %-12s %12s %10s %14s" % ("threshold", "edges", "% kept", "rel. speed"))
base = None
for thr in (1, 2, 3, 5, 10, 20):
    n = int((syn_sum >= thr).sum())
    if base is None:
        base = n
    log("  >= %-9d %12d %9.1f%% %13.2fx" % (thr, n, 100 * n / len(syn_sum),
                                            base / n))

log("")
log("synapse count per edge: median %.0f  mean %.1f  max %d"
    % (np.median(syn_sum), syn_sum.mean(), syn_sum.max()))

# sign from the dominant predicted neurotransmitter
probs = np.stack([g.column(c + "_sum").to_numpy() for c in nt_cols], axis=1)
probs /= np.maximum(syn_sum[:, None], 1)
top = probs.argmax(axis=1)
names = [c.replace("_avg", "") for c in nt_cols]
log("")
log("dominant predicted transmitter per EDGE (synapse-count weighted):")
for i, nm in enumerate(names):
    k = int((top == i).sum())
    log("  %-6s %10d edges (%.1f%%)" % (nm, k, 100 * k / len(top)))

# standard assumption: ACh/DA/OA excitatory, GABA/Glu inhibitory
inhib = np.isin(top, [names.index("gaba"), names.index("glut")])
log("")
log("under the standard sign assumption (ACh/DA/OA/5-HT +, GABA/Glu -):")
log("  excitatory edges %d (%.1f%%)   inhibitory %d (%.1f%%)"
    % ((~inhib).sum(), 100 * (~inhib).mean(), inhib.sum(), 100 * inhib.mean()))

out = "data/flywire/edges_783.npz"
pre = g.column("pre").to_numpy()
post = g.column("post").to_numpy()
np.savez_compressed(out, pre=pre, post=post, syn=syn_sum.astype(np.int32),
                    top_nt=top.astype(np.int8), nt_names=np.array(names))
log("")
log("written %s" % out)
