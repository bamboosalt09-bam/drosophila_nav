"""Download the MaleCNS v1.0 flat connectome into data/malecns/ (~1.1 GB).

Only needed to REBUILD the subnet or to query the whole connectome (e.g.
diag scripts that look up cell types).  The main experiment runs from the
committed subnet file results/subnet/vp_h4_rho0.50_ol_intrinsic.pkl.

    python scripts/get_malecns.py

Source: the public FlyEM bucket, CC-BY, no account needed.
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": 14483314,
    "body-neurotransmitters-male-cns-v1.0.feather": 43282834,
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": 1051241946,
}
DEST = Path(__file__).resolve().parents[1] / "data" / "malecns"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        out = DEST / name
        if out.exists() and out.stat().st_size == size:
            print("have   %s" % name)
            continue
        print("fetch  %s (%.0f MB)" % (name, size / 2**20), flush=True)
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(BASE + name, tmp)
        if tmp.stat().st_size != size:
            print("size mismatch for %s" % name, file=sys.stderr)
            return 1
        tmp.replace(out)
    print("data/malecns complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
