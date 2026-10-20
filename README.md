# Flight artifacts — 2026-10-20

This independent data tag preserves every release archive and original MaleCNS source file as SHA256-verified parts of at most 32MiB. The source code and measurements are on the repository master branch. This tag has no source-branch parent and can be excluded from source-only checkouts with `git clone --depth 1 --single-branch --branch master --no-tags https://github.com/bamboosalt09-bam/drosophila_nav.git`. A full clone including all tags downloads these data objects.

From the master checkout run `python flight_recovery/restore_artifacts.py` for runtime caches or append `--all` for all derived/raw/result assets. The script verifies every part and the reconstructed full file before extracting it. See github_assets.json in this tag or on master for all hashes and raw download URLs.

Sources: FlyEM MaleCNS v1.0, CC-BY, https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/ . Flybody references and licenses are in the source repository. These are engineering-flight data, not validation of natural fly physiology.
