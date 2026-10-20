"""Restore hash-verified flight artifacts from the repository's GitHub release.

Default: the runtime's graph/body/readout cache only. Historical source files in
the runtime backup never overwrite the current checked-out source code.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

BASE = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def fetch(asset, root):
    destination = root / asset['cache_path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == asset['bytes'] and sha256(destination) == asset['sha256']:
            print('Verified existing asset:', destination.name, flush=True)
            return destination
        raise RuntimeError('Existing file differs; preserving it: ' + str(destination))
    partial = destination.with_name(destination.name + '.restore.part')
    print('Downloading:', asset['name'], flush=True)
    urllib.request.urlretrieve(asset['url'], partial)
    if partial.stat().st_size != asset['bytes'] or sha256(partial) != asset['sha256']:
        partial.unlink(missing_ok=True)
        raise RuntimeError('Asset checksum mismatch: ' + asset['name'])
    partial.replace(destination)
    return destination

def extract(path, root, prefix):
    count = 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise RuntimeError('Unsafe archive path')
            if not member.name.startswith(prefix):
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError('Only ordinary files are restored')
            destination = root.joinpath(*name.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + '.restore.part')
            with archive.extractfile(member) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output)
            if destination.exists():
                same = sha256(destination) == sha256(partial)
                partial.unlink()
                if not same:
                    raise RuntimeError('Existing file differs; preserving it: ' + str(destination))
            else:
                partial.replace(destination)
            count += 1
    if not count:
        raise RuntimeError('No matching artifact files found')
    print('Restored/verified files:', count, flush=True)
    return count

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--all-derived', action='store_true', help='Also restore contact counts and calibration responses')
    parser.add_argument('--results', action='store_true', help='Restore the sealed original results packet')
    parser.add_argument('--raw', action='store_true', help='Restore the three original MaleCNS files, about 1.1GB')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((BASE / 'recovery_backups/github_assets.json').read_text())
    categories = {'derived' if args.all_derived or args.all else 'runtime'}
    if args.results or args.all:
        categories.add('results')
    if args.raw or args.all:
        categories.add('raw')
    root = args.root.resolve()
    for asset in manifest['assets']:
        if asset['category'] not in categories:
            continue
        path = fetch(asset, root)
        if asset['category'] in {'runtime', 'derived'}:
            extract(path, root, 'flight_recovery/derived/')
        elif asset['category'] == 'results':
            extract(path, root, 'results/flight_recovery_2026-10-18/')
    print('Flight artifacts are available under:', root)

if __name__ == '__main__':
    main()
