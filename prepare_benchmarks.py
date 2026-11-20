"""Convert the pinned PyGOD Books and Disney releases to the local NPZ format."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch

from src.data import validate


SOURCE = 'https://raw.githubusercontent.com/pygod-team/data/main'
ARCHIVE_HASHES = {
    'books': 'c9de331ddf4c893e205fad552a9d1074c0fe4730edd7b47952f0c05ac360612c',
    'disney': '761d111a6ec9d545445241fab8a6056333d9f1c65a61a6b72604ce486fffe949',
}


class DataRecord:
    """Plain state holder for an archived PyG Data object."""


class StorageRecord:
    """Plain state holder for an archived PyG GlobalStorage object."""


def convert(name, archive_bytes):
    digest = sha256(archive_bytes).hexdigest()
    if digest != ARCHIVE_HASHES[name]:
        raise ValueError(f'Unexpected SHA-256 for {name}; use the pinned dataset archive.')
    with ZipFile(BytesIO(archive_bytes)) as archive:
        payload = BytesIO(archive.read(f'{name}.pt'))
    # Read only tensors and plain state from these verified releases. No PyG import.
    with torch.serialization.safe_globals([
        (DataRecord, 'torch_geometric.data.data.Data'),
        (StorageRecord, 'torch_geometric.data.storage.GlobalStorage'),
    ]):
        fields = torch.load(payload, map_location='cpu', weights_only=True)._store._mapping
    x = fields['x'].numpy()
    edge_index = fields['edge_index'].numpy()
    if edge_index.shape[0] != 2 or edge_index.min() < 0 or edge_index.max() >= len(x):
        raise ValueError(f'Invalid node indices in {name}.')
    a = np.zeros((len(x), len(x)), dtype=np.uint8)
    source, target = edge_index
    a[source, target] = 1
    a[target, source] = 1
    return validate(x, a, fields['y'].numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--download', action='store_true', help='Retrieve missing public source archives')
    args = parser.parse_args()
    source_dir = args.data_dir / 'source'
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, digest in ARCHIVE_HASHES.items():
        archive_path = source_dir / f'{name}.pt.zip'
        url = f'{SOURCE}/{archive_path.name}'
        downloaded = not archive_path.is_file()
        if downloaded and not args.download:
            parser.error(f'Missing {archive_path}. Use --download to retrieve it.')
        if downloaded:
            with urlopen(url, timeout=30) as response:
                archive_bytes = response.read(1_000_001)
        else:
            archive_bytes = archive_path.read_bytes()
        x, a, y = convert(name, archive_bytes)
        if downloaded:
            archive_path.write_bytes(archive_bytes)
        npz = args.data_dir / f'{name}.npz'
        np.savez_compressed(npz, x=x.astype(np.float32), a=a.astype(np.uint8), y=y.astype(np.uint8))
        metadata[name] = {
            'source': url,
            'archive_sha256': digest,
            'npz_sha256': sha256(npz.read_bytes()).hexdigest(),
            'nodes': len(y),
            'edges_undirected': int(a.sum() / 2),
            'features': x.shape[1],
            'constant_features': int((x.std(axis=0) == 0).sum()),
            'anomalies': int(y.sum()),
        }
        print(f'{name}: {metadata[name]}', flush=True)
    manifest = args.data_dir / 'benchmarks.json'
    manifest.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
