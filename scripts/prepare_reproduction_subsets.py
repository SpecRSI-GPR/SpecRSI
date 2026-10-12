#!/usr/bin/env python3
"""Materialize registered dataset subsets from downloaded canonical data.

The metadata contain ordered native IDs, parent file identities, and expected
output identities. No examples are downloaded separately or generated here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root containing data and metadata.')
    parser.add_argument('--subset', action='append', help='Subset name; repeat to select several. Default: all.')
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = json.loads((root / 'data/reproduction/materialized_subsets.json').read_text())
    selected = set(args.subset or [x['subset'] for x in metadata['subsets']])
    known = {x['subset'] for x in metadata['subsets']}
    if selected - known:
        parser.error(f'Unknown subsets: {sorted(selected - known)}')
    cache = {}
    for subset in metadata['subsets']:
        if subset['subset'] not in selected:
            continue
        manifests = {}
        for record in subset['files']:
            source = root / record['source_path']
            if source not in cache:
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != record['source_sha256']:
                    raise ValueError(f'Parent dataset differs from its registered identity: {source}')
                cache[source] = {str(row['id']):(offset, row)
                    for offset, line in enumerate(data.decode('utf-8').split('\n'))
                    if line.strip() for row in [json.loads(line)]}
            rows = []
            for key in record['ids']:
                offset, source_row = cache[source][key]
                row = copy.deepcopy(source_row)
                row['split'] = record['split']
                row['metadata']['source_split'] = 'development'
                row['metadata']['source_offset_zero_based'] = offset
                rows.append(canonical(row) + '\n')
            output = ''.join(rows).encode()
            if len(output) != record['bytes'] or hashlib.sha256(output).hexdigest() != record['sha256']:
                raise ValueError(f'Prepared subset differs from its registered identity: {subset["subset"]}/{record["split"]}')
            target = root / record['processed_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(output)
            manifest = manifests.setdefault(record['dataset'], {
                'schema_version':1, 'dataset':record['dataset'], 'family':'prompts',
                'counts':{}, 'files':[], 'parent_source': {
                    'path':record['source_path'], 'sha256':record['source_sha256']},
                'split_policy':{'algorithm':'registered_ordered_native_ids',
                                'source_split':'development'}, 'status':'prepared'})
            manifest['counts'][record['split']] = record['count']
            data_root = root / subset['data_root']
            manifest['files'].append({'path':str(target.relative_to(data_root.parent)),
                'sha256':record['sha256'], 'bytes':record['bytes']})
            print(json.dumps({'subset':subset['subset'], 'split':record['split'], 'count':record['count']}))
        for dataset, manifest in manifests.items():
            target = root / subset['data_root'] / 'manifests' / (dataset + '.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
