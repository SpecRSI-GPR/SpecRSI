#!/usr/bin/env python3
"""Register then run the pinned official FlashEvolve GEPA runtime locally."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from specrsi.baselines.flash_evolve import audit_only, execute, register, search_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    reg = sub.add_parser('register')
    reg.add_argument('--config', required=True)
    reg.add_argument('--output', required=True)
    run = sub.add_parser('run')
    run.add_argument('directory')
    search = sub.add_parser('search-only', help='Optimize and freeze without any heldout model call')
    search.add_argument('directory')
    audit = sub.add_parser('audit-only', help='Verify frozen search and optional campaign barrier, then evaluate')
    audit.add_argument('directory')
    audit.add_argument('--expected-frozen-sha256', required=True)
    audit.add_argument('--barrier')
    audit.add_argument('--barrier-sha256')
    args = parser.parse_args()
    if args.action == 'register':
        value = register(yaml.safe_load(Path(args.config).read_text()), args.output)
        print(json.dumps({'status': 'registered', 'output': args.output, 'upstream': value['upstream'],
                          'data_source_splits': value['data_source_splits']}, indent=2))
    elif args.action == 'search-only':
        print(json.dumps(search_only(args.directory), indent=2))
    elif args.action == 'audit-only':
        print(json.dumps(audit_only(args.directory, expected_frozen_sha256=args.expected_frozen_sha256,
            barrier_path=args.barrier, barrier_sha256=args.barrier_sha256), indent=2))
    else:
        print(json.dumps(execute(args.directory), indent=2))


if __name__ == '__main__':
    main()
