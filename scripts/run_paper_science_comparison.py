#!/usr/bin/env python3
"""Run/resume frozen common-holdout native science HF reference comparisons."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import yaml
    from specrsi.science_comparison import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', choices=('search', 'heldout', 'all'), default='all')
    parser.add_argument('--expected-freeze-digest')
    parser.add_argument('--freeze-barrier', type=Path)
    parser.add_argument('--freeze-barrier-sha256')
    args = parser.parse_args()
    result = run(yaml.safe_load(args.config.read_text()), stage=args.stage,
        expected_freeze_digest=args.expected_freeze_digest, freeze_barrier=args.freeze_barrier,
        freeze_barrier_sha256=args.freeze_barrier_sha256)
    print(json.dumps({key: result[key] for key in ('output', 'status', 'domain', 'seed',
        'frozen_candidates_digest', 'equal_budget_comparison', 'sota_established', 'paper_named_baselines_missing')}, indent=2))
    if result['status'] not in {'complete', 'search_complete'}:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
