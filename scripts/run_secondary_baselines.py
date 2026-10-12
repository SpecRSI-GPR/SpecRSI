#!/usr/bin/env python3
"""Run pinned original TextGrad, Trace/OptoPrime, or MIPRO with demonstrations."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from specrsi.baselines.native_backend import native_splits
from specrsi.baselines.secondary_common import create_model


def run(config):
    if config['method'] not in {'textgrad', 'trace-optoprime', 'mipro-demos'}:
        raise ValueError('Unknown secondary optimizer')
    splits = native_splits(config)
    model = create_model(config)
    if config['method'] == 'mipro-demos':
        from specrsi.baselines.secondary_mipro import run_mipro_demos
        return run_mipro_demos(config, splits, model)
    from specrsi.baselines.secondary_gradients import run_gradients
    return run_gradients(config, splits, model)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.output:
        config['output'] = args.output
    if args.plan:
        print(json.dumps(config, indent=2)); return
    print(json.dumps(run(config), indent=2))


if __name__ == '__main__':
    main()
