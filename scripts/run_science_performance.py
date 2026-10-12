#!/usr/bin/env python3
"""Run matched numerical IRP controls with sealed, frozen-candidate audit."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from specrsi.science_optimization import ExperimentConfig, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260928])
    parser.add_argument("--development-calls", type=int, default=25)
    parser.add_argument("--execution-cap-ms", type=int, default=30000)
    parser.add_argument("--case-id", type=int, default=0)
    args = parser.parse_args()
    config = ExperimentConfig(tuple(args.seeds), args.development_calls, args.execution_cap_ms, args.case_id)
    result = run_experiment(args.output, config)
    print(json.dumps({"summary": str(args.output / "summary.json"),
                      "physical_calls": result["physical_calls"],
                      "scores": [{key: row[key] for key in ("method", "seed", "audit_score", "audit_baseline", "audit_gain")}
                                 for row in result["results"]]}, indent=2))


if __name__ == "__main__":
    main()
