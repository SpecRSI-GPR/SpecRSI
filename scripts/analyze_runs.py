#!/usr/bin/env python3
"""Paired/clustered analysis from explicit arm records. See --help for schema."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from specrsi.statistics import paired_bootstrap


def main():
    p = argparse.ArgumentParser(description='JSONL rows: comparison_id, pair_id, arm (specrsi/baseline), score, optional cluster_id')
    p.add_argument("input")
    p.add_argument("--output", default="results/analysis.json")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--family-size", type=int, default=13)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rows = [json.loads(line) for line in Path(a.input).read_text().splitlines() if line.strip()]
    comparisons = {}
    for row in rows:
        key = (row["comparison_id"], row["pair_id"])
        pair = comparisons.setdefault(key, {})
        if row["arm"] not in {"specrsi", "baseline"} or row["arm"] in pair:
            raise ValueError(f"Unknown or duplicated arm for {key}")
        pair[row["arm"]] = row
    results = {}
    for comp in sorted({key[0] for key in comparisons}):
        pairs = [pair for key, pair in comparisons.items() if key[0] == comp]
        if any(set(pair) != {"specrsi", "baseline"} for pair in pairs):
            raise ValueError(f"Unpaired observations in {comp}; refusing silent dropping")
        for pair in pairs:
            if pair["specrsi"].get("cluster_id") != pair["baseline"].get("cluster_id"):
                raise ValueError("Cluster identities differ across a pair")
        clusters = [pair["specrsi"].get("cluster_id", pair["specrsi"]["pair_id"]) for pair in pairs]
        results[comp] = asdict(paired_bootstrap([p["specrsi"]["score"] for p in pairs],
            [p["baseline"]["score"] for p in pairs], clusters=clusters, resamples=a.resamples,
            family_size=a.family_size, seed=a.seed))
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
