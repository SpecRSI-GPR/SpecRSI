#!/usr/bin/env python3
"""Plot saved complete incumbents from actual SpecRSI search transition logs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="search.json files; all curves must use the same cost unit")
    parser.add_argument("--output", default="results/figures/selection_curve.pdf")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = [json.loads(Path(p).read_text()) for p in args.runs]
    units = {d["cost_unit"] for d in data}
    if len(units) != 1:
        raise ValueError("Do not plot tokens, CPU time and normalized paper B on a common cost axis")
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for filename, run in zip(args.runs, data):
        base, margin = run["acceptance"]["incumbent_score"], run["acceptance"]["margin"]
        costs, scores = [0], [base]
        for transition in run["transitions"]:
            costs.append(costs[-1] + transition["cost"])
            gain = transition["f_after"]
            scores.append(base + gain if gain > margin else base)
        label = str(Path(filename).parent)
        ax.step(costs, scores, where="post", label=label, linewidth=1.5)
    ax.set(xlabel="Charged search cost (" + next(iter(units)) + ")", ylabel="Complete selected score")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
