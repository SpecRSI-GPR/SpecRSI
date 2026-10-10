"""Command-line access to package versions and explicitly requested executions."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def doctor(*, base_url: str | None = None, model: str | None = None) -> dict:
    """Report installed versions and optionally query one caller-selected model."""
    if (base_url is None) != (model is None):
        raise ValueError("Provide both base_url and model when requesting a model health check")
    report = {"python": ".".join(str(value) for value in sys.version_info[:3]), "packages": {}}
    for name in ("specrsi-research", "numpy", "scipy", "PyYAML", "httpx", "torch",
                 "transformers", "peft", "huggingface-hub"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    if base_url is not None:
        from .models import LocalModelClient
        try:
            health = LocalModelClient(base_url, model).health()
            report["model"] = {"name": model, "status": health.get("status", "unknown")}
            if "revision" in health:
                report["model"]["revision"] = health["revision"]
        except Exception as error:
            report["model"] = {"name": model, "status": "unavailable",
                               "error_type": type(error).__name__}
    return report


def main():
    parser = argparse.ArgumentParser(prog="specrsi")
    sub = parser.add_subparsers(dest="command", required=True)
    diagnostic = sub.add_parser("doctor", help="Report installed package versions")
    diagnostic.add_argument("--base-url", help="Explicit model service URL for an optional health check")
    diagnostic.add_argument("--model", help="Explicit model name for the optional health check")
    demo = sub.add_parser("demo", help="Run the self-contained integer-expression example")
    run = sub.add_parser("run", help="Run an explicitly supplied experiment configuration")
    run.add_argument("--config", required=True)
    for command in (diagnostic, demo, run):
        command.add_argument("--output", type=Path, help="Optional path for the JSON result")
    arguments = parser.parse_args()
    if arguments.command == "doctor":
        if (arguments.base_url is None) != (arguments.model is None):
            parser.error("doctor requires --base-url and --model together")
        result = doctor(base_url=arguments.base_url, model=arguments.model)
    elif arguments.command == "demo":
        from .core.demo import run_demo
        result = run_demo()
    else:
        import yaml
        from .experiments import run_local_experiment
        config = yaml.safe_load(Path(arguments.config).read_text())
        if not isinstance(config, dict):
            parser.error("--config must contain a YAML mapping")
        result = run_local_experiment(config)
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
