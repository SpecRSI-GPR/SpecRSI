#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
mkdir -p reports
.venv/bin/python -m pip install -e '.[models,data,solver,test]'
.venv/bin/python -m pip freeze > reports/environment.freeze.txt
for option in "$@"; do
  case "$option" in
    --with-model)
      .venv/bin/python scripts/download_model.py
      ;;
    --with-baselines)
      .venv/bin/python scripts/prepare_baselines.py
      .venv/bin/python scripts/prepare_upstream.py
      .venv/bin/python -m pip install ./third_party/gepa ./third_party/dspy
      .venv/bin/python -m pip install -r configs/baselines/secondary_requirements.txt
      .venv/bin/python -m pip install --no-deps ./third_party/secondary_textgrad ./third_party/secondary_trace
      ;;
    --with-data)
      .venv/bin/python scripts/prepare_data.py
      .venv/bin/python -m pip install ./data/raw/ifbench/source/IFBench-*
      ;;
    *)
      echo "Usage: $0 [--with-model] [--with-baselines] [--with-data]" >&2
      exit 2
      ;;
  esac
done
.venv/bin/python -m pip freeze > reports/environment.freeze.txt
