#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports logs .envs
if command -v dnf >/dev/null; then
  dnf install -y bubblewrap golang rust cargo java-17-openjdk-devel boost-devel boost-date-time podman
fi
.venv/bin/python -m pip install 'bm25s==0.2.14' 'PyStemmer==3.0.0'
if [ ! -x .envs/math/bin/python ]; then
  /usr/local/bin/python -m venv .envs/math
fi
.envs/math/bin/python -m pip install -r configs/runtimes/math.requirements.txt
.venv/bin/python scripts/prepare_data.py --datasets math
.venv/bin/python scripts/prepare_retrieval.py
.venv/bin/python scripts/prepare_polyglot.py
.venv/bin/python scripts/prepare_science.py --install
.venv/bin/python scripts/prepare_secondary_baselines.py
.venv/bin/python -m pip install -r configs/baselines/secondary_requirements.txt
.venv/bin/python -m pip install --no-deps -e third_party/secondary_textgrad -e third_party/secondary_trace
.venv/bin/python -m pip check
