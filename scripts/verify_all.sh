#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
.venv/bin/python -m pip check > reports/pip_check.txt
.venv/bin/python -m pytest -q --junitxml=reports/pytest.xml
.venv/bin/python scripts/verify_reproduction.py "$@"
.venv/bin/python -m specrsi.cli doctor
