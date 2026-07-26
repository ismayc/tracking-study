#!/bin/bash
# Every check in this repo: unit tests, then the study's reconciliation /
# validation gate. Non-zero exit on any failure. PY overrides the
# interpreter (the family orchestrator sets it to a shared venv).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

echo "=== unit tests: Python (pytest) ==="
$PY -m pytest tests/python -q

echo
echo "=== reconciliation / validation ==="
$PY python/04_validate.py | tail -1

echo
echo "=== modern aggregates: cached-data gate replay ==="
$PY python/07_modern_aggregates.py --check | tail -1

echo
echo "=== findings: README numbers regenerate from outputs ==="
$PY python/08_findings.py --check | tail -1
