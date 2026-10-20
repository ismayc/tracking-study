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
echo "=== frame coverage: every game's tracking covers its clock ==="
$PY python/09_frame_coverage.py --check | tail -1

echo
echo "=== tracking box score: per-player-game distance and minutes vs the NBA's BoxScorePlayerTrackV3 ==="
$PY python/10_trackv3_check.py --check | tail -1

echo
echo "=== modern aggregates: cached-data gate replay ==="
$PY python/07_modern_aggregates.py --check | tail -1

echo
echo "=== findings: README numbers regenerate from outputs ==="
$PY python/08_findings.py --check | tail -1

echo
echo "=== join diagnostics: lead sweep agrees with the calibrated leads ==="
$PY python/11_join_diagnostics.py --check | tail -1

echo
echo "=== diagrams: committed SVGs rebuild from outputs ==="
$PY python/12_diagrams.py --check | tail -1
