#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
LEVEL="${1:-all}"

case "$LEVEL" in
  1|basic|smoke)
    python -m pytest -q hh_validation_suite/tests/test_level1_smoke.py
    ;;
  2|algorithm|alg)
    python -m pytest -q hh_validation_suite/tests/test_level2_algorithms.py
    ;;
  3|abstract|deep|fuzz)
    python -m pytest -q hh_validation_suite/tests/test_level3_abstract.py
    ;;
  4|energy|ground|vqe)
    python -m pytest -q hh_validation_suite/tests/test_level4_vqe_hh_ground_states.py
    ;;
  all|"")
    python -m pytest -q hh_validation_suite/tests/test_level1_smoke.py
    python -m pytest -q hh_validation_suite/tests/test_level2_algorithms.py
    python -m pytest -q hh_validation_suite/tests/test_level3_abstract.py
    python -m pytest -q hh_validation_suite/tests/test_level4_vqe_hh_ground_states.py
    ;;
  *)
    echo "Unknown level '$LEVEL'. Use 1|2|3|4 or all."
    exit 1
    ;;
esac
echo "HH validation suite finished for: $LEVEL"
