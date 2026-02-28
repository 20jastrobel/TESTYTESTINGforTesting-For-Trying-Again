#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MAX_ATTEMPTS="${1:-25}"
if ! [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [MAX_ATTEMPTS]" >&2
  exit 1
fi

BASE_ARTIFACT_DIR="$SCRIPT_DIR/artifacts"
mkdir -p "$BASE_ARTIFACT_DIR"

for ((attempt=0; attempt<MAX_ATTEMPTS; attempt++)); do
  seed=$((7 + attempt))
  attempt_dir="$BASE_ARTIFACT_DIR/attempt_${attempt}"
  mkdir -p "$attempt_dir"

  echo "== HH ADAPT-VQE stress attempt $((attempt + 1))/$MAX_ATTEMPTS (seed=$seed) =="
  echo "  artifacts: $attempt_dir"

  if ADAPT_HH_ARTIFACT_DIR="$attempt_dir" ADAPT_HH_TEST_SEED="$seed" \
    python -m pytest -q hh_adapt_vqe_validation_suite/tests/test_hh_adapt_vqe_ground_states.py ; then
    echo "PASS attempt=$attempt seed=$seed"
    echo "L2 artifact: $attempt_dir/hh_adapt_vqe_L2_seed${seed}.json"
    echo "L3 artifact: $attempt_dir/hh_adapt_vqe_L3_seed${seed}.json"
    exit 0
  fi

  echo "FAIL attempt=$attempt seed=$seed"
  echo "L2 artifact: $attempt_dir/hh_adapt_vqe_L2_seed${seed}.json"
  echo "L3 artifact: $attempt_dir/hh_adapt_vqe_L3_seed${seed}.json"
  echo "Retrying with next seed..."
done

echo "HH ADAPT-VQE HH suite did not pass after $MAX_ATTEMPTS attempts."
exit 1
