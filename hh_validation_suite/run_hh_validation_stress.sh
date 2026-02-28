#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

cycles="${1:-5}"
if ! [[ "$cycles" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [cycles]" >&2
  exit 1
fi

for ((i=1; i<=cycles; i++)); do
  echo "== HH validation stress cycle $i/$cycles =="
  bash hh_validation_suite/run_hh_validation.sh all
  python -m pytest -q \
    pydephasing/quantum/test_hubbard_holstein.py \
    pydephasing/quantum/test_hubbard_holstein_vqe_ansatz.py \
    pydephasing/quantum/test_hh_ed_restricted_benchmark.py
done

echo "HH validation stress run complete: $cycles cycles"
