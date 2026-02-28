#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ARTIFACT_DIR="$SCRIPT_DIR/artifacts"
mkdir -p "$ARTIFACT_DIR"
export ADAPT_HH_ARTIFACT_DIR="$ARTIFACT_DIR"

python -m pytest -q hh_adapt_vqe_validation_suite/tests/test_hh_adapt_vqe_ground_states.py
