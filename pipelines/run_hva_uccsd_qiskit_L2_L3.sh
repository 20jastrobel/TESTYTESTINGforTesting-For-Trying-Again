#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/opt/anaconda3/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not executable at ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p artifacts

# Shared physics settings.
T_VAL="1.0"
U_VAL="4.0"
DV_VAL="0.0"
BOUNDARY="periodic"
ORDERING="blocked"
SUZUKI_ORDER="2"
T_FINAL="20.0"
INITIAL_STATE_SOURCE="vqe"

# Regression-grade settings.
L2_TROTTER_STEPS="64"
L2_NUM_TIMES="201"
L2_VQE_REPS="2"
L2_VQE_RESTARTS="1"
L2_VQE_MAXITER="120"

L3_TROTTER_STEPS="128"
L3_NUM_TIMES="401"
L3_VQE_REPS="2"
L3_VQE_RESTARTS="3"
L3_VQE_MAXITER="600"

# Shared seeds.
HARDCODED_VQE_SEED="7"
QISKIT_VQE_SEED="7"

# QPE disabled for this comparison path.
QPE_EVAL_QUBITS="5"
QPE_SHOTS="256"
QPE_SEED="11"

run_triplet_for_L() {
  local L="$1"
  local TROTTER_STEPS="$2"
  local NUM_TIMES="$3"
  local VQE_REPS="$4"
  local VQE_RESTARTS="$5"
  local VQE_MAXITER="$6"

  echo "Running hardcoded UCCSD pipeline for L=${L}..."
  "${PYTHON_BIN}" pipelines/hardcoded_hubbard_pipeline.py \
    --L "${L}" \
    --t "${T_VAL}" \
    --u "${U_VAL}" \
    --dv "${DV_VAL}" \
    --boundary "${BOUNDARY}" \
    --ordering "${ORDERING}" \
    --t-final "${T_FINAL}" \
    --num-times "${NUM_TIMES}" \
    --suzuki-order "${SUZUKI_ORDER}" \
    --trotter-steps "${TROTTER_STEPS}" \
    --term-order sorted \
    --vqe-ansatz uccsd \
    --vqe-reps "${VQE_REPS}" \
    --vqe-restarts "${VQE_RESTARTS}" \
    --vqe-seed "${HARDCODED_VQE_SEED}" \
    --vqe-maxiter "${VQE_MAXITER}" \
    --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
    --qpe-shots "${QPE_SHOTS}" \
    --qpe-seed "${QPE_SEED}" \
    --skip-qpe \
    --initial-state-source "${INITIAL_STATE_SOURCE}" \
    --output-json "artifacts/hardcoded_uccsd_pipeline_L${L}.json" \
    --output-pdf "artifacts/hardcoded_uccsd_pipeline_L${L}.pdf" \
    --skip-pdf

  echo "Running hardcoded HVA pipeline for L=${L}..."
  "${PYTHON_BIN}" pipelines/hardcoded_hubbard_pipeline.py \
    --L "${L}" \
    --t "${T_VAL}" \
    --u "${U_VAL}" \
    --dv "${DV_VAL}" \
    --boundary "${BOUNDARY}" \
    --ordering "${ORDERING}" \
    --t-final "${T_FINAL}" \
    --num-times "${NUM_TIMES}" \
    --suzuki-order "${SUZUKI_ORDER}" \
    --trotter-steps "${TROTTER_STEPS}" \
    --term-order sorted \
    --vqe-ansatz hva \
    --vqe-reps "${VQE_REPS}" \
    --vqe-restarts "${VQE_RESTARTS}" \
    --vqe-seed "${HARDCODED_VQE_SEED}" \
    --vqe-maxiter "${VQE_MAXITER}" \
    --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
    --qpe-shots "${QPE_SHOTS}" \
    --qpe-seed "${QPE_SEED}" \
    --skip-qpe \
    --initial-state-source "${INITIAL_STATE_SOURCE}" \
    --output-json "artifacts/hardcoded_hva_pipeline_L${L}.json" \
    --output-pdf "artifacts/hardcoded_hva_pipeline_L${L}.pdf" \
    --skip-pdf

  echo "Running Qiskit UCCSD pipeline for L=${L}..."
  "${PYTHON_BIN}" pipelines/qiskit_hubbard_baseline_pipeline.py \
    --L "${L}" \
    --t "${T_VAL}" \
    --u "${U_VAL}" \
    --dv "${DV_VAL}" \
    --boundary "${BOUNDARY}" \
    --ordering "${ORDERING}" \
    --t-final "${T_FINAL}" \
    --num-times "${NUM_TIMES}" \
    --suzuki-order "${SUZUKI_ORDER}" \
    --trotter-steps "${TROTTER_STEPS}" \
    --term-order sorted \
    --vqe-reps "${VQE_REPS}" \
    --vqe-restarts "${VQE_RESTARTS}" \
    --vqe-seed "${QISKIT_VQE_SEED}" \
    --vqe-maxiter "${VQE_MAXITER}" \
    --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
    --qpe-shots "${QPE_SHOTS}" \
    --qpe-seed "${QPE_SEED}" \
    --skip-qpe \
    --initial-state-source "${INITIAL_STATE_SOURCE}" \
    --output-json "artifacts/qiskit_pipeline_L${L}.json" \
    --output-pdf "artifacts/qiskit_pipeline_L${L}.pdf" \
    --skip-pdf
}

run_triplet_for_L "2" "${L2_TROTTER_STEPS}" "${L2_NUM_TIMES}" "${L2_VQE_REPS}" "${L2_VQE_RESTARTS}" "${L2_VQE_MAXITER}"
run_triplet_for_L "3" "${L3_TROTTER_STEPS}" "${L3_NUM_TIMES}" "${L3_VQE_REPS}" "${L3_VQE_RESTARTS}" "${L3_VQE_MAXITER}"

echo "Building comparison PDFs/summary from generated JSON..."
"${PYTHON_BIN}" pipelines/compare_hva_uccsd_qiskit_pipeline.py \
  --l-values 2,3 \
  --no-run-pipelines \
  --with-per-l-pdfs \
  --artifacts-dir artifacts

overall_pass=1
if [[ ! -f artifacts/hva_uccsd_qiskit_summary.json ]]; then
  overall_pass=0
else
  compare_all_pass="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
path = Path("artifacts/hva_uccsd_qiskit_summary.json")
data = json.loads(path.read_text(encoding="utf-8"))
print("1" if bool(data.get("all_pass", False)) else "0")
PY
)"
  if [[ "${compare_all_pass}" != "1" ]]; then
    overall_pass=0
  fi
fi

for required in \
  artifacts/hva_uccsd_qiskit_L2_comparison.pdf \
  artifacts/hva_uccsd_qiskit_L3_comparison.pdf \
  artifacts/hva_uccsd_qiskit_bundle.pdf \
  artifacts/hva_uccsd_qiskit_L2_metrics.json \
  artifacts/hva_uccsd_qiskit_L3_metrics.json; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    overall_pass=0
  fi
done

if [[ "${overall_pass}" == "1" ]]; then
  echo "HVA/UCCSD/QISKIT REGRESSION PASS"
  exit 0
fi

echo "HVA/UCCSD/QISKIT REGRESSION FAIL"
exit 1
