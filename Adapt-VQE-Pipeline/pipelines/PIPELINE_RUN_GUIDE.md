# ADAPT-VQE Pipeline Run Guide

## Overview

This folder contains ADAPT-VQE pipelines for the Fermi-Hubbard
model (plus a unified ADAPT+VQE energy comparator), structured identically to the VQE pipeline in
`Fermi-Hamil-JW-VQE-TROTTER-PIPELINE/`.

| Script | Purpose |
|--------|---------|
| `hardcoded_adapt_pipeline.py` | Standard ADAPT-VQE using the repo's `HardcodedUCCSDAnsatz` UCCSD pool, parameter-shift gradients, and COBYLA inner optimizer. |
| `qiskit_adapt_pipeline.py` | Qiskit `AdaptVQE` wrapping VQE with the Qiskit UCCSD excitation pool and COBYLA inner optimizer. |
| `compare_adapt_pipelines.py` | Runs both pipelines, compares trajectories, emits pass/fail JSON and a bundled PDF report. |
| `compare_adapt_vqe_vqe_exact_filtered.py` | Runs/loads ADAPT + VQE pipelines and compares ADAPT/VQE energies against exact filtered ground-state energy. |

All scripts share the same JSON schema and Trotter-dynamics trajectory format.

---

## Quick Start

```bash
# From the repo root
cd Adapt-VQE-Pipeline

# 1) Run hardcoded ADAPT-VQE for L=2 (fast, ~30s)
python pipelines/hardcoded_adapt_pipeline.py --L 2

# 2) Run Qiskit ADAPT-VQE for L=2
python pipelines/qiskit_adapt_pipeline.py --L 2

# 3) Run both + compare
python pipelines/compare_adapt_pipelines.py --l-values 2,3

# 4) Unified ADAPT + VQE + exact-filtered energy comparison
python pipelines/compare_adapt_vqe_vqe_exact_filtered.py --l-values 2,3
```

---

## Hardcoded ADAPT Pipeline

```
python pipelines/hardcoded_adapt_pipeline.py [OPTIONS]
```

### Physics / lattice

| Flag | Default | Description |
|------|---------|-------------|
| `--L` | `2` | Number of lattice sites |
| `--t` | `1.0` | Hopping coefficient |
| `--u` | `4.0` | On-site interaction U |
| `--dv` | `0.0` | Uniform local potential v |
| `--boundary` | `periodic` | `periodic` or `open` |
| `--ordering` | `blocked` | `blocked` or `interleaved` |
| `--term-order` | `sorted` | `native` or `sorted` Pauli ordering |

### ADAPT-VQE controls

| Flag | Default | Description |
|------|---------|-------------|
| `--adapt-max-depth` | `20` | Maximum operators to add |
| `--adapt-eps-grad` | `1e-4` | Gradient convergence threshold |
| `--adapt-eps-energy` | `1e-8` | Energy convergence threshold |
| `--adapt-maxiter` | `300` | COBYLA maxiter per re-optimization |
| `--adapt-seed` | `7` | RNG seed |
| `--adapt-allow-repeats` | *(default)* | Allow pool operators to repeat |
| `--adapt-no-repeats` | | Forbid repeated pool operators |

### Trotter dynamics

| Flag | Default | Description |
|------|---------|-------------|
| `--t-final` | `20.0` | End time for Trotter trajectory |
| `--num-times` | `201` | Number of time-points |
| `--suzuki-order` | `2` | Suzuki-Trotter order (must be 2) |
| `--trotter-steps` | `64` | Number of Trotter steps |

### Initial state & output

| Flag | Default | Description |
|------|---------|-------------|
| `--initial-state-source` | `adapt_vqe` | `exact`, `adapt_vqe`, or `hf` |
| `--output-json` | `artifacts/hardcoded_adapt_pipeline_L{L}.json` | JSON path |
| `--output-pdf` | `artifacts/hardcoded_adapt_pipeline_L{L}.pdf` | PDF path |
| `--skip-pdf` | | Skip PDF generation |

### Example

```bash
python pipelines/hardcoded_adapt_pipeline.py \
    --L 3 --u 6.0 --adapt-max-depth 30 --adapt-eps-grad 1e-5 \
    --trotter-steps 128 --t-final 30.0
```

---

## Qiskit ADAPT Pipeline

```
python pipelines/qiskit_adapt_pipeline.py [OPTIONS]
```

### Physics / lattice

Same flags as hardcoded pipeline (`--L`, `--t`, `--u`, `--dv`, `--boundary`,
`--ordering`, `--term-order`).

### ADAPT-VQE controls

| Flag | Default | Description |
|------|---------|-------------|
| `--adapt-max-iterations` | `20` | Max ADAPT iterations |
| `--adapt-gradient-threshold` | `1e-4` | Gradient convergence threshold |
| `--adapt-cobyla-maxiter` | `300` | COBYLA maxiter per inner VQE |
| `--adapt-seed` | `7` | RNG seed |

### Trotter dynamics & output

Same flags as hardcoded pipeline.

### Example

```bash
python pipelines/qiskit_adapt_pipeline.py \
    --L 2 --adapt-max-iterations 30 --adapt-cobyla-maxiter 500
```

---

## Comparison Script

```
python pipelines/compare_adapt_pipelines.py [OPTIONS]
```

### Core flags

| Flag | Default | Description |
|------|---------|-------------|
| `--l-values` | `2,3` | Comma-separated L values to compare |
| `--run-pipelines` / `--no-run-pipelines` | `--run-pipelines` | Whether to execute the child pipelines |
| `--with-per-l-pdfs` | off | Also emit standalone per-L comparison PDFs |

### Physics + dynamics

Same flags as the individual pipelines (`--t`, `--u`, `--dv`, `--boundary`,
`--ordering`, `--t-final`, `--num-times`, `--suzuki-order`, `--trotter-steps`).

### Hardcoded ADAPT controls (prefixed `--hc-*`)

| Flag | Default |
|------|---------|
| `--hc-adapt-max-depth` | `20` |
| `--hc-adapt-eps-grad` | `1e-4` |
| `--hc-adapt-eps-energy` | `1e-8` |
| `--hc-adapt-maxiter` | `300` |
| `--hc-adapt-seed` | `7` |
| `--hc-adapt-allow-repeats` / `--hc-adapt-no-repeats` | allow |

### Qiskit ADAPT controls (prefixed `--qk-*`)

| Flag | Default |
|------|---------|
| `--qk-adapt-max-iterations` | `20` |
| `--qk-adapt-gradient-threshold` | `1e-4` |
| `--qk-adapt-cobyla-maxiter` | `300` |
| `--qk-adapt-seed` | `7` |

### Output

| Flag | Default |
|------|---------|
| `--initial-state-source` | `adapt_vqe` |
| `--artifacts-dir` | `artifacts/` |

### Example

```bash
python pipelines/compare_adapt_pipelines.py \
    --l-values 2,3 --with-per-l-pdfs \
    --hc-adapt-max-depth 30 --qk-adapt-max-iterations 30 \
    --trotter-steps 128
```

### Acceptance thresholds

The comparison script uses these thresholds (hard-coded, same as VQE comparison):

| Metric | Threshold |
|--------|-----------|
| `ground_state_energy_abs_delta` | 1e-8 |
| `fidelity_max_abs_delta` | 1e-4 |
| `energy_trotter_max_abs_delta` | 1e-3 |
| `n_up_site0_trotter_max_abs_delta` | 5e-3 |
| `n_dn_site0_trotter_max_abs_delta` | 5e-3 |
| `doublon_trotter_max_abs_delta` | 1e-3 |

---

## Unified ADAPT+VQE+Exact-Filtered Comparison

```
python pipelines/compare_adapt_vqe_vqe_exact_filtered.py [OPTIONS]
```

This script compares per-L energies across:
- hardcoded ADAPT (`adapt_vqe.energy`)
- qiskit ADAPT (`adapt_vqe.energy`)
- hardcoded VQE (`vqe.energy`)
- qiskit VQE (`vqe.energy`)
- exact filtered reference (`qiskit_vqe.vqe.exact_filtered_energy`)

### Core flags

| Flag | Default | Description |
|------|---------|-------------|
| `--l-values` | `2,3` | Comma-separated L values |
| `--run-pipelines` / `--no-run-pipelines` | `--run-pipelines` | Execute child pipelines or reuse JSONs |
| `--artifacts-dir` | `artifacts/` | Output directory for all JSON/CSV results |

### Inputs / settings

Supports the same physics/trotter knobs as the pipeline scripts (`--t`, `--u`,
`--dv`, `--boundary`, `--ordering`, `--t-final`, `--num-times`,
`--suzuki-order`, `--trotter-steps`) plus ADAPT/VQE optimizer controls.

### Output artifacts

| File | Description |
|------|-------------|
| `artifacts/adapt_vqe_vqe_exact_filtered_L{L}_metrics.json` | Per-L energies, deltas, and pass/fail checks |
| `artifacts/adapt_vqe_vqe_exact_filtered_summary.json` | Multi-L summary with `all_pass` |
| `artifacts/adapt_vqe_vqe_exact_filtered_table.csv` | Flat table for quick review/spreadsheets |
| `artifacts/adapt_vqe_vqe_exact_filtered_commands_run.txt` | Child commands executed by the script |

---

## Output Artifacts

| File | Description |
|------|-------------|
| `artifacts/hardcoded_adapt_pipeline_L{L}.json` | Full payload from hardcoded pipeline |
| `artifacts/hardcoded_adapt_pipeline_L{L}.pdf` | Plots from hardcoded pipeline |
| `artifacts/qiskit_adapt_pipeline_L{L}.json` | Full payload from Qiskit pipeline |
| `artifacts/qiskit_adapt_pipeline_L{L}.pdf` | Plots from Qiskit pipeline |
| `artifacts/hardcoded_vs_qiskit_adapt_L{L}_metrics.json` | Per-L comparison metrics |
| `artifacts/hardcoded_vs_qiskit_adapt_summary.json` | Overall summary with `all_pass` |
| `artifacts/hardcoded_vs_qiskit_adapt_bundle.pdf` | Bundled PDF report |
| `artifacts/adapt_pipeline_commands_run.txt` | Logged subprocess commands |

---

## JSON Schema (per pipeline)

```json
{
  "generated_utc": "...",
  "pipeline": "hardcoded_adapt | qiskit_adapt",
  "settings": { "L": 2, "t": 1.0, "u": 4.0, ... },
  "hamiltonian": { "num_qubits": 4, "num_terms": 9, "coefficients_exyz": [...] },
  "ground_state": { "exact_energy": -2.828..., "method": "matrix_diagonalization" },
  "adapt_vqe": {
    "success": true,
    "method": "hardcoded_adapt_vqe_uccsd | qiskit_adapt_vqe_uccsd",
    "energy": -2.828...,
    "ansatz_depth": 5,
    "num_parameters": 5,
    "optimal_point": [...],
    "stop_reason": "eps_grad | eps_energy | max_depth",
    ...
  },
  "initial_state": { "source": "adapt_vqe", "amplitudes_qn_to_q0": {...} },
  "trajectory": [
    { "time": 0.0, "fidelity": 1.0, "energy_exact": ..., "energy_trotter": ..., ... },
    ...
  ]
}
```

---

## Algorithm Summary

### Hardcoded ADAPT-VQE

1. Build UCCSD operator pool from `HardcodedUCCSDAnsatz.base_terms`
2. Start with HF reference state |ψ_ref⟩
3. Each iteration:
   - Compute parameter-shift gradients for all pool operators: $\frac{\partial E}{\partial \theta} = \frac{E(\theta+\pi/2) - E(\theta-\pi/2)}{2}$
   - Select operator with largest |gradient|
   - Append to ansatz, re-optimize ALL parameters with COBYLA
   - Check gradient and energy convergence
4. Trotter dynamics from the converged ADAPT state

### Qiskit ADAPT-VQE

1. Build UCCSD excitation pool via `qiskit_nature.second_q.circuit.library.UCCSD`
2. `AdaptVQE(solver=VQE(estimator, ansatz, COBYLA), gradient_threshold, max_iterations)`
3. `compute_minimum_eigenvalue(qop)` — Qiskit handles the ADAPT loop internally
4. Extract statevector, run Trotter dynamics
