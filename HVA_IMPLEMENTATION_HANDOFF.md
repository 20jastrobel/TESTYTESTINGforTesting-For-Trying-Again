# HVA/UCCSD Layer-Wise Implementation Handoff

## 1) Purpose and Scope

Goal: replicate this repo's HVA integration pattern in another pipeline, while preserving operator conventions and filtered-sector benchmarking. You already have most or all of these initial classes defined in this repo, but you do not have the full implemented runnable pipeline which this document describes how we implemented this HVA anzats to be runnable.

Critical rule: during ground-state (GS) optimization, the Hamiltonian is static. Any time-dependent potential is disabled at GS stage and only introduced later in dynamics -- We already do this in our VQE.

---

## 2) What Was Implemented (Code-Level)

- Hardcoded pipeline ansatz selection is exposed via:
  - `--vqe-ansatz {uccsd,hva}`
- Current mapping in hardcoded pipeline:
  - `uccsd` -> `HardcodedUCCSDLayerwiseAnsatz`
  - `hva` -> `HubbardLayerwiseAnsatz`
- Legacy classes are retained (not removed):
  - `HardcodedUCCSDAnsatz` (term-wise)
  - `HubbardTermwiseAnsatz` (term-wise)
- VQE payload includes:
  - `vqe.ansatz`
  - `vqe.parameterization` (now option to set to `"layerwise"`)
  - `vqe.exact_filtered_energy`
- GS baseline for fairness is filtered-sector exact energy via:
  - `exact_ground_energy_sector(...)` (we already do/have this)

---

## 3) Exact Source References

### `pydephasing/quantum/vqe_latex_python_pairs_test.py`

- Term-wise classes retained:
  - `HubbardTermwiseAnsatz` (line 388)
  - `HardcodedUCCSDAnsatz`(line )490
- New layer-wise classes added:
  - `HubbardLayerwiseAnsatz`
  - `HardcodedUCCSDLayerwiseAnsatz`

### `pipelines/hardcoded_hubbard_pipeline.py`

- Required symbols include both legacy and layer-wise classes:
  - in `_load_hardcoded_vqe_namespace(...)`
- Ansatz dispatch (`uccsd` vs `hva`) now targets layer-wise classes:
  - in `_run_hardcoded_vqe(...)` ansatz selection branch
- Filtered-sector exact energy in payload:
  - in `_run_hardcoded_vqe(...)` payload assembly
- CLI arg `--vqe-ansatz`:
  - in `parse_args()`

### `pipelines/compare_hva_uccsd_qiskit_pipeline.py`

- 3-way metrics + sanity extraction:
  - in the per-`L` metric extraction/sanity builder
- Numerical sanity condition:
  - `vqe.energy >= exact_filtered_energy - 1e-8`
  - enforced in the same metric/sanity pass

### `pipelines/run_hva_uccsd_qiskit_L2_L3.sh`

- Reproducible L=2,3 orchestrator for hardcoded HVA/UCCSD + qiskit reference.

### `pipelines/PIPELINE_RUN_GUIDE.md`

- Updated usage notes, including `--vqe-ansatz` and layer-wise mapping.

---

## 4) Ground-State + Time-Dependent Potential Rule

- GS solve uses static Hamiltonian only.
- Time-dependent potential is excluded during GS optimization.
- Time-dependent potential can be introduced only in dynamics stage.
- GS acceptance condition:
  - compare VQE energies to exact filtered-sector baseline from the same static Hamiltonian.
  - For L=2, it is expected that layer-wise HVA anzats does not produce small |Delta E|, but for higher L |Delta E| will appraoch zero.
---

## 5) Validation Commands (Verbatim)

### CLI smoke

```bash
python -m pipelines.hardcoded_hubbard_pipeline --help
```

Confirm `--vqe-ansatz {uccsd,hva}` appears.

### HVA smoke

```bash
python -m pipelines.hardcoded_hubbard_pipeline \
  --L 2 \
  --vqe-ansatz hva \
  --skip-qpe \
  --skip-pdf \
  --output-json /tmp/hardcoded_hva_L2_smoke.json
```

### Heavy runs used for current layer-wise validation

```bash
python -m pipelines.hardcoded_hubbard_pipeline \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 \
  --boundary periodic --ordering blocked \
  --t-final 20.0 --num-times 401 --suzuki-order 2 --trotter-steps 128 \
  --term-order sorted \
  --vqe-ansatz uccsd --vqe-reps 6 --vqe-restarts 6 --vqe-seed 7 --vqe-maxiter 2200 \
  --skip-qpe --initial-state-source vqe --skip-pdf \
  --output-json artifacts/hva_vs_uccsd_heavy/hardcoded_uccsd_pipeline_L2.json
```

```bash
python -m pipelines.hardcoded_hubbard_pipeline \
  --L 2 --t 1.0 --u 4.0 --dv 0.0 \
  --boundary periodic --ordering blocked \
  --t-final 20.0 --num-times 401 --suzuki-order 2 --trotter-steps 128 \
  --term-order sorted \
  --vqe-ansatz hva --vqe-reps 6 --vqe-restarts 6 --vqe-seed 7 --vqe-maxiter 2200 \
  --skip-qpe --initial-state-source vqe --skip-pdf \
  --output-json artifacts/hva_vs_uccsd_heavy/hardcoded_hva_pipeline_L2.json
```

```bash
python -m pipelines.hardcoded_hubbard_pipeline \
  --L 3 --t 1.0 --u 4.0 --dv 0.0 \
  --boundary periodic --ordering blocked \
  --t-final 20.0 --num-times 401 --suzuki-order 2 --trotter-steps 128 \
  --term-order sorted \
  --vqe-ansatz uccsd --vqe-reps 6 --vqe-restarts 6 --vqe-seed 7 --vqe-maxiter 2200 \
  --skip-qpe --initial-state-source vqe --skip-pdf \
  --output-json artifacts/hva_vs_uccsd_heavy/hardcoded_uccsd_pipeline_L3.json
```

```bash
python -m pipelines.hardcoded_hubbard_pipeline \
  --L 3 --t 1.0 --u 4.0 --dv 0.0 \
  --boundary periodic --ordering blocked \
  --t-final 20.0 --num-times 401 --suzuki-order 2 --trotter-steps 128 \
  --term-order sorted \
  --vqe-ansatz hva --vqe-reps 6 --vqe-restarts 6 --vqe-seed 7 --vqe-maxiter 2200 \
  --skip-qpe --initial-state-source vqe --skip-pdf \
  --output-json artifacts/hva_vs_uccsd_heavy/hardcoded_hva_pipeline_L3.json
```

### Optional 3-way compare run

```bash
python -m pipelines.compare_hva_uccsd_qiskit_pipeline \
  --l-values 2,3 \
  --t 1.0 --u 4.0 --dv 0.0 --boundary periodic --ordering blocked \
  --initial-state-source vqe \
  --skip-qpe \
  --with-per-l-pdfs \
  --artifacts-dir artifacts/hva_uccsd_heavy
```

---

## 6) Convergence Evidence (Current Layer-Wise Runs)

| L | E_exact (filtered) | E_uccsd (layerwise) | \|Δ_uccsd\| | E_hva (layerwise) | \|Δ_hva\| |
|---|---:|---:|---:|---:|---:|
| 2 | -0.828427124746 | -0.828427124674 | 7.19e-11 | 1.585786447090 | 2.41e+00 |
| 3 | -1.274917217635 | -1.274917140830 | 7.68e-08 | -1.274916488779 | 7.29e-07 |

Interpretation:

- Layer-wise UCCSD converges well for `L=2,3` under heavy settings above.
- Layer-wise HVA converges well for `L=3` but does not converge for `L=2` under strict shared-parameter tying (`2(+1)` parameters/layer).
- If strict `2(+1)` layer-wise HVA at `L=2` is unacceptable, the retained `HubbardTermwiseAnsatz` remains available as fallback.

---

## 7) Expected Artifacts

### JSON

- `artifacts/hva_vs_uccsd_heavy/hardcoded_uccsd_pipeline_L2.json`
- `artifacts/hva_vs_uccsd_heavy/hardcoded_hva_pipeline_L2.json`
- `artifacts/hva_vs_uccsd_heavy/hardcoded_uccsd_pipeline_L3.json`
- `artifacts/hva_vs_uccsd_heavy/hardcoded_hva_pipeline_L3.json`

### PDFs

- Direct HVA-vs-UCCSD PDFs:
  - `artifacts/hva_vs_uccsd_heavy/hva_vs_uccsd_L2_comparison.pdf`
  - `artifacts/hva_vs_uccsd_heavy/hva_vs_uccsd_L3_comparison.pdf`
- 3-way compare runner output (if run):
  - `artifacts/hva_uccsd_heavy/hva_uccsd_qiskit_L2_comparison.pdf`
  - `artifacts/hva_uccsd_heavy/hva_uccsd_qiskit_L3_comparison.pdf`
  - `artifacts/hva_uccsd_heavy/hva_uccsd_qiskit_bundle.pdf`

---

## API Notes

- `hardcoded_hubbard_pipeline.py`
  - `--vqe-ansatz {uccsd,hva}` (default `uccsd`)
  - VQE JSON adds `ansatz`, `parameterization`, `exact_filtered_energy`
- Existing keys remain intact for backward compatibility.

---

## Test Scenarios

### Smoke

- `--vqe-ansatz` appears in help.
- `--vqe-ansatz hva` run succeeds for `L=2` and emits:
  - `vqe.ansatz = "hva"`
  - finite `vqe.exact_filtered_energy`

### Schema

- UCCSD/HVA JSON both contain:
  - `vqe.energy`
  - `vqe.ansatz`
  - `vqe.parameterization`
  - `vqe.exact_filtered_energy`

### Numerical sanity

- For each `L` and method:
  - `vqe.energy >= exact_filtered_energy - 1e-8`

### Artifacts

- Expected JSONs exist under `artifacts/hva_vs_uccsd_heavy/`.
- Optional compare PDFs exist when compare runner is executed.

---

## Assumptions and Defaults

- GS benchmark baseline is exact filtered-sector energy.
- Time-dependent potential is off during GS optimization.
- Potential time dependence is only applied in dynamics.
- QPE is skipped in this GS-convergence validation path.

## 8) Post-Update Addendum (Do Not Replace Prior Notes)

“This addendum captures the latest implementation/validation delta and does not supersede user-authored notes above.”

### A) Latest Code-State Delta

- Current hardcoded runtime mapping:
  - `--vqe-ansatz uccsd` -> `HardcodedUCCSDLayerwiseAnsatz`
  - `--vqe-ansatz hva` -> `HubbardLayerwiseAnsatz`
- Legacy classes remain available (not removed):
  - `HardcodedUCCSDAnsatz`
  - `HubbardTermwiseAnsatz`
- VQE payload fields confirmed in current outputs:
  - `vqe.ansatz`
  - `vqe.parameterization = "layerwise"`
  - `vqe.exact_filtered_energy`
- Source anchors (already established above, reused):
  - `pydephasing/quantum/vqe_latex_python_pairs_test.py` (`HubbardLayerwiseAnsatz`, `HardcodedUCCSDLayerwiseAnsatz`)
  - `pipelines/hardcoded_hubbard_pipeline.py` (`_run_hardcoded_vqe(...)` ansatz dispatch + `exact_filtered_energy` payload)

### B) L=2 Layer-Wise HVA Diagnosis

- Current diagnosis: the large `L=2` HVA(layerwise) gap is most likely an ansatz expressivity/tying limitation, not a pipeline wiring bug.
- Supporting non-mutating diagnostics:
  - exact filtered (`L=2`): `-0.8284271247461902`
  - strict layerwise HVA (`reps=1`, 2 params) exhaustive grid best: `~1.587724568164`
  - strict layerwise HVA (`reps=2`) random best: `~1.585825256703`
  - increasing reps in pipeline remained near the same high-energy basin.
- Implication:
  - strict `2(+1)` shared-parameter tying can be too restrictive for `L=2` from the current HF reference state.

### C) Current Heavy-Run Numbers (Layer-Wise)

- `L=2`: `E_exact=-0.828427124746`, `E_uccsd=-0.828427124674`, `|Δ_uccsd|=7.19e-11`, `E_hva=1.585786447090`, `|Δ_hva|=2.41e+00`
- `L=3`: `E_exact=-1.274917217635`, `E_uccsd=-1.274917140830`, `|Δ_uccsd|=7.68e-08`, `E_hva=-1.274916488779`, `|Δ_hva|=7.29e-07`

### D) Porting Guidance for Time-Dependent Pipeline

- Ground-state optimization must remain static-Hamiltonian only.
- Time-dependent potential should enter only in the dynamics stage.
- Acceptance criterion for GS benchmarking:
  - compare only against exact filtered-sector baseline from the same static Hamiltonian.

## 9) Implementation Recipe (Exact Pattern Used Here)

This section is intentionally explicit so another agent can replicate the implementation style, not re-interpret it.

### A) Do Not Rebuild From Scratch

- We **did not** replace or delete the existing term-wise classes.
- We **added** new classes by subclassing:
  - `HubbardLayerwiseAnsatz(HubbardTermwiseAnsatz)`
  - `HardcodedUCCSDLayerwiseAnsatz(HardcodedUCCSDAnsatz)`
- We overrode class internals needed for layer-wise tying:
  - `_build_base_terms(...)`
  - `prepare_state(...)`

### B) Class Construction Pattern

```python
class HubbardLayerwiseAnsatz(HubbardTermwiseAnsatz):
    def _build_base_terms(self) -> None:
        # 1) Build primitive term lists (hop terms, onsite terms, optional potential terms)
        # 2) Keep full aggregated representatives in self.base_terms for parameter counting
        # 3) Store grouped primitive terms in self.layer_term_groups, e.g.:
        #    [("hop_layer", [...]), ("onsite_layer", [...]), ("potential_layer", [...])]
        pass

    def prepare_state(self, theta, psi_ref, **kwargs):
        # Per layer, use ONE shared theta per group, then apply that theta
        # across all primitive terms in the group sequentially.
        pass
```

```python
class HardcodedUCCSDLayerwiseAnsatz(HardcodedUCCSDAnsatz):
    def _build_base_terms(self) -> None:
        # 1) Build primitive singles/doubles generators
        # 2) Aggregate into group reps in self.base_terms
        # 3) Keep grouped primitives in self.layer_term_groups, e.g.:
        #    [("uccsd_singles_layer", [...]), ("uccsd_doubles_layer", [...])]
        pass

    def prepare_state(self, theta, psi_ref, **kwargs):
        # Per layer, apply one shared theta for singles group and one
        # shared theta for doubles group (if enabled).
        pass
```

### C) Runtime Dispatch Pattern

Use the same CLI flag, but map to the new layer-wise classes:

```python
if ansatz_name_s == "uccsd":
    ansatz = HardcodedUCCSDLayerwiseAnsatz(...)
elif ansatz_name_s == "hva":
    ansatz = HubbardLayerwiseAnsatz(...)
else:
    raise ValueError(...)
```

### D) Payload Pattern

Keep existing keys intact and add/emit:

```python
vqe_payload = {
    "ansatz": ansatz_name_s,
    "parameterization": "layerwise",
    "exact_filtered_energy": exact_filtered_energy,
    # plus existing fields already used by downstream scripts
}
```

### E) Namespace Loader Pattern

In the hardcoded pipeline namespace loader, require both legacy and new layer-wise classes so dispatch never fails when imported dynamically.

### F) Minimal Porting Checklist for Another Repo

- Keep existing term-wise classes untouched.
- Add the two layer-wise subclasses.
- Ensure `prepare_state(...)` uses group-shared thetas (not per-term unique thetas).
- Route existing ansatz CLI flags to the new layer-wise classes.
- Emit `vqe.parameterization` and `vqe.exact_filtered_energy` in output JSON.
- Validate with `L=2,3` heavy runs and compare against filtered-sector exact baseline.

### G) Important Nuance for Replication

- The `L=2` strict layer-wise HVA behavior observed here is linked to tying/expressivity, not a known dispatch/payload bug.
- If your target repo must achieve small `|ΔE|` for `L=2` with HVA, plan a relaxed tying variant (while still keeping the strict implementation for parity/reference).
