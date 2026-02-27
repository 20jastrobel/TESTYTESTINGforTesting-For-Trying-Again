# Repo-Specific Technical Specification for LLM Use

This document describes the exact mathematical and implementation conventions currently used in this repository snapshot.

Repository root:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing`

Primary implementation files:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/pauli_letters_module.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/pauli_words.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/pauli_polynomial_class.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/hubbard_latex_python_pairs.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/hartree_fock_reference_state.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/vqe_latex_python_pairs_test.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/hardcoded_hubbard_pipeline.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/qiskit_hubbard_baseline_pipeline.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/compare_hardcoded_vs_qiskit_pipeline.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/manual_compare_jsons.py`

## 1) Canonical Symbol and Ordering Conventions

### 1.1 Pauli alphabet
Internal symbols are lower-case:
- `e` for identity
- `x`, `y`, `z`

Upper-case `I/X/Y/Z` is accepted at some boundaries and normalized to `e/x/y/z`.

### 1.2 Pauli string qubit ordering (global invariant)
All Pauli strings are ordered:
- left to right = `q_(n-1) ... q_0`
- qubit `0` is the rightmost character

So for `nq=4`, pauli string `abcz` means:
- char 0 acts on `q3`
- char 1 acts on `q2`
- char 2 acts on `q1`
- char 3 acts on `q0`

### 1.3 Computational basis indexing
Statevector index uses little-endian qubit weights:
- basis index `idx = sum_q bit_q * 2^q`
- `q0` is least significant bit

Bitstring labels stored/read in many outputs use `q_(n-1)...q_0` text order.

## 2) Operator Algebra Layer

### 2.1 Pauli letter multiplication table
Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/pauli_letters_module.py`

Exact map (`a*b = phase * symbol`):
- `e*e=e`, `e*x=x`, `e*y=y`, `e*z=z`
- `x*e=x`, `x*x=e`, `x*y=+i z`, `x*z=-i y`
- `y*e=y`, `y*x=-i z`, `y*y=e`, `y*z=+i x`
- `z*e=z`, `z*x=+i y`, `z*y=-i x`, `z*z=e`

### 2.2 `PauliTerm`
Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/pauli_words.py`

Data model:
- `nq` qubits
- `pw`: list of `PauliLetter` of length `nq`
- `p_coeff`: complex coefficient

Multiplication:
- qubit-wise letter multiplication
- total coefficient multiplies old coefficients and all per-qubit phases

String conversion:
- `pw2strng()` concatenates letter symbols in stored order (`q_(n-1)...q_0`)

### 2.3 `PauliPolynomial`
Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/pauli_polynomial_class.py`

Data model:
- list of `PauliTerm`

Operations:
- addition concatenates and reduces
- multiplication distributes term-wise
- scalar multiplication multiplies each term coefficient

Reduction:
- combine terms with identical pauli string
- delete terms with `|coeff| < 1e-7`

Important: canonical `PauliTerm` in active code path is from `pauli_words.py` (not `qubitization_module.py`, which is absent in this snapshot).

## 3) Jordan-Wigner (JW) Operators as Implemented

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/pauli_polynomial_class.py`

For mode index `j` (`0 <= j < nq`):
- `c_j^dagger = 0.5 * X_j Z_{j-1}...Z_0 - 0.5i * Y_j Z_{j-1}...Z_0`
- `c_j       = 0.5 * X_j Z_{j-1}...Z_0 + 0.5i * Y_j Z_{j-1}...Z_0`

String construction in code:
- prefix `e` repeated `nq-j-1`
- then `x` or `y`
- suffix `z` repeated `j`

So qubit action is consistent with rightmost-char = `q0`.

### 3.1 Number operator
Implemented in two places consistently:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/hubbard_latex_python_pairs.py`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/vqe_latex_python_pairs_test.py`

Definition:
- `n_p = c_p^dagger c_p = (I - Z_p)/2`

String placement rule:
- char position for `Z_p` is `nq - 1 - p`

## 4) Lattice and Spin-Orbital Indexing

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/hubbard_latex_python_pairs.py`

### 4.1 Spin labels
- `SPIN_UP = 0`
- `SPIN_DN = 1`

### 4.2 Mode index maps
For site `i` and spin `sigma`:
- `interleaved`: `p(i,sigma) = 2*i + sigma`
- `blocked`: `p(i,up)=i`, `p(i,dn)=n_sites+i`

### 4.3 Lattice dims and coordinate linearization
- `dims` may be int (`L`) or tuple (`Lx, Ly, ...`)
- `n_sites = product(dims)`
- row-major linearization with x-fastest:
  - `i = x + Lx*(y + Ly*(z + ...))`

### 4.4 Nearest-neighbor edges
`bravais_nearest_neighbor_edges(dims, pbc)`:
- builds unique undirected edges
- increments each coordinate axis by +1
- wraps if periodic on that axis

1D examples:
- `L=2`: open and periodic both produce `[(0,1)]`
- `L=3`: open `[(0,1),(1,2)]`, periodic `[(0,1),(1,2),(0,2)]`

## 5) Hubbard Hamiltonian Construction (Hardcoded Core)

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/hubbard_latex_python_pairs.py`

Total Hamiltonian:
- `H = H_t + H_U + H_v`

Kinetic:
- `H_t = -t * sum_{(i,j) in edges} sum_{sigma in {up,dn}} (c^dag_{i,sigma} c_{j,sigma} + c^dag_{j,sigma} c_{i,sigma})`

Onsite:
- `H_U = U * sum_i n_{i,up} n_{i,dn}`

Potential:
- `H_v = - sum_i sum_sigma v_i n_{i,sigma}`

Potential parsing:
- `v=None` -> all zeros
- scalar -> broadcast
- dict `{site: value}` -> sparse map
- sequence length must equal `n_sites`

## 6) Hartree-Fock Reference State Conventions

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/hartree_fock_reference_state.py`

Occupied modes in HF determinant:
- fill alpha sites `0..n_alpha-1`
- fill beta sites `0..n_beta-1`
- map via chosen ordering (`blocked` or `interleaved`)

Functions:
- occupied qubit list
- bitstring in `q_(n-1)...q_0`
- statevector with amplitude 1 at integer basis index from occupied qubits

Self-check examples in file:
- `L=2,(1,1),blocked -> "0101"`
- `L=2,(1,1),interleaved -> "0011"`
- `L=3,(2,1),blocked -> "001011"`
- `L=3,(2,1),interleaved -> "000111"`

## 7) Statevector Pauli Action and Exponentials (Hardcoded)

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/vqe_latex_python_pairs_test.py`

### 7.1 Pauli action on state
`apply_pauli_string(state, pauli)` loops over basis index `idx` and qubits `q`:
- retrieves operator on qubit `q` as `ps[nq-1-q]`
- applies:
  - `e`: no change
  - `z`: phase `(-1)^bit`
  - `x`: flip bit `q` (`j ^= 1<<q`)
  - `y`: flip bit and multiply phase by `+i` if bit=0 else `-i`

### 7.2 Expectation values
- `<psi|P|psi>` by `vdot(psi, Ppsi)`
- `<psi|H|psi> = sum_j h_j <psi|P_j|psi>`
- identity term handled directly
- imaginary residual check: error if `|Im(E)| > 1e-8`

### 7.3 Single Pauli rotation
`R_P(angle) = exp(-i angle/2 * P)` implemented as:
- `cos(angle/2)*psi - i*sin(angle/2)*(P psi)`

### 7.4 Polynomial exponential (first-order product)
`apply_exp_pauli_polynomial(state,H,theta)` approximates:
- `exp(-i theta H) ~= prod_j exp(-i theta h_j P_j)`

Implementation details:
- optional term sorting by pauli label (`sort_terms=True` default)
- optional identity skip (`ignore_identity=True` default)
- coefficient tolerance default `1e-12`
- rejects non-negligible imaginary coefficients
- per-term rotation angle passed to `R_P` is `2*theta*h_j`, which reproduces `exp(-i theta h_j P_j)`

## 8) Ansatz Implementations

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/vqe_latex_python_pairs_test.py`

### 8.1 `HubbardTermwiseAnsatz` (HVA-like)
Per repetition (`reps`):
1. all hopping generators over `(edge,spin)`
2. all onsite generators over sites
3. optional potential generators over `(site,spin)` with nonzero `v_i`

State preparation:
- sequentially apply `exp(-i theta_k H_k)` via `apply_exp_pauli_polynomial`

### 8.2 `HardcodedUCCSDAnsatz`
Used by hardcoded pipeline VQE.

Generator forms:
- single: `G = i(T - T^dag)`, `T = c_a^dag c_i`
- double: `G = i(T - T^dag)`, `T = c_a^dag c_b^dag c_j c_i`

Excitation sets:
- occupied and virtual sets split per spin using chosen ordering
- include categories:
  - singles: alpha and beta
  - doubles: alpha-alpha, beta-beta, alpha-beta

State preparation:
- sequential `exp(-i theta_k G_k)` with same primitive as above

### 8.3 Half-filled particle rule used by pipelines
`half_filled_num_particles(L) = ((L+1)//2, L//2)`

## 9) VQE Optimization Logic

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/vqe_latex_python_pairs_test.py`

`vqe_minimize(H, ansatz, psi_ref, ...)`:
- objective: `E(theta)=<psi(theta)|H|psi(theta)>`
- restarts with Gaussian random init:
  - `x0 = initial_point_stddev * Normal(0,1)`
- SciPy path if available:
  - `scipy.optimize.minimize`
  - default method `SLSQP`
  - optional box bounds default `[-pi, pi]`
- fallback if SciPy unavailable:
  - coordinate search with shrinking step (`0.2 -> /2` until `<1e-6`)
- best result over restarts is returned

## 10) Exact Matrix Helpers

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pydephasing/quantum/vqe_latex_python_pairs_test.py`

- `pauli_matrix(pauli)` via Kronecker in string order `q_(n-1)...q_0`
- `hamiltonian_matrix(H)` sum of `coeff * pauli_matrix`
- `exact_ground_energy_sector(H, num_sites, num_particles, indexing)`:
  - builds full matrix
  - filters basis to fixed `(N_alpha,N_beta)` sector by counting bits at spin-orbital index sets
  - returns minimum eigenvalue of sector submatrix

## 11) Hardcoded Pipeline Behavior

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/hardcoded_hubbard_pipeline.py`

Flow:
1. Build hardcoded JW Hubbard Hamiltonian from core helpers
2. Build coefficient map `label_exyz -> coeff`
3. Exact diagonalization for global ground state
4. Run hardcoded VQE (`HardcodedUCCSDAnsatz` from notebook module)
5. Optional temporary Qiskit QPE adapter
6. Simulate exact and hardcoded Suzuki-2 Trotter trajectories
7. Emit JSON and optional PDF

Important details:
- term order for dynamics: `native` insertion order or lexicographically `sorted`
- Suzuki-2 step implementation:
  - `dt = t / trotter_steps`
  - forward half-step across terms, then reverse half-step
- exact evolution from eigendecomposition of dense Hamiltonian matrix

### 11.1 QPE adapter details
Inside `_run_qpe_adapter_qiskit` only:
- imports Qiskit locally
- constructs `SparsePauliOp` from hardcoded coefficients
- computes `bound = sum |Re(coeff)|`
- sets `evolution_time = pi / bound`
- if `num_qubits >= 8`: fastpath to `NumPyMinimumEigensolver`
- else run `PhaseEstimation`
- phase-to-energy map:
  - `phase_shift = phase if phase<=0.5 else phase-1`
  - `energy = -2 * bound * phase_shift`

## 12) Qiskit Baseline Pipeline Behavior

Defined in:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/qiskit_hubbard_baseline_pipeline.py`

Flow:
1. Build Qiskit Nature Fermi-Hubbard model and JW map
2. Run Qiskit VQE (UCCSD + HartreeFock) or eigensolver fallback
3. Run Qiskit QPE (or eigensolver fallback)
4. Run exact + Suzuki-2 trajectories (using Qiskit-derived terms)
5. Emit JSON and optional PDF

Hamiltonian construction details:
- lattice: `LineLattice(num_nodes=L, edge_parameter=-t, onsite_parameter=0.0, boundary_condition=...)`
- model: `FermiHubbardModel(..., onsite_interaction=U)`
- if ordering=`blocked`, permute spin-orbitals with:
  - `[0, L, 1, L+1, ..., L-1, 2L-1]`
- map with `JordanWignerMapper`
- add uniform potential operator for `dv`:
  - `-dv * sum_p n_p`
  - expanded to constant plus Z terms

VQE details:
- tries `StatevectorEstimator`
- ansatz: Qiskit `HartreeFock` + `UCCSD`
- optimizer: `SLSQP(maxiter=...)`
- restarts over random initial points (stddev 0.3)
- best restart kept
- also computes filtered exact energy in particle sector when possible

QPE details:
- similar phase-to-energy map as hardcoded adapter
- `num_qubits>=8` fastpath to `NumPyMinimumEigensolver`

## 13) Trajectory Observables and Caveat

Both pipeline files define:
- `_occupation_site0(psi, num_sites)`
- `_doublon_total(psi, num_sites)`

Current implementation interprets spin blocks as:
- up qubits: `0..num_sites-1`
- down qubits: `num_sites..2*num_sites-1`

This is consistent with `ordering="blocked"`, but not with `ordering="interleaved"`.
So if interleaved ordering is used, reported site occupations/doublon diagnostics are currently inconsistent with the Hamiltonian indexing convention.

## 14) Output JSON Schemas

Per-pipeline JSON top-level keys:
- `generated_utc`, `pipeline`, `settings`, `hamiltonian`, `ground_state`, `vqe`, `qpe`, `initial_state`, `trajectory`, `sanity`

`settings` includes:
- `L,t,u,dv,boundary,ordering,t_final,num_times,suzuki_order,trotter_steps,term_order,initial_state_source,skip_qpe`

`hamiltonian` includes:
- `num_qubits`, `num_terms`, `coefficients_exyz=[{label_exyz, coeff:{re,im}}]`

`initial_state`:
- `source`
- `amplitudes_qn_to_q0` map from bitstring label to `{re,im}`

`trajectory` row:
- `time`
- `fidelity`
- `energy_exact`, `energy_trotter`
- `n_up_site0_exact`, `n_up_site0_trotter`
- `n_dn_site0_exact`, `n_dn_site0_trotter`
- `doublon_exact`, `doublon_trotter`

`sanity.jw_reference`:
- either comparison stats against bundled reference JSONs, or `{checked:false, reason:...}`

## 15) Compare and Manual-Check Semantics

Compare pipeline:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/compare_hardcoded_vs_qiskit_pipeline.py`

Acceptance thresholds:
- ground-state energy abs delta <= `1e-8`
- max fidelity delta <= `1e-4`
- max trotter energy delta <= `1e-3`
- max `n_up_site0_trotter` delta <= `5e-3`
- max `n_dn_site0_trotter` delta <= `5e-3`
- max doublon delta <= `1e-3`

Metrics are computed on absolute per-time-point deltas for:
- fidelity
- energy_trotter
- n_up_site0_trotter
- n_dn_site0_trotter
- doublon_trotter

Compare `all_pass` additionally requires hardcoded-path Qiskit import isolation check to pass.

Manual checker:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/manual_compare_jsons.py`

Exit codes:
- `0`: pass
- `1`: threshold fail
- `2`: hard mismatch (settings/time-grid/schema)
- `3`: mismatch vs provided metrics JSON

## 16) Current Snapshot Concrete Values (Bundled Artifacts)

From:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/artifacts/hardcoded_pipeline_L2.json`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/artifacts/qiskit_pipeline_L2.json`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/artifacts/hardcoded_pipeline_L3.json`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/artifacts/qiskit_pipeline_L3.json`
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/artifacts/hardcoded_vs_qiskit_pipeline_summary.json`

Observed runs:
- L2: exact energy `-1.0`, both 11 terms, VQE around `-0.82842712`, QPE `-0.625`
- L3: exact energy `-3.1231056256176615`, both 22 terms, VQE around `-1.2749172`, QPE `-1.125`

Comparison summary:
- L2 pass = true
- L3 pass = false (dominant failure from fidelity max delta and one occupation threshold)
- overall `all_pass = false`

## 17) Canonical Pauli Dictionaries for `t=1, U=4, dv=0, periodic`

### L=2, blocked
`{`
`eeee: 2.0,` `eeez: -1.0,` `eexx: -0.5,` `eeyy: -0.5,` `eeze: -1.0,`
`ezee: -1.0,` `ezez: 1.0,` `xxee: -0.5,` `yyee: -0.5,` `zeee: -1.0,` `zeze: 1.0`
`}`

### L=2, interleaved
`{`
`eeee: 2.0,` `eeez: -1.0,` `eeze: -1.0,` `eezz: 1.0,` `exzx: -0.5,`
`eyzy: -0.5,` `ezee: -1.0,` `xzxe: -0.5,` `yzye: -0.5,` `zeee: -1.0,` `zzee: 1.0`
`}`

### L=3, blocked
`{`
`eeeeee: 3.0,` `eeeeez: -1.0,` `eeeexx: -0.5,` `eeeeyy: -0.5,` `eeeeze: -1.0,`
`eeexxe: -0.5,` `eeexzx: -0.5,` `eeeyye: -0.5,` `eeeyzy: -0.5,` `eeezee: -1.0,`
`eezeee: -1.0,` `eezeez: 1.0,` `exxeee: -0.5,` `eyyeee: -0.5,` `ezeeee: -1.0,`
`ezeeze: 1.0,` `xxeeee: -0.5,` `xzxeee: -0.5,` `yyeeee: -0.5,` `yzyeee: -0.5,`
`zeeeee: -1.0,` `zeezee: 1.0`
`}`

### L=3, interleaved
`{`
`eeeeee: 3.0,` `eeeeez: -1.0,` `eeeeze: -1.0,` `eeeezz: 1.0,` `eeexzx: -0.5,`
`eeeyzy: -0.5,` `eeezee: -1.0,` `eexzxe: -0.5,` `eeyzye: -0.5,` `eezeee: -1.0,`
`eezzee: 1.0,` `exzxee: -0.5,` `exzzzx: -0.5,` `eyzyee: -0.5,` `eyzzzy: -0.5,`
`ezeeee: -1.0,` `xzxeee: -0.5,` `xzzzxe: -0.5,` `yzyeee: -0.5,` `yzzzye: -0.5,`
`zeeeee: -1.0,` `zzeeee: 1.0`
`}`

## 18) Defaults in Current Code (not docs)

Hardcoded pipeline defaults:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/hardcoded_hubbard_pipeline.py`
- `num_times=201`, `trotter_steps=64`, `vqe_reps=1`, `vqe_restarts=1`, `vqe_maxiter=120`, initial state source `exact`

Qiskit pipeline defaults:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/qiskit_hubbard_baseline_pipeline.py`
- `num_times=201`, `trotter_steps=64`, `vqe_reps=2`, `vqe_restarts=3`, `vqe_maxiter=120`, initial state source `exact`

Compare pipeline defaults:
- `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/compare_hardcoded_vs_qiskit_pipeline.py`
- `l_values=2,3,4,5`, `num_times=201`, `trotter_steps=64`, `hardcoded_vqe_maxiter=600`, `qiskit_vqe_maxiter=600`, initial state source `vqe`

Note: `/Users/jakestrobel/Downloads/Testing-For-Trying-Again-main_copy-testforatestcuziknothing/pipelines/PIPELINE_RUN_GUIDE.md` still lists older compare defaults (`num_times=121`, `trotter_steps=32`, lower VQE maxiters). Treat code as source of truth.

## 19) Missing/Legacy References in This Snapshot

AGENTS mentions some files/data that are not present here:
- `qubitization_module.py` is not in active tree
- `pydephasing/quantum/exports/hubbard_jw_*.json` is not present
- `Tests/hubbard_jw_L4_L5_periodic_blocked_qiskit.json` is not present

Effect:
- reference-sanity checks in pipeline JSON often report unchecked with reason `no matching bundled reference for these settings`

## 20) LLM Instruction Block (Use This as Hard Constraints)

When generating theory or new code for this repo, enforce all of the following:
1. Use `e/x/y/z` internally; only convert to `I/X/Y/Z` at interfaces.
2. Pauli string order is always `q_(n-1)...q_0` (qubit 0 rightmost).
3. Basis index is little-endian (`sum bit_q 2^q`).
4. JW ladder operators must match:
   - `c_j^dagger = (X_j - iY_j)/2 * Z_{j-1}...Z_0`
   - `c_j = (X_j + iY_j)/2 * Z_{j-1}...Z_0`
5. Number operator is exactly `n_p=(I-Z_p)/2` with char index `nq-1-p`.
6. Mode indexing must support both:
   - interleaved `2*i+spin`
   - blocked `i` / `n_sites+i`
7. Hamiltonian builder must produce `H_t + H_U + H_v` exactly as implemented above.
8. Hardcoded VQE path is statevector-based, minimal dependencies, SciPy optional fallback.
9. Time evolution primitive should remain Pauli-exponential composition compatible with Suzuki/Trotter.
10. Keep pipeline JSON schema and comparison thresholds unchanged unless intentionally versioning.
11. If enabling `ordering=interleaved` in production analytics, update site-occupation/doublon extraction to match ordering (current code assumes blocked layout for those observables).

