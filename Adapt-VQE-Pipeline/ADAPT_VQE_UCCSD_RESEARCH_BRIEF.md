# ADAPT-VQE with UCCSD Pool: Failure Modes on the 1D Fermi-Hubbard Model

> **Purpose:** This document is a self-contained research brief designed to be fed to a deep-research LLM. It summarizes our findings from implementing and extensively testing ADAPT-VQE with a UCCSD operator pool on the 1D Fermi-Hubbard model. The core result is that the algorithm is **implemented correctly** (verified against Qiskit's independent implementation) but **fails to reach the exact ground-state energy** for L=2 and L=4, while succeeding for L=3. We seek insight into the root cause and potential remedies.

---

## 1. System Description

**Model:** 1D Fermi-Hubbard model at half-filling.

$$
H = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Hopping $t$ | 1.0 |
| On-site repulsion $U$ | 4.0 |
| Ratio $U/t$ | 4.0 (strongly correlated regime) |
| Boundary conditions | Periodic (PBC) |
| Filling | Half-filling ($N_e = L$) |

**System sizes tested:**
| $L$ (sites) | $N_\alpha, N_\beta$ | Qubits ($2L$) | Hilbert dim ($2^{2L}$) |
|:-----------:|:-------------------:|:--------------:|:----------------------:|
| 2 | 1, 1 | 4 | 16 |
| 3 | 2, 1 | 6 | 64 |
| 4 | 2, 2 | 8 | 256 |

**Encoding:**
- Jordan-Wigner transformation
- **Blocked** site ordering: qubits $0, \ldots, L{-}1$ → spin-up (alpha); qubits $L, \ldots, 2L{-}1$ → spin-down (beta)
- Hartree-Fock (HF) reference state: lowest-index orbitals filled per spin species

**HF reference bitstrings** (qubit $n$ → qubit 0 ordering):
| $L$ | HF bitstring | HF energy $\langle \text{HF} | H | \text{HF} \rangle$ |
|:---:|:------------:|:-------------------------------------------:|
| 2 | `0101` | 4.0 |
| 3 | `001011` | 4.0 |
| 4 | `00110011` | 8.0 |

---

## 2. ADAPT-VQE Implementation

### 2.1 Algorithm

Standard ADAPT-VQE (Grimsley et al., Nat. Commun. 2019):

1. Initialize $|\psi\rangle = |\text{HF}\rangle$.
2. Compute the commutator gradient for every operator $G_k$ in the pool:
$$
g_k = 2 \, \text{Im} \left( \langle \psi | H \, G_k | \psi \rangle \right)
$$
3. Select the operator $G_{k^*}$ with the largest $|g_k|$.
4. Append $e^{\theta_{k^*} G_{k^*}}$ to the ansatz.
5. Re-optimize **all** parameters $\{\theta_j\}$ via COBYLA to minimize $\langle \psi(\vec\theta) | H | \psi(\vec\theta) \rangle$.
6. Repeat until $\max_k |g_k| < \epsilon_{\text{grad}}$ or max depth is reached.

### 2.2 Operator Pool: UCCSD

The pool consists of spin-adapted **singles** and **doubles** excitation generators constructed from the UCCSD ansatz:

- **Singles:** $G^{(1)}_{p \to q} = a^\dagger_q a_p - a^\dagger_p a_q$  
  One generator per occupied→virtual orbital pair, within each spin species.
  
- **Doubles:** $G^{(2)}_{pq \to rs} = a^\dagger_r a^\dagger_s a_q a_p - \text{h.c.}$  
  One generator per pair of occupied→virtual orbital pairs. Includes same-spin ($\alpha\alpha$, $\beta\beta$) and cross-spin ($\alpha\beta$) excitations.

**Pool sizes:**
| $L$ | Singles | Doubles | Total pool |
|:---:|:-------:|:-------:|:----------:|
| 2 | 2 | 1 | 3 |
| 3 | 4 | 4 | 8 |
| 4 | 8 | 18 | 26 |

Each generator is expanded in the Pauli basis via Jordan-Wigner. The unitary $e^{\theta G}$ is applied using a first-order Trotter product formula over the Pauli terms. **We have verified** that this product formula is **exact** (not approximate) for UCCSD generators because the constituent Pauli terms within each generator commute, giving errors at machine precision (~$10^{-14}$). Trotterization is therefore **not** a source of error.

### 2.3 Optimization

- **Optimizer:** COBYLA (gradient-free, constraint-based)
- **Re-optimization:** Full re-optimization of all accumulated parameters at every ADAPT iteration
- **Convergence criteria:**
  - Gradient threshold $\epsilon_{\text{grad}} = 10^{-4}$ (hardcoded pipeline) or $10^{-6}$ (Qiskit pipeline)
  - Maximum ansatz depth: 30–50 operators
  - Energy change threshold: $10^{-8}$

### 2.4 Verification: Two Independent Implementations

We built **two completely independent** ADAPT-VQE implementations:

1. **Hardcoded pipeline** (`hardcoded_adapt_pipeline.py`): Custom statevector simulation using our own `apply_exp_pauli_polynomial` routine. Commutator gradient computed analytically. COBYLA optimization via SciPy.

2. **Qiskit pipeline** (`qiskit_adapt_pipeline.py`): Uses Qiskit Nature's `AdaptVQE` wrapped around `VQE` with a `UCCSD` operator pool and `StatevectorEstimator`.

Both implementations produce **identical results** to within optimizer noise, confirming correctness.

---

## 3. Results

### 3.1 Summary Table

| $L$ | $E_{\text{HF}}$ | $E_{\text{ADAPT}}$ (hardcoded) | $E_{\text{ADAPT}}$ (Qiskit) | $E_{\text{exact}}^{\text{sector}}$ | $\Delta E$ | Converged? |
|:---:|:-----:|:-------------------:|:------------------:|:--------------------:|:----------:|:----------:|
| 2 | 4.0 | **−0.50000** | **−0.50000** | −0.82843 | **0.328** | ❌ |
| 3 | 4.0 | **−1.27492** | **−1.27491** | −1.27492 | **8.2×10⁻⁸** | ✅ |
| 4 | 8.0 | **−1.00000** | **−1.00000** | −2.10275 | **1.103** | ❌ |

$E_{\text{exact}}^{\text{sector}}$ is the exact ground-state energy within the $(N_\alpha, N_\beta)$ particle-number sector matching the HF reference (half-filling). This is the physically relevant comparison target.

### 3.2 Full Hilbert Space Context

The sector-filtered exact GS is **not** the full Hilbert space GS for L=2 and L=4:

| $L$ | Full GS | Sector GS | Full GS sector |
|:---:|:-------:|:---------:|:--------------:|
| 2 | −1.000 | −0.828 | Different $(N_\alpha, N_\beta)$ sector |
| 3 | −3.123 | −1.275 | Different $(N_\alpha, N_\beta)$ sector |
| 4 | −3.419 | −2.103 | Different $(N_\alpha, N_\beta)$ sector |

ADAPT-VQE correctly preserves particle number (UCCSD generators conserve $N_\alpha$ and $N_\beta$), so the sector-filtered GS is the correct target.

### 3.3 Detailed Iteration Histories

#### L=2 (FAILS — stuck at E ≈ −0.5)

The pool has only 3 operators: 2 singles + 1 double.

| Depth | Selected operator | $\max|g|$ | Energy | $\Delta E$ |
|:-----:|:----------------|:---------:|:------:|:----------:|
| 1 | `uccsd_sing(alpha:0→1)` | 2.00 | −0.236 | −4.24 |
| 2 | `uccsd_sing(beta:2→3)` | 2.00 | **−0.500** | −4.24 |
| 3 | `uccsd_sing(beta:2→3)` | 7.1×10⁻⁴ | −0.500 | +6.1×10⁻⁸ |
| 4–30 | Only singles (repeats) | ~10⁻³ | −0.500 | ~10⁻⁸ |

**Key observations:**
- After depth 2, the energy is locked at −0.500 and all subsequent iterations add only singles.
- The double `uccsd_dbl(ab:0,2→1,3)` is **never selected** — its gradient is always smaller than the residual singles gradients.
- Over 30 iterations, the double is never picked. The operator frequency is: `sing(beta:2→3)` 24×, `sing(alpha:0→1)` 6×, and `dbl(ab:0,2→1,3)` **0×**.
- The energy −0.500 is **not an eigenvalue** of $H$. The L=2 spectrum is: {−1.0, −1.0, −0.828, 0 (×4), 1.0, 1.0, 3.0, 3.0, 4.0, 4.828, 5.0, 5.0, 8.0}. So ADAPT is stuck at a point that is not even a stationary state of the Hamiltonian — it is a **local minimum of the variational energy landscape** defined by the restricted ansatz.

#### L=3 (SUCCEEDS — reaches exact GS)

The pool has 8 operators: 4 singles + 4 doubles.

| Depth | Selected operator | $\max|g|$ | Energy | $\Delta E$ |
|:-----:|:----------------|:---------:|:------:|:----------:|
| 1 | `uccsd_sing(alpha:0→2)` | 2.00 | −0.236 | − |
| 2 | `uccsd_sing(beta:3→4)` | 2.00 | −0.484 | − |
| 3 | `uccsd_sing(beta:3→5)` | 2.35 | −0.858 | − |
| 4 | `uccsd_sing(alpha:1→2)` | 0.389 | −1.004 | − |
| **5** | **`uccsd_dbl(ab:1,3→2,4)`** | **0.186** | **−1.173** | − |
| 6 | `uccsd_sing(alpha:1→2)` | 0.314 | −1.271 | − |
| 7 | `uccsd_sing(beta:3→5)` | 0.050 | −1.2749 | − |
| 8–50 | Repeat singles | ~10⁻³ | −1.27492 | ~10⁻⁸ |

**Key observations:**
- One double is selected at depth 5 with a gradient of 0.186 — **large enough** to be picked by greedy selection.
- After depth 7, the energy is within $10^{-4}$ of the exact GS and slowly refines via repeated singles.
- Final accuracy: $|E_{\text{ADAPT}} - E_{\text{exact}}^{\text{sector}}| = 8.2 \times 10^{-8}$.
- Operator frequency: `sing(beta:3→4)` 28×, `sing(alpha:0→2)` 10×, `sing(beta:3→5)` 8×, `sing(alpha:1→2)` 3×, **`dbl(ab:1,3→2,4)` 1×**.

#### L=4 (FAILS — stuck at E ≈ −1.0)

The pool has 26 operators: 8 singles + 18 doubles.

| Depth | Selected operator | $\max|g|$ | Energy | $\Delta E$ |
|:-----:|:----------------|:---------:|:------:|:----------:|
| 1 | `uccsd_sing(alpha:0→3)` | 2.00 | +3.764 | −4.24 |
| 2 | `uccsd_sing(alpha:1→2)` | 2.00 | −0.472 | −4.24 |
| 3 | `uccsd_sing(beta:4→7)` | 2.00 | −0.736 | −0.264 |
| 4 | `uccsd_sing(beta:5→6)` | 2.00 | **−1.000** | −0.264 |
| 5 | `uccsd_dbl(ab:0,4→3,7)` | 3.4×10⁻⁴ | −1.000 | −2.7×10⁻⁸ |
| 6–22 | Mix of singles & doubles | ~10⁻⁴ | −1.000 | ~10⁻⁸ |

**Key observations:**
- After 4 singles, the energy is locked at −1.000 and the maximum gradient drops **5 orders of magnitude** (from 2.0 to 3.4×10⁻⁴).
- Despite using all 26 pool operators across 22 iterations (each used exactly once with `--adapt-no-repeats`), the energy never improves beyond −1.000.
- The energy −1.000 is **not an eigenvalue** of $H$ in the half-filling sector.
- The gradient of every remaining operator is $O(10^{-4})$ at this state, far below the initial gradients of $O(1)$. These tiny gradients produce negligible energy improvements (~$10^{-8}$).
- This is the same behavior in both our hardcoded pipeline AND Qiskit's AdaptVQE.

---

## 4. Diagnosis: Why Does ADAPT-VQE + UCCSD Fail?

### 4.1 What Has Been Ruled Out

We systematically ruled out the following potential causes:

| Hypothesis | Test | Result |
|:-----------|:-----|:-------|
| Code bug | Qiskit's independent AdaptVQE gives identical results | ❌ Not a bug |
| Trotter error | Compared product formula vs exact matrix exp; diff ~$10^{-14}$ | ❌ Exact for UCCSD |
| Gradient formula error | Commutator gradient matches parameter-shift gradient | ❌ Correct |
| Wrong reference energy | Sector-filtered exact diag matches Qiskit FermionicGaussianState | ❌ Correct |
| Optimizer failure | COBYLA converges; tried different maxiter, tolerances | ❌ Not optimizer |
| Insufficient depth | Ran up to depth 50 (L=2) and 22 (L=4, all ops used) | ❌ Not depth |

### 4.2 Root Cause Hypothesis: UCCSD Pool Expressibility Gap

The fundamental issue appears to be that the **UCCSD operator pool is insufficiently expressive** to connect the Hartree-Fock state to the true ground state for L=2 and L=4, given the greedy gradient-based selection rule.

**The Brillouin condition and the doubles gradient problem:**

At the HF reference state:
$$
g_k^{(2)} = 2\,\text{Im}\langle\text{HF}|H\,G_k^{(2)}|\text{HF}\rangle
$$

For doubles generators $G^{(2)}_{pq\to rs}$, this gradient is proportional to:
$$
\langle\text{HF}|H \left(a^\dagger_r a^\dagger_s a_q a_p - \text{h.c.}\right)|\text{HF}\rangle
$$

By the **generalized Brillouin theorem**, this is nonzero only if $H$ has a direct matrix element between the HF determinant and the doubly-excited determinant. In the Hubbard model, $H$ contains only one-body (hopping) and two-body (on-site) terms. The on-site interaction $U n_{i\uparrow}n_{i\downarrow}$ is diagonal and contributes zero. The hopping terms are one-body and connect the HF state to singly-excited determinants, not doubly-excited ones. Therefore:

$$
g_k^{(2)}\big|_{|\psi\rangle = |\text{HF}\rangle} = 0 \quad \text{(exactly)}
$$

**This means doubles always have zero gradient at the HF state.** Singles are selected first, and after a few singles are applied, the state $|\psi\rangle$ is a rotated Slater determinant. The doubles gradients at this rotated state are generically nonzero but may be very small — and this is exactly what we observe.

### 4.3 The Greedy Selection Trap

ADAPT-VQE uses a **greedy** selection rule: always pick the operator with the largest gradient. After the initial singles drive the energy to a Slater-determinant minimum (the "singles plateau"), the situation is:

1. **Singles residual gradients:** $O(10^{-3})$ — nonzero because COBYLA doesn't perfectly optimize, and because the energy surface has flat directions where adding redundant singles can produce tiny improvements.

2. **Doubles gradients:** $O(10^{-4})$ — genuinely small because the doubles correlations needed are weak at the current state.

3. **Greedy selection picks singles** because $10^{-3} > 10^{-4}$, even though the doubles are what's actually needed to break out of the singles subspace.

For L=3, this doesn't happen — the doubles gradient at depth 5 is 0.186, which is **large enough** to beat the residual singles gradients. This may be related to the odd electron count (3 electrons: 2α + 1β) creating an asymmetry that amplifies cross-spin correlations.

### 4.4 Why −0.5 and −1.0?

The stuck energies are **not eigenvalues** of $H$. They are:

- **Local minima of the variational energy landscape** restricted to the manifold of states reachable by products of singles excitations applied to the HF state.
- These are essentially the **best Slater determinants** (for L=2) or **best generalized Slater determinants** (for L=4) achievable within the singles-only UCCSD subspace.
- The states contain **zero correlation energy** beyond what orbital rotation (singles) can capture.

For the Hubbard model at $U/t = 4$ (strongly correlated), the correlation energy is a large fraction of the total energy:

| $L$ | $E_{\text{HF}}$ | $E_{\text{singles-only}}$ | $E_{\text{exact}}^{\text{sector}}$ | Correlation energy (beyond singles) |
|:---:|:------:|:-------------:|:--------------------:|:--------------------------:|
| 2 | 4.000 | −0.500 | −0.828 | −0.328 (39.7% of $E_{\text{exact}}$) |
| 4 | 8.000 | −1.000 | −2.103 | −1.103 (52.4% of $E_{\text{exact}}$) |

The doubles are responsible for capturing this missing correlation energy but are never (L=2) or insufficiently (L=4) activated by the greedy ADAPT selection.

---

## 5. What We Have Not Tried

The following modifications might resolve the issue but have **not been implemented or tested:**

### 5.1 Operator Pool Modifications

1. **Qubit-ADAPT pool** (Tang et al., PRX Quantum 2021): Replace fermionic UCCSD generators with individual Pauli strings. Each Pauli string is a separate pool operator. This dramatically increases the pool size but each operator is simpler, and the gradient landscape may not have the same bottleneck.

2. **Generalized singles and doubles (UCCGSD):** Include all orbital pairs, not just occupied→virtual. This enlarges the pool.

3. **Higher-order excitations (triples, quadruples):** UCCSDT or UCCSDTQ pools. Computationally expensive but may capture the missing correlations.

4. **Hardware-efficient ansatz (HEA) operators:** Use parameterized $R_Y$, $R_Z$, CNOT layers. Not chemically motivated but may have better expressibility.

5. **Symmetry-adapted pool:** Operators respecting lattice symmetries (translation, point group) of the Hubbard model. May concentrate the pool on the relevant symmetry sector.

### 5.2 Selection Rule Modifications

1. **Batch selection:** Instead of picking 1 operator per iteration, pick the top-$k$ operators simultaneously. This could allow doubles to be selected alongside singles.

2. **Mutual information-based selection:** Use entanglement measures rather than energy gradients to identify important operators.

3. **Second-order gradients (Hessian-based selection):** The gradient $g_k$ measures the first-order energy change. A second-order criterion could identify operators that, while having small first-order gradients, produce large energy changes when combined with parameter re-optimization.

4. **Random or scheduled injection of doubles:** Force doubles into the ansatz periodically, then re-optimize.

5. **Overlap-based selection:** Select operators that maximize overlap with the exact GS (not practical for real hardware, but useful for diagnostics).

### 5.3 Reference State Modifications

1. **Multi-reference ADAPT:** Start from a superposition or a CASCI-type reference instead of single-determinant HF.

2. **Symmetry-broken HF:** Use an unrestricted HF reference that breaks spin symmetry to better capture correlations.

3. **DMRG-initialized reference:** Use DMRG to prepare an approximate GS and then refine with ADAPT.

### 5.4 Optimization Modifications

1. **Global re-optimization with different optimizers:** L-BFGS-B, SPSA, or Adam instead of COBYLA.

2. **Parameter initialization from previous iteration:** Already implemented (warm-start), but could try random restarts.

---

## 6. Open Questions for Deep Research

1. **Is the UCCSD pool provably incomplete** for the half-filled Hubbard model at $U/t = 4$ with PBC and blocked ordering? Specifically, does the Lie algebra generated by the UCCSD operators span the full sector Hilbert space?

2. **Why does L=3 succeed but L=2 and L=4 fail?** Is there a symmetry-based explanation related to even vs. odd site count, or to the particle-number distribution (1α+1β vs. 2α+1β vs. 2α+2β)?

3. **Is the greedy selection the bottleneck, or is the pool itself insufficient?** If we could somehow force all doubles to be applied simultaneously with optimal parameters, would the UCCSD ansatz reach the exact GS?

4. **What is the minimum-depth UCCSD circuit** (with arbitrary operator ordering and optimal parameters) that achieves the exact sector GS for L=2 and L=4? This would distinguish "pool expressibility" from "selection rule" failures.

5. **How do alternative ADAPT pools (qubit-ADAPT, UCCGSD, etc.)** perform on this specific system? Are there published benchmarks for the 1D Hubbard model with ADAPT-VQE?

6. **Is there a known relationship** between the Brillouin condition, the structure of the Hubbard Hamiltonian, and ADAPT-VQE convergence failures?

7. **Would a non-greedy selection strategy** (e.g., mutual-information-based, or even random doubles injection) suffice to break out of the singles plateau?

---

## 7. Appendix: Detailed Run Configurations

### Hardcoded Pipeline

```
python -m pipelines.hardcoded_adapt_pipeline \
    --lattice-sites L \
    --t 1.0 --U 4.0 \
    --bc periodic \
    --ordering blocked \
    --indexing blocked \
    --adapt-max-depth 50 \
    --adapt-eps-grad 1e-4 \
    --adapt-eps-energy 1e-8 \
    --adapt-maxiter 300 \
    --adapt-allow-repeats \
    -o artifacts/hardcoded_adapt_pipeline_L{L}.json
```

### Qiskit Pipeline

```
python -m pipelines.qiskit_adapt_pipeline \
    --lattice-sites L \
    --t 1.0 --U 4.0 \
    --bc periodic \
    --ordering blocked \
    --indexing blocked \
    --adapt-gradient-threshold 1e-6 \
    --adapt-max-iterations 50 \
    --cobyla-maxiter 1000 \
    -o artifacts/qiskit_adapt_pipeline_L{L}.json
```

### Exact Sector-Filtered Ground State

Computed via full diagonalization of $H$ restricted to the $(N_\alpha, N_\beta)$ particle-number sector matching the HF reference.

---

## 8. Appendix: Code and Artifact Locations

```
Adapt-VQE-Pipeline/
├── pipelines/
│   ├── hardcoded_adapt_pipeline.py    # Custom ADAPT-VQE (952 lines)
│   ├── qiskit_adapt_pipeline.py       # Qiskit ADAPT-VQE (945 lines)
│   └── compare_adapt_pipelines.py     # Cross-validation script
├── artifacts/
│   ├── hardcoded_adapt_pipeline_L2.json
│   ├── hardcoded_adapt_pipeline_L3.json
│   ├── hardcoded_adapt_pipeline_L4.json
│   ├── qiskit_adapt_pipeline_L2.json
│   ├── qiskit_adapt_pipeline_L3.json
│   └── qiskit_adapt_pipeline_L4.json
└── ADAPT_VQE_UCCSD_RESEARCH_BRIEF.md  # This document
```

Source code for shared infrastructure:
```
src/quantum/
├── vqe_latex_python_pairs.py          # VQE utilities, exact diag, Pauli algebra
└── hubbard_latex_python_pairs.py      # Hubbard Hamiltonian builder
```
