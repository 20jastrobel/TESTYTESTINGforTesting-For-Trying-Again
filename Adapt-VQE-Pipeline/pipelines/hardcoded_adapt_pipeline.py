#!/usr/bin/env python3
"""Hardcoded ADAPT-VQE end-to-end Hubbard pipeline.

Flow:
1) Build Hubbard Hamiltonian (JW) from repo source-of-truth helpers.
2) Build UCCSD operator pool from HardcodedUCCSDAnsatz (same as VQE pipeline).
3) Run standard ADAPT-VQE: commutator gradients, one operator per
   iteration, COBYLA inner optimizer, optional repeats.
4) Run Suzuki-2 Trotter dynamics + exact dynamics from the ADAPT ground state.
5) Emit JSON + compact PDF artifact.

Uses the *same* src/quantum/ primitives as the regular VQE hardcoded pipeline.
No dependency on the old Adapt/ meta-learner code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import compatibility shim:
#   - Prefer src.quantum (legacy layout used by earlier checkpoints).
#   - Fallback to pydephasing.quantum (current repo layout).
# ---------------------------------------------------------------------------
try:
    from src.quantum.hubbard_latex_python_pairs import (
        build_hubbard_hamiltonian,
        build_hubbard_holstein_hamiltonian,
        boson_qubits_per_site,
    )
    from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.pauli_words import PauliTerm
    from src.quantum.vqe_latex_python_pairs import (
        AnsatzTerm,
        HardcodedUCCSDAnsatz,
        HubbardHolsteinLayerwiseAnsatz,
        HubbardTermwiseAnsatz,
        apply_exp_pauli_polynomial,
        apply_pauli_string,
        basis_state,
        exact_ground_energy_sector,
        expval_pauli_polynomial,
        half_filled_num_particles,
        hamiltonian_matrix,
        hartree_fock_bitstring,
        hubbard_holstein_reference_state,
    )
except Exception:
    from pydephasing.quantum.hubbard_latex_python_pairs import (
        build_hubbard_hamiltonian,
        build_hubbard_holstein_hamiltonian,
        boson_qubits_per_site,
    )
    from pydephasing.quantum.hartree_fock_reference_state import hartree_fock_statevector
    from pydephasing.quantum.pauli_polynomial_class import PauliPolynomial
    from pydephasing.quantum.pauli_words import PauliTerm
    from pydephasing.quantum.vqe_latex_python_pairs_test import (
        AnsatzTerm,
        HardcodedUCCSDAnsatz,
        HubbardHolsteinLayerwiseAnsatz,
        HubbardTermwiseAnsatz,
        apply_exp_pauli_polynomial,
        apply_pauli_string,
        basis_state,
        exact_ground_energy_sector,
        expval_pauli_polynomial,
        half_filled_num_particles,
        hamiltonian_matrix,
        hartree_fock_bitstring,
        hubbard_holstein_reference_state,
    )

EXACT_LABEL = "Exact_Hardcode"
EXACT_METHOD = "python_matrix_eigendecomposition"

PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


# ---------------------------------------------------------------------------
# Utility helpers (mirror hardcoded VQE pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "I").upper()


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((num_sites + 1) // 2, num_sites // 2)


def _collect_hardcoded_terms_exyz(poly: Any, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


# ---------------------------------------------------------------------------
# Compiled Pauli + Trotter (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
    dim = 1 << nq
    idx = np.arange(dim, dtype=np.int64)
    perm = idx.copy()
    phase = np.ones(dim, dtype=complex)
    for q in range(nq):
        op = label_exyz[nq - 1 - q]
        bits = ((idx >> q) & 1).astype(np.int8)
        sign = (1 - 2 * bits).astype(np.int8)
        if op == "e":
            continue
        if op == "x":
            perm ^= (1 << q)
            continue
        if op == "y":
            perm ^= (1 << q)
            phase *= 1j * sign
            continue
        if op == "z":
            phase *= sign
            continue
        raise ValueError(f"Unsupported Pauli symbol '{op}' in '{label_exyz}'.")
    return CompiledPauliAction(label_exyz=label_exyz, perm=perm, phase=phase)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    out = np.empty_like(psi)
    out[action.perm] = action.phase * psi
    return out


def _apply_exp_term(
    psi: np.ndarray, action: CompiledPauliAction, coeff: complex, alpha: float, tol: float = 1e-12,
) -> np.ndarray:
    if abs(coeff.imag) > tol:
        raise ValueError(f"Imaginary coefficient encountered for {action.label_exyz}: {coeff}")
    theta = float(alpha) * float(coeff.real)
    ppsi = _apply_compiled_pauli(psi, action)
    return math.cos(theta) * psi - 1j * math.sin(theta) * ppsi


def _evolve_trotter_suzuki2_absolute(
    psi0, ordered_labels, coeff_map, compiled_actions, time_value, trotter_steps,
) -> np.ndarray:
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return psi
    dt = float(time_value) / float(trotter_steps)
    half = 0.5 * dt
    for _ in range(trotter_steps):
        for label in ordered_labels:
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map[label], half)
        for label in reversed(ordered_labels):
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map[label], half)
    return _normalize_state(psi)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


# ---------------------------------------------------------------------------
# Observables (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _occupation_site0(psi: np.ndarray, num_sites: int) -> tuple[float, float]:
    probs = np.abs(psi) ** 2
    n_up = 0.0
    n_dn = 0.0
    for idx, prob in enumerate(probs):
        n_up += float((idx >> 0) & 1) * float(prob)
        n_dn += float((idx >> num_sites) & 1) * float(prob)
    return float(n_up), float(n_dn)


def _doublon_total(psi: np.ndarray, num_sites: int) -> float:
    probs = np.abs(psi) ** 2
    out = 0.0
    for idx, prob in enumerate(probs):
        count = 0
        for site in range(num_sites):
            up = (idx >> site) & 1
            dn = (idx >> (num_sites + site)) & 1
            count += int(up * dn)
        out += float(count) * float(prob)
    return float(out)


def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
    nq = int(round(math.log2(psi.size)))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) < cutoff:
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


# ============================================================================
# ADAPT-VQE core — standard algorithm, no meta-learner
# ============================================================================

@dataclass
class AdaptVQEResult:
    """Result container for the hardcoded ADAPT-VQE run."""
    energy: float
    theta: np.ndarray
    selected_ops: list[AnsatzTerm]
    history: list[dict[str, Any]]
    stop_reason: str
    nfev_total: int


def _build_uccsd_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
    """Build the UCCSD operator pool using HardcodedUCCSDAnsatz.base_terms.

    This reuses the exact same excitation generators the VQE pipeline uses,
    ensuring apples-to-apples comparison with the Qiskit UCCSD pool.
    """
    dummy_ansatz = HardcodedUCCSDAnsatz(
        dims=int(num_sites),
        num_particles=num_particles,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_cse_pool(
    num_sites: int,
    ordering: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
) -> list[AnsatzTerm]:
    """Build a CSE-style pool from the term-wise Hubbard ansatz base terms."""
    dummy_ansatz = HubbardTermwiseAnsatz(
        dims=int(num_sites),
        t=float(t),
        U=float(u),
        v=float(dv),
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_potential_terms=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_full_hamiltonian_pool(h_poly: Any, tol: float = 1e-12) -> list[AnsatzTerm]:
    """Build a pool with one generator per non-identity Hamiltonian Pauli term."""
    pool: list[AnsatzTerm] = []
    terms = h_poly.return_polynomial()
    if not terms:
        return pool
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label:
            continue
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(
                f"Non-negligible imaginary Hamiltonian coefficient for term {label}: {coeff}"
            )
        generator = PauliPolynomial("JW")
        generator.add_term(PauliTerm(nq, ps=label, pc=float(coeff.real)))
        pool.append(AnsatzTerm(label=f"ham_term({label})", polynomial=generator))
    return pool


def _build_hva_pool(
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
) -> list[AnsatzTerm]:
    # 1) Preserve the original HH layerwise generators used by the HVA form.
    layerwise = HubbardHolsteinLayerwiseAnsatz(
        dims=int(num_sites),
        J=float(t),
        U=float(u),
        omega0=float(omega0),
        g=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        v=None,
        v_t=None,
        v0=float(dv),
        t_eval=None,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )
    pool: list[AnsatzTerm] = list(layerwise.base_terms)

    # 2) Augment with fermionic UCCSD-style generators lifted to HH register.
    # This keeps the electron-number-conserving structure while allowing the
    # ADAPT search to explore a richer, HH-relevant operator manifold.
    n_sites = int(num_sites)
    boson_bits = n_sites * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    hf_num_particles = _half_filled_particles(n_sites)
    uccsd_kwargs = {
        "dims": n_sites,
        "num_particles": hf_num_particles,
        "include_singles": True,
        "include_doubles": True,
        "repr_mode": "JW",
        "indexing": str(ordering),
    }
    if str(boundary).strip().lower() == "periodic":
        try:
            uccsd_kwargs["pbc"] = True
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
        except TypeError as exc:
            if "pbc" not in str(exc):
                raise
            uccsd_kwargs.pop("pbc", None)
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
    else:
        uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
    for t_i in uccsd.base_terms:
        for term_idx, term in enumerate(t_i.polynomial.return_polynomial()):
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-12:
                raise ValueError(
                    f"Non-negligible imaginary UCCSD coefficient for HH pool term {t_i.label}: {coeff}"
                )
            base = str(term.pw2strng())
            padded = ("e" * boson_bits) + base
            lifted = PauliPolynomial("JW")
            lifted.add_term(PauliTerm(2 * n_sites + boson_bits, ps=padded, pc=float(coeff.real)))
            pool.append(AnsatzTerm(label=f"{t_i.label}_{term_idx}", polynomial=lifted))

    return pool


def _apply_pauli_polynomial(state: np.ndarray, poly: Any) -> np.ndarray:
    r"""Compute G|psi> where G is a PauliPolynomial (sum of weighted Pauli strings).

    G = \sum_j c_j P_j   =>   G|psi> = \sum_j c_j P_j|psi>
    """
    terms = poly.return_polynomial()
    nq = int(terms[0].nqubit())
    id_str = "e" * nq
    result = np.zeros_like(state)
    for term in terms:
        ps = term.pw2strng()
        coeff = complex(term.p_coeff)
        if abs(coeff) < 1e-15:
            continue
        if ps == id_str:
            result += coeff * state
        else:
            result += coeff * apply_pauli_string(state, ps)
    return result


def _commutator_gradient(
    h_poly: Any,
    pool_op: AnsatzTerm,
    psi_current: np.ndarray,
) -> float:
    r"""Compute dE/dtheta at theta=0 for appending pool_op to the current state.

    E(theta) = <psi|exp(+i theta G) H exp(-i theta G)|psi>

    The analytic gradient at theta=0 is:
        dE/dtheta|_0 = i <psi|[H, G]|psi> = 2 Im(<psi|H G|psi>)

    Since H is Hermitian: <psi|H G|psi> = <H psi | G psi>.

    This is exact and works for multi-term PauliPolynomial generators
    (unlike the parameter-shift rule which requires single-Pauli generators).
    """
    G_psi = _apply_pauli_polynomial(psi_current, pool_op.polynomial)
    H_psi = _apply_pauli_polynomial(psi_current, h_poly)
    hg_expect = np.vdot(H_psi, G_psi)  # <H psi | G psi> = <psi|H G|psi>
    return float(2.0 * hg_expect.imag)


def _prepare_adapt_state(
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
) -> np.ndarray:
    """Apply the current ADAPT ansatz: prod_k exp(-i theta_k G_k) |ref>."""
    psi = np.array(psi_ref, copy=True)
    for k, op in enumerate(selected_ops):
        psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta[k]))
    return psi


def _adapt_energy_fn(
    h_poly: Any,
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
) -> float:
    """Energy of the current ADAPT ansatz at parameters theta."""
    psi = _prepare_adapt_state(psi_ref, selected_ops, theta)
    return float(expval_pauli_polynomial(psi, h_poly))


def _run_hardcoded_adapt_vqe(
    *,
    h_poly: Any,
    num_sites: int,
    ordering: str,
    problem: str,
    adapt_pool: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    seed: int,
    allow_repeats: bool,
    finite_angle_fallback: bool,
    finite_angle: float,
    finite_angle_min_improvement: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run standard ADAPT-VQE and return (payload, psi_ground)."""
    if float(finite_angle) <= 0.0:
        raise ValueError("finite_angle must be > 0.")
    if float(finite_angle_min_improvement) < 0.0:
        raise ValueError("finite_angle_min_improvement must be >= 0.")

    t0 = time.perf_counter()
    hf_bits = "N/A"
    _ai_log(
        "hardcoded_adapt_vqe_start",
        L=int(num_sites),
        problem=str(problem),
        adapt_pool=str(adapt_pool),
        max_depth=int(max_depth),
        maxiter=int(maxiter),
        finite_angle_fallback=bool(finite_angle_fallback),
        finite_angle=float(finite_angle),
        finite_angle_min_improvement=float(finite_angle_min_improvement),
    )

    num_particles = _half_filled_particles(int(num_sites))
    problem_key = str(problem).strip().lower()
    if problem_key == "hh":
        psi_ref = np.asarray(
            hubbard_holstein_reference_state(
                dims=int(num_sites),
                num_particles=num_particles,
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                indexing=str(ordering),
            ),
            dtype=complex,
        )
    else:
        hf_bits = str(hartree_fock_bitstring(
            n_sites=int(num_sites),
            num_particles=num_particles,
            indexing=str(ordering),
        ))
        nq = 2 * int(num_sites)
        psi_ref = np.asarray(basis_state(nq, hf_bits), dtype=complex)

    # Build operator pool
    pool_key = str(adapt_pool).strip().lower()
    if problem_key == "hh":
        if pool_key != "hva":
            raise ValueError("For problem='hh', supported ADAPT pools are: hva")
        pool = _build_hva_pool(
            int(num_sites),
            float(t),
            float(u),
            float(omega0),
            float(g_ep),
            float(dv),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
        )
        method_name = "hardcoded_adapt_vqe_hva_hh"
    else:
        if pool_key == "uccsd":
            pool = _build_uccsd_pool(int(num_sites), num_particles, str(ordering))
            method_name = "hardcoded_adapt_vqe_uccsd"
        elif pool_key == "cse":
            pool = _build_cse_pool(
                int(num_sites),
                str(ordering),
                float(t),
                float(u),
                float(dv),
                str(boundary),
            )
            method_name = "hardcoded_adapt_vqe_cse"
        elif pool_key == "full_hamiltonian":
            pool = _build_full_hamiltonian_pool(h_poly)
            method_name = "hardcoded_adapt_vqe_full_hamiltonian"
        elif pool_key == "hva":
            raise ValueError(
                "For problem='hubbard', pool='hva' is not valid. "
                "Use uccsd, cse, or full_hamiltonian."
            )
        else:
            raise ValueError(f"Unsupported adapt pool '{adapt_pool}'.")
    if len(pool) == 0:
        raise ValueError(f"ADAPT pool '{pool_key}' produced no operators for problem='{problem_key}'.")
    _ai_log(
        "hardcoded_adapt_pool_built",
        pool_type=pool_key,
        pool_size=int(len(pool)),
    )

    # ADAPT-VQE main loop
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    history: list[dict[str, Any]] = []
    nfev_total = 0
    stop_reason = "max_depth"

    # Pool availability tracking (for no-repeat mode)
    available_indices = set(range(len(pool)))

    energy_current = float(expval_pauli_polynomial(psi_ref, h_poly))
    nfev_total += 1
    _ai_log("hardcoded_adapt_initial_energy", energy=energy_current)

    from scipy.optimize import minimize as scipy_minimize

    for depth in range(int(max_depth)):
        iter_t0 = time.perf_counter()

        # 1) Compute the current state
        psi_current = _prepare_adapt_state(psi_ref, selected_ops, theta)

        # 2) Compute commutator gradients for all pool operators
        gradients = np.zeros(len(pool), dtype=float)
        for i in available_indices:
            gradients[i] = _commutator_gradient(h_poly, pool[i], psi_current)

        # Find the operator with the largest gradient magnitude
        grad_magnitudes = np.abs(gradients)
        # Zero out unavailable operators so they can't be selected
        mask = np.zeros(len(pool), dtype=bool)
        for i in available_indices:
            mask[i] = True
        grad_magnitudes[~mask] = 0.0

        best_idx = int(np.argmax(grad_magnitudes))
        max_grad = float(grad_magnitudes[best_idx])

        _ai_log(
            "hardcoded_adapt_iter",
            depth=int(depth + 1),
            max_grad=float(max_grad),
            best_op=str(pool[best_idx].label),
            energy=float(energy_current),
        )

        # 3) Check gradient convergence (with optional finite-angle fallback)
        selection_mode = "gradient"
        init_theta = 0.0
        fallback_scan_size = 0
        fallback_best_probe_delta_e = None
        fallback_best_probe_theta = None
        if max_grad < float(eps_grad):
            if bool(finite_angle_fallback) and available_indices:
                fallback_scan_size = int(len(available_indices))
                best_probe_energy = float(energy_current)
                best_probe_idx = None
                best_probe_theta = None

                for idx in available_indices:
                    trial_ops = selected_ops + [pool[idx]]
                    for trial_theta in (float(finite_angle), -float(finite_angle)):
                        trial_theta_vec = np.append(theta, trial_theta)
                        probe_energy = _adapt_energy_fn(h_poly, psi_ref, trial_ops, trial_theta_vec)
                        nfev_total += 1
                        if probe_energy < best_probe_energy:
                            best_probe_energy = float(probe_energy)
                            best_probe_idx = int(idx)
                            best_probe_theta = float(trial_theta)

                fallback_best_probe_delta_e = float(best_probe_energy - energy_current)
                fallback_best_probe_theta = float(best_probe_theta) if best_probe_theta is not None else None
                _ai_log(
                    "hardcoded_adapt_fallback_scan",
                    depth=int(depth + 1),
                    scan_size=int(fallback_scan_size),
                    finite_angle=float(finite_angle),
                    best_probe_idx=(int(best_probe_idx) if best_probe_idx is not None else None),
                    best_probe_op=(str(pool[best_probe_idx].label) if best_probe_idx is not None else None),
                    best_probe_delta_e=float(fallback_best_probe_delta_e),
                )

                if (
                    best_probe_idx is not None
                    and (energy_current - best_probe_energy) > float(finite_angle_min_improvement)
                ):
                    best_idx = int(best_probe_idx)
                    selection_mode = "finite_angle_fallback"
                    init_theta = float(best_probe_theta)
                    _ai_log(
                        "hardcoded_adapt_fallback_selected",
                        depth=int(depth + 1),
                        selected_idx=int(best_idx),
                        selected_op=str(pool[best_idx].label),
                        init_theta=float(init_theta),
                        probe_delta_e=float(fallback_best_probe_delta_e),
                    )
                else:
                    stop_reason = "eps_grad"
                    _ai_log(
                        "hardcoded_adapt_converged_grad",
                        max_grad=float(max_grad),
                        eps_grad=float(eps_grad),
                        fallback_attempted=True,
                        fallback_best_probe_delta_e=float(fallback_best_probe_delta_e),
                        finite_angle_min_improvement=float(finite_angle_min_improvement),
                    )
                    break
            else:
                stop_reason = "eps_grad"
                _ai_log("hardcoded_adapt_converged_grad", max_grad=float(max_grad), eps_grad=float(eps_grad))
                break

        # 4) Append the selected operator
        selected_ops.append(pool[best_idx])
        theta = np.append(theta, float(init_theta))
        if not allow_repeats:
            available_indices.discard(best_idx)

        # 5) Re-optimize ALL parameters with COBYLA
        energy_prev = energy_current

        def _obj(x: np.ndarray) -> float:
            return _adapt_energy_fn(h_poly, psi_ref, selected_ops, x)

        result = scipy_minimize(
            _obj,
            theta,
            method="COBYLA",
            options={"maxiter": int(maxiter), "rhobeg": 0.3},
        )
        theta = np.asarray(result.x, dtype=float)
        energy_current = float(result.fun)
        nfev_total += int(getattr(result, "nfev", 0))

        history.append({
            "depth": int(depth + 1),
            "selected_op": str(pool[best_idx].label),
            "pool_index": int(best_idx),
            "selection_mode": str(selection_mode),
            "init_theta": float(init_theta),
            "max_grad": float(max_grad),
            "fallback_scan_size": int(fallback_scan_size),
            "fallback_best_probe_delta_e": (
                float(fallback_best_probe_delta_e) if fallback_best_probe_delta_e is not None else None
            ),
            "fallback_best_probe_theta": (
                float(fallback_best_probe_theta) if fallback_best_probe_theta is not None else None
            ),
            "energy_before_opt": float(energy_prev),
            "energy_after_opt": float(energy_current),
            "delta_energy": float(energy_current - energy_prev),
            "nfev_opt": int(getattr(result, "nfev", 0)),
            "opt_success": bool(getattr(result, "success", False)),
            "iter_elapsed_s": float(time.perf_counter() - iter_t0),
        })

        _ai_log(
            "hardcoded_adapt_iter_done",
            depth=int(depth + 1),
            energy=float(energy_current),
            delta_e=float(energy_current - energy_prev),
        )

        # 6) Check energy convergence
        if abs(energy_current - energy_prev) < float(eps_energy):
            stop_reason = "eps_energy"
            _ai_log(
                "hardcoded_adapt_converged_energy",
                delta_e=float(abs(energy_current - energy_prev)),
                eps_energy=float(eps_energy),
            )
            break

        # Check if pool exhausted
        if not allow_repeats and not available_indices:
            stop_reason = "pool_exhausted"
            _ai_log("hardcoded_adapt_pool_exhausted")
            break

    # Build final state
    psi_adapt = _prepare_adapt_state(psi_ref, selected_ops, theta)
    psi_adapt = _normalize_state(psi_adapt)

    elapsed = time.perf_counter() - t0

    # Exact sector-filtered energy for comparison
    # ADAPT-VQE preserves particle number, so compare against the GS
    # within the same (n_alpha, n_beta) sector as the HF reference.
    exact_gs = exact_ground_energy_sector(
        h_poly,
        num_sites=int(num_sites),
        num_particles=num_particles,
        indexing=str(ordering),
    )

    payload = {
        "success": True,
        "method": method_name,
        "energy": float(energy_current),
        "exact_gs_energy": float(exact_gs),
        "delta_e": float(energy_current - exact_gs),
        "abs_delta_e": float(abs(energy_current - exact_gs)),
        "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        "ansatz_depth": int(len(selected_ops)),
        "num_parameters": int(theta.size),
        "optimal_point": [float(x) for x in theta.tolist()],
        "operators": [str(op.label) for op in selected_ops],
        "pool_size": int(len(pool)),
        "pool_type": str(pool_key),
        "stop_reason": str(stop_reason),
        "nfev_total": int(nfev_total),
        "allow_repeats": bool(allow_repeats),
        "finite_angle_fallback": bool(finite_angle_fallback),
        "finite_angle": float(finite_angle),
        "finite_angle_min_improvement": float(finite_angle_min_improvement),
        "history": history,
        "elapsed_s": float(elapsed),
        "hf_bitstring_qn_to_q0": str(hf_bits),
    }

    _ai_log(
        "hardcoded_adapt_vqe_done",
        L=int(num_sites),
        energy=float(energy_current),
        exact_gs=float(exact_gs),
        abs_delta_e=float(abs(energy_current - exact_gs)),
        depth=int(len(selected_ops)),
        stop_reason=str(stop_reason),
        elapsed_sec=round(elapsed, 6),
    )
    return payload, psi_adapt


# ---------------------------------------------------------------------------
# Trajectory simulation (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _simulate_trajectory(
    *,
    num_sites: int,
    psi0: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else int(np.log2(max(1, psi0.size)))
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_adapt_trajectory_start",
        L=int(num_sites),
        num_times=n_times,
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
    )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        tv = float(time_val)
        psi_exact = evecs @ (np.exp(-1j * evals * tv) * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)

        psi_trot = _evolve_trotter_suzuki2_absolute(
            psi0, ordered_labels_exyz, coeff_map_exyz, compiled, tv, int(trotter_steps),
        )

        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)
        n_up_trot, n_dn_trot = _occupation_site0(psi_trot, num_sites)

        rows.append({
            "time": tv,
            "fidelity": fidelity,
            "energy_exact": _expectation_hamiltonian(psi_exact, hmat),
            "energy_trotter": _expectation_hamiltonian(psi_trot, hmat),
            "n_up_site0_exact": n_up_exact,
            "n_up_site0_trotter": n_up_trot,
            "n_dn_site0_exact": n_dn_exact,
            "n_dn_site0_trotter": n_dn_trot,
            "doublon_exact": _doublon_total(psi_exact, num_sites),
            "doublon_trotter": _doublon_total(psi_trot, num_sites),
        })
        exact_states.append(psi_exact)
        if idx == 0 or idx == n_times - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "hardcoded_adapt_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=tv,
                fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log("hardcoded_adapt_trajectory_done", L=int(num_sites), num_times=n_times)
    return rows, exact_states


# ---------------------------------------------------------------------------
# PDF writer (compact — mirrors VQE pipeline)
# ---------------------------------------------------------------------------

def _current_command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _render_text_page(
    pdf: PdfPages, lines: list[str], *, fontsize: int = 9, line_spacing: float = 0.028, max_line_width: int = 115,
) -> None:
    expanded: list[str] = []
    for raw in lines:
        if len(raw) <= max_line_width:
            expanded.append(raw)
        else:
            expanded.extend(textwrap.wrap(raw, width=max_line_width, subsequent_indent="    "))
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    x0, y = 0.05, 0.95
    for line in expanded:
        ax.text(x0, y, line, transform=ax.transAxes, va="top", ha="left", family="monospace", fontsize=fontsize)
        y -= line_spacing
        if y < 0.02:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(11.0, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    with PdfPages(str(pdf_path)) as pdf:
        # Command page
        _render_text_page(pdf, [
            "Executed Command",
            "",
            "Script: pipelines/hardcoded_adapt_pipeline.py",
            "",
            run_command,
        ], fontsize=10, line_spacing=0.03, max_line_width=110)

        settings = payload.get("settings", {})
        adapt = payload.get("adapt_vqe", {})

        # Settings + ADAPT summary page
        lines = [
            "Hardcoded ADAPT-VQE Pipeline Summary",
            "",
            f"L={settings.get('L')}  t={settings.get('t')}  u={settings.get('u')}  dv={settings.get('dv')}",
            f"boundary={settings.get('boundary')}  ordering={settings.get('ordering')}",
            f"initial_state_source={settings.get('initial_state_source')}",
            "",
            f"ADAPT-VQE energy:  {adapt.get('energy')}",
            f"Exact GS energy:   {adapt.get('exact_gs_energy')}",
            f"|ΔE|:              {adapt.get('abs_delta_e')}",
            f"Ansatz depth:      {adapt.get('ansatz_depth')}",
            f"Pool size:         {adapt.get('pool_size')}",
            f"Stop reason:       {adapt.get('stop_reason')}",
            f"Total nfev:        {adapt.get('nfev_total')}",
            f"Elapsed:           {adapt.get('elapsed_s'):.2f}s" if adapt.get("elapsed_s") is not None else "",
            "",
            "Selected operators:",
        ]
        for op_label in (adapt.get("operators") or []):
            lines.append(f"  {op_label}")
        _render_text_page(pdf, lines)

        # Trajectory plots
        rows = payload.get("trajectory", [])
        if rows:
            times = np.array([r["time"] for r in rows])
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
            ax_f, ax_e = axes[0]
            ax_n, ax_d = axes[1]

            ax_f.plot(times, [r["fidelity"] for r in rows], color="#0b3d91")
            ax_f.set_title("Fidelity (Trotter vs Exact)")
            ax_f.set_ylabel("F(t)")
            ax_f.grid(alpha=0.25)

            ax_e.plot(times, [r["energy_trotter"] for r in rows], label="Trotter", color="#d62728")
            ax_e.plot(times, [r["energy_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_e.set_title("Energy")
            ax_e.set_ylabel("E(t)")
            ax_e.legend(fontsize=8)
            ax_e.grid(alpha=0.25)

            ax_n.plot(times, [r["n_up_site0_trotter"] for r in rows], label="n_up trot", color="#17becf")
            ax_n.plot(times, [r["n_dn_site0_trotter"] for r in rows], label="n_dn trot", color="#9467bd")
            ax_n.set_title("Site-0 Occupations (Trotter)")
            ax_n.set_xlabel("Time")
            ax_n.legend(fontsize=8)
            ax_n.grid(alpha=0.25)

            ax_d.plot(times, [r["doublon_trotter"] for r in rows], label="Trotter", color="#e377c2")
            ax_d.plot(times, [r["doublon_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_d.set_title("Doublon")
            ax_d.set_xlabel("Time")
            ax_d.legend(fontsize=8)
            ax_d.grid(alpha=0.25)

            fig.suptitle(f"Hardcoded ADAPT-VQE Pipeline L={settings.get('L')}", fontsize=13)
            fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hardcoded ADAPT-VQE Hubbard pipeline.")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--problem", choices=["hubbard", "hh"], default="hubbard")
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=0.0)
    p.add_argument("--g-ep", type=float, default=0.0, help="Holstein electron-phonon coupling g.")
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--term-order", choices=["native", "sorted"], default="sorted")

    # ADAPT-VQE controls
    p.add_argument("--adapt-pool", choices=["uccsd", "cse", "full_hamiltonian", "hva"], default="uccsd")
    p.add_argument("--adapt-max-depth", type=int, default=20)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-4)
    p.add_argument("--adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--adapt-maxiter", type=int, default=300, help="COBYLA maxiter per re-optimization")
    p.add_argument("--adapt-seed", type=int, default=7)
    p.set_defaults(adapt_allow_repeats=True)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.set_defaults(adapt_finite_angle_fallback=True)
    p.add_argument(
        "--adapt-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_true",
        help="If gradients are below threshold, scan finite ±theta probes to continue ADAPT when beneficial.",
    )
    p.add_argument(
        "--adapt-no-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_false",
        help="Disable finite-angle fallback and stop immediately when gradients are below threshold.",
    )
    p.add_argument(
        "--adapt-finite-angle",
        type=float,
        default=0.1,
        help="Probe angle theta used by finite-angle fallback (tests ±theta).",
    )
    p.add_argument(
        "--adapt-finite-angle-min-improvement",
        type=float,
        default=1e-12,
        help="Minimum required energy drop from finite-angle probe to accept fallback selection.",
    )

    # Trotter dynamics
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--num-times", type=int, default=201)
    p.add_argument("--suzuki-order", type=int, default=2)
    p.add_argument("--trotter-steps", type=int, default=64)

    p.add_argument("--initial-state-source", choices=["exact", "adapt_vqe", "hf"], default="adapt_vqe")

    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--skip-pdf", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("hardcoded_adapt_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (artifacts_dir / f"hardcoded_adapt_pipeline_L{args.L}.json")
    output_pdf = args.output_pdf or (artifacts_dir / f"hardcoded_adapt_pipeline_L{args.L}.pdf")

    # 1) Build Hamiltonian
    problem_key = str(args.problem).strip().lower()
    if problem_key == "hh":
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=int(args.L),
            J=float(args.t),
            U=float(args.u),
            omega0=float(args.omega0),
            g=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            v_t=None,
            v0=float(args.dv),
            t_eval=None,
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
            include_zero_point=True,
        )
    else:
        h_poly = build_hubbard_hamiltonian(
            dims=int(args.L),
            t=float(args.t),
            U=float(args.u),
            v=float(args.dv),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
        )

    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    _ai_log("hardcoded_adapt_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "native":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)

    hmat = _build_hamiltonian_matrix(coeff_map_exyz)

    # Sector-filtered exact ground state: ADAPT-VQE preserves particle number,
    # so compare against the GS within the same (n_alpha, n_beta) sector.
    num_particles_main = _half_filled_particles(int(args.L))
    gs_energy_exact = exact_ground_energy_sector(
        h_poly,
        num_sites=int(args.L),
        num_particles=num_particles_main,
        indexing=str(args.ordering),
    )
    # Full-spectrum eigenvectors — pick the one closest to sector GS energy
    # that lives in the correct particle-number sector (for initial-state fallback).
    evals_full, evecs_full = np.linalg.eigh(hmat)
    psi_exact_ground = None
    for idx in range(len(evals_full)):
        if abs(evals_full[idx] - gs_energy_exact) < 1e-8:
            psi_exact_ground = _normalize_state(
                np.asarray(evecs_full[:, idx], dtype=complex).reshape(-1)
            )
            break
    if psi_exact_ground is None:
        gs_idx_fallback = int(np.argmin(evals_full))
        psi_exact_ground = _normalize_state(
            np.asarray(evecs_full[:, gs_idx_fallback], dtype=complex).reshape(-1)
        )

    # 2) Run ADAPT-VQE
    adapt_payload: dict[str, Any]
    try:
        adapt_payload, psi_adapt = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=int(args.L),
            ordering=str(args.ordering),
            problem=str(args.problem),
            adapt_pool=str(args.adapt_pool),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            boundary=str(args.boundary),
            omega0=float(args.omega0),
            g_ep=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            max_depth=int(args.adapt_max_depth),
            eps_grad=float(args.adapt_eps_grad),
            eps_energy=float(args.adapt_eps_energy),
            maxiter=int(args.adapt_maxiter),
            seed=int(args.adapt_seed),
            allow_repeats=bool(args.adapt_allow_repeats),
            finite_angle_fallback=bool(args.adapt_finite_angle_fallback),
            finite_angle=float(args.adapt_finite_angle),
            finite_angle_min_improvement=float(args.adapt_finite_angle_min_improvement),
        )
    except Exception as exc:
        _ai_log("hardcoded_adapt_vqe_failed", L=int(args.L), error=str(exc))
        adapt_payload = {
            "success": False,
            "method": f"hardcoded_adapt_vqe_{str(args.adapt_pool).lower()}",
            "energy": None,
            "error": str(exc),
        }
        psi_adapt = psi_exact_ground

    # 3) Select initial state for dynamics
    num_particles = _half_filled_particles(int(args.L))
    if problem_key == "hh":
        psi_hf = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(args.L),
                    num_particles=num_particles,
                    n_ph_max=int(args.n_ph_max),
                    boson_encoding=str(args.boson_encoding),
                    indexing=str(args.ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi_hf = _normalize_state(
            np.asarray(
                hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
                dtype=complex,
            ).reshape(-1)
        )

    if args.initial_state_source == "adapt_vqe" and bool(adapt_payload.get("success", False)):
        psi0 = psi_adapt
        _ai_log("hardcoded_adapt_initial_state_selected", source="adapt_vqe")
    elif args.initial_state_source == "adapt_vqe":
        raise RuntimeError("Requested --initial-state-source adapt_vqe but ADAPT-VQE failed.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        _ai_log("hardcoded_adapt_initial_state_selected", source="hf")
    else:
        psi0 = psi_exact_ground
        _ai_log("hardcoded_adapt_initial_state_selected", source="exact")

    # 4) Trajectory
    trajectory, _exact_states = _simulate_trajectory(
        num_sites=int(args.L),
        psi0=psi0,
        hmat=hmat,
        ordered_labels_exyz=ordered_labels_exyz,
        coeff_map_exyz=coeff_map_exyz,
        trotter_steps=int(args.trotter_steps),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        suzuki_order=int(args.suzuki_order),
    )

    # 5) Emit JSON
    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded_adapt",
        "settings": {
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "problem": str(args.problem),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "term_order": str(args.term_order),
            "adapt_pool": str(args.adapt_pool),
            "adapt_finite_angle_fallback": bool(args.adapt_finite_angle_fallback),
            "adapt_finite_angle": float(args.adapt_finite_angle),
            "adapt_finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
            "initial_state_source": str(args.initial_state_source),
        },
        "hamiltonian": {
            "num_qubits": int(
                len(ordered_labels_exyz[0])
                if ordered_labels_exyz
                else int(round(math.log2(hmat.shape[0])))
            ),
            "num_terms": int(len(coeff_map_exyz)),
            "coefficients_exyz": [
                {
                    "label_exyz": lbl,
                    "coeff": {"re": float(np.real(coeff_map_exyz[lbl])), "im": float(np.imag(coeff_map_exyz[lbl]))},
                }
                for lbl in ordered_labels_exyz
            ],
        },
        "ground_state": {
            "exact_energy": float(gs_energy_exact),
            "method": EXACT_METHOD,
        },
        "adapt_vqe": adapt_payload,
        "initial_state": {
            "source": str(args.initial_state_source if args.initial_state_source != "adapt_vqe" or adapt_payload.get("success") else "exact"),
            "amplitudes_qn_to_q0": _state_to_amplitudes_qn_to_q0(psi0),
        },
        "trajectory": trajectory,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.skip_pdf:
        _write_pipeline_pdf(output_pdf, payload, run_command)

    _ai_log(
        "hardcoded_adapt_main_done",
        L=int(args.L),
        output_json=str(output_json),
        output_pdf=(str(output_pdf) if not args.skip_pdf else None),
        adapt_energy=adapt_payload.get("energy"),
    )
    print(f"Wrote JSON: {output_json}")
    if not args.skip_pdf:
        print(f"Wrote PDF:  {output_pdf}")


if __name__ == "__main__":
    main()
