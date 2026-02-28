#!/usr/bin/env python3
"""Qiskit ADAPT-VQE end-to-end Hubbard pipeline.

Flow:
1) Build Hubbard Hamiltonian with Qiskit Nature + JW mapping.
2) Run Qiskit AdaptVQE (wrapping VQE with UCCSD excitation pool).
3) Compute sector-filtered exact energy for reference.
4) Run Suzuki-2 Trotter dynamics + exact dynamics from the ADAPT ground state.
5) Emit JSON + compact PDF artifact.

This is the Qiskit counterpart to hardcoded_adapt_pipeline.py.
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

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit_algorithms.optimizers import COBYLA as QiskitCOBYLA
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hartree_fock_reference_state import hartree_fock_statevector

EXACT_LABEL = "Exact_Qiskit"
EXACT_METHOD = "python_matrix_eigendecomposition"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


# ---------------------------------------------------------------------------
# Utility helpers (mirror qiskit VQE baseline pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


def _to_exyz(label_ixyz: str) -> str:
    return str(label_ixyz).lower().replace("i", "e")


def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "I").upper()


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((num_sites + 1) // 2, num_sites // 2)


def _interleaved_to_blocked_permutation(n_sites: int) -> list[int]:
    return [idx for site in range(n_sites) for idx in (site, n_sites + site)]


def _spin_orbital_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    normalized_ordering = ordering.strip().lower()
    if normalized_ordering == "blocked":
        return list(range(num_sites)), list(range(num_sites, 2 * num_sites))
    if normalized_ordering == "interleaved":
        return list(range(0, 2 * num_sites, 2)), list(range(1, 2 * num_sites, 2))
    raise ValueError(f"Unsupported ordering '{ordering}'.")


def _number_operator_qop(num_qubits: int, indices: list[int]) -> SparsePauliOp:
    coeffs: dict[str, complex] = {}
    id_label = "I" * num_qubits
    coeffs[id_label] = coeffs.get(id_label, 0.0 + 0.0j) + complex(0.5 * len(indices))
    for q in indices:
        chars = ["I"] * num_qubits
        chars[num_qubits - 1 - q] = "Z"
        lbl = "".join(chars)
        coeffs[lbl] = coeffs.get(lbl, 0.0 + 0.0j) + complex(-0.5)
    return SparsePauliOp.from_list([(k, v) for k, v in coeffs.items()]).simplify(atol=1e-12)


def _filtered_exact_energy(
    qop: SparsePauliOp,
    *,
    num_sites: int,
    ordering: str,
    num_particles: tuple[int, int],
) -> float:
    from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

    alpha_idx, beta_idx = _spin_orbital_index_sets(int(num_sites), ordering)
    n_alpha_op = _number_operator_qop(qop.num_qubits, alpha_idx)
    n_beta_op = _number_operator_qop(qop.num_qubits, beta_idx)

    def _filter(_state, _energy, aux_values):
        n_alpha = float(np.real(aux_values["N_alpha"][0]))
        n_beta = float(np.real(aux_values["N_beta"][0]))
        return np.isclose(n_alpha, num_particles[0]) and np.isclose(n_beta, num_particles[1])

    solver = NumPyMinimumEigensolver(filter_criterion=_filter)
    res = solver.compute_minimum_eigenvalue(
        qop,
        aux_operators={"N_alpha": n_alpha_op, "N_beta": n_beta_op},
    )
    return float(np.real(res.eigenvalue))


def _uniform_potential_qubit_op(num_sites: int, dv: float) -> SparsePauliOp:
    nq = 2 * int(num_sites)
    coeffs: dict[str, complex] = {}
    id_label = "I" * nq
    coeffs[id_label] = coeffs.get(id_label, 0.0 + 0.0j) + complex(-0.5 * dv * nq)
    for p in range(nq):
        chars = ["I"] * nq
        chars[nq - 1 - p] = "Z"
        z_label = "".join(chars)
        coeffs[z_label] = coeffs.get(z_label, 0.0 + 0.0j) + complex(0.5 * dv)
    return SparsePauliOp.from_list([(lbl, coeff) for lbl, coeff in coeffs.items()]).simplify(atol=1e-12)


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

def _build_qiskit_qubit_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
) -> tuple[SparsePauliOp, Any]:
    boundary_enum = BoundaryCondition.PERIODIC if boundary.strip().lower() == "periodic" else BoundaryCondition.OPEN
    lattice = LineLattice(
        num_nodes=int(num_sites),
        edge_parameter=-float(t),
        onsite_parameter=0.0,
        boundary_condition=boundary_enum,
    )
    ferm_op = FermiHubbardModel(lattice=lattice, onsite_interaction=float(u)).second_q_op()

    if ordering.strip().lower() == "blocked":
        ferm_op = ferm_op.permute_indices(_interleaved_to_blocked_permutation(int(num_sites)))

    mapper = JordanWignerMapper()
    qop = mapper.map(ferm_op).simplify(atol=1e-12)

    if abs(float(dv)) > 1e-15:
        qop = (qop + _uniform_potential_qubit_op(int(num_sites), float(dv))).simplify(atol=1e-12)

    return qop, mapper


def _qiskit_terms_exyz(qop: SparsePauliOp, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    order: list[str] = []
    coeff_map: dict[str, complex] = {}
    for label_ixyz, coeff in qop.to_list():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= tol:
            continue
        lbl = _to_exyz(str(label_ixyz))
        if lbl not in coeff_map:
            order.append(lbl)
            coeff_map[lbl] = 0.0 + 0.0j
        coeff_map[lbl] += coeff_c
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _ordered_qop_from_exyz(
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    *,
    tol: float = 1e-12,
) -> SparsePauliOp:
    terms: list[tuple[str, complex]] = []
    for lbl in ordered_labels_exyz:
        coeff = complex(coeff_map_exyz[lbl])
        if abs(coeff) <= tol:
            continue
        terms.append((_to_ixyz(lbl), coeff))
    if not terms:
        nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else 1
        terms = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(terms)


# ---------------------------------------------------------------------------
# Compiled Pauli + Trotter helpers (match Qiskit VQE baseline)
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


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


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
# Qiskit ADAPT-VQE runner
# ============================================================================

def _run_qiskit_adapt_vqe(
    *,
    num_sites: int,
    qop: SparsePauliOp,
    mapper: Any,
    ordering: str,
    max_iterations: int,
    gradient_threshold: float,
    cobyla_maxiter: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Run Qiskit AdaptVQE and return (payload, psi_adapt)."""
    t0 = time.perf_counter()
    _ai_log(
        "qiskit_adapt_vqe_start",
        L=int(num_sites),
        max_iterations=int(max_iterations),
        gradient_threshold=float(gradient_threshold),
        cobyla_maxiter=int(cobyla_maxiter),
        seed=int(seed),
    )

    def _finish(payload: dict[str, Any], psi: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray | None]:
        _ai_log(
            "qiskit_adapt_vqe_done",
            L=int(num_sites),
            method=str(payload.get("method", "")),
            success=bool(payload.get("success", False)),
            energy=payload.get("energy"),
            elapsed_sec=round(time.perf_counter() - t0, 6),
        )
        return payload, psi

    num_particles = _half_filled_particles(int(num_sites))

    # Build estimator
    estimator = None
    try:
        from qiskit.primitives import StatevectorEstimator
        estimator = StatevectorEstimator()
    except Exception:
        estimator = None

    if estimator is None:
        # Fallback: use NumPy eigensolver
        from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop, num_sites=int(num_sites), ordering=str(ordering), num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None
        return _finish(
            {
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy": float(np.real(eig.eigenvalue)),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
            },
            None,
        )

    try:
        hf = HartreeFock(
            num_spatial_orbitals=int(num_sites),
            num_particles=tuple(num_particles),
            qubit_mapper=mapper,
        )

        # UCCSD operator pool — individual excitations used by AdaptVQE
        ansatz = UCCSD(
            num_spatial_orbitals=int(num_sites),
            num_particles=tuple(num_particles),
            qubit_mapper=mapper,
            initial_state=hf,
            reps=1,
        )

        # Inner VQE with COBYLA (matches hardcoded pipeline's inner optimizer)
        rng = np.random.default_rng(int(seed))
        initial_point = 0.3 * rng.normal(size=ansatz.num_parameters)

        optimizer = QiskitCOBYLA(maxiter=int(cobyla_maxiter))
        inner_vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
        )

        # Wrap in AdaptVQE
        adapt_vqe = AdaptVQE(
            solver=inner_vqe,
            gradient_threshold=float(gradient_threshold),
            max_iterations=int(max_iterations),
        )

        result = adapt_vqe.compute_minimum_eigenvalue(qop)
        energy = float(np.real(result.eigenvalue))

        # Extract statevector from the final optimised circuit
        psi_adapt = None
        try:
            if hasattr(result, "optimal_circuit") and result.optimal_circuit is not None:
                if hasattr(result, "optimal_parameters") and result.optimal_parameters is not None:
                    bound = result.optimal_circuit.assign_parameters(result.optimal_parameters)
                else:
                    bound = result.optimal_circuit
                psi_adapt = _normalize_state(np.asarray(Statevector(bound).data, dtype=complex))
            elif hasattr(result, "optimal_point") and result.optimal_point is not None:
                bound = ansatz.assign_parameters(result.optimal_point)
                psi_adapt = _normalize_state(np.asarray(Statevector(bound).data, dtype=complex))
        except Exception as psi_exc:
            _ai_log("qiskit_adapt_psi_extraction_failed", error=str(psi_exc))
            psi_adapt = None

        # Sector-filtered exact energy for reference
        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop, num_sites=int(num_sites), ordering=str(ordering), num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None

        # Collect ADAPT-specific information
        num_iterations = int(getattr(result, "num_iterations", 0))
        # Try to determine ansatz depth from result
        ansatz_depth = num_iterations

        optimal_point_list = []
        if hasattr(result, "optimal_point") and result.optimal_point is not None:
            optimal_point_list = [float(x) for x in np.asarray(result.optimal_point).tolist()]

        return _finish(
            {
                "success": True,
                "method": "qiskit_adapt_vqe_uccsd",
                "energy": float(energy),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
                "num_iterations": num_iterations,
                "ansatz_depth": ansatz_depth,
                "num_parameters": int(len(optimal_point_list)),
                "optimal_point": optimal_point_list,
                "gradient_threshold": float(gradient_threshold),
                "max_iterations": int(max_iterations),
                "cobyla_maxiter": int(cobyla_maxiter),
                "pool_type": "UCCSD",
            },
            psi_adapt,
        )

    except Exception as exc:
        from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
        _ai_log("qiskit_adapt_vqe_exception", error=str(exc))
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop, num_sites=int(num_sites), ordering=str(ordering), num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None
        return _finish(
            {
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy": float(np.real(eig.eigenvalue)),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
                "warning": str(exc),
            },
            None,
        )


# ---------------------------------------------------------------------------
# Trajectory simulation (Qiskit PauliEvolutionGate + exact statevector)
# ---------------------------------------------------------------------------

def _simulate_trajectory(
    *,
    num_sites: int,
    psi0: np.ndarray,
    hmat: np.ndarray,
    trotter_hamiltonian_qop: SparsePauliOp,
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = int(trotter_hamiltonian_qop.num_qubits)
    if nq != 2 * int(num_sites):
        raise ValueError("Qubit-count mismatch between qiskit trotter operator and lattice size.")
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    synthesis = SuzukiTrotter(order=int(suzuki_order), reps=int(trotter_steps), preserve_order=True)
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    _ai_log(
        "qiskit_adapt_trajectory_start",
        L=int(num_sites),
        num_times=n_times,
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
        suzuki_order=int(suzuki_order),
    )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        t = float(time_val)
        psi_exact = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)

        if abs(t) <= 1e-15:
            psi_trot = np.array(psi0, copy=True)
        else:
            evo_gate = PauliEvolutionGate(
                trotter_hamiltonian_qop,
                time=t,
                synthesis=synthesis,
            )
            evo_circuit = synthesis.synthesize(evo_gate)
            psi_trot = np.asarray(
                Statevector(np.asarray(psi0, dtype=complex)).evolve(evo_circuit).data,
                dtype=complex,
            )
            psi_trot = _normalize_state(psi_trot)

        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)
        n_up_trot, n_dn_trot = _occupation_site0(psi_trot, num_sites)

        rows.append({
            "time": t,
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
                "qiskit_adapt_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=float(t),
                fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log(
        "qiskit_adapt_trajectory_done",
        total_steps=n_times,
        elapsed_sec=round(time.perf_counter() - t0, 6),
        final_fidelity=float(rows[-1]["fidelity"]) if rows else None,
        final_energy_trotter=float(rows[-1]["energy_trotter"]) if rows else None,
    )

    return rows, exact_states


# ---------------------------------------------------------------------------
# PDF writer
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
            "Script: pipelines/qiskit_adapt_pipeline.py",
            "",
            run_command,
        ], fontsize=10, line_spacing=0.03, max_line_width=110)

        settings = payload.get("settings", {})
        adapt = payload.get("adapt_vqe", {})

        # Settings + ADAPT summary page
        lines = [
            "Qiskit ADAPT-VQE Pipeline Summary",
            "",
            f"L={settings.get('L')}  t={settings.get('t')}  u={settings.get('u')}  dv={settings.get('dv')}",
            f"boundary={settings.get('boundary')}  ordering={settings.get('ordering')}",
            f"initial_state_source={settings.get('initial_state_source')}",
            "",
            f"ADAPT-VQE energy:       {adapt.get('energy')}",
            f"Exact filtered energy:  {adapt.get('exact_filtered_energy')}",
            f"Method:                 {adapt.get('method')}",
            f"Num iterations:         {adapt.get('num_iterations')}",
            f"Ansatz depth:           {adapt.get('ansatz_depth')}",
            f"Pool type:              {adapt.get('pool_type')}",
            f"Gradient threshold:     {adapt.get('gradient_threshold')}",
            f"COBYLA maxiter:         {adapt.get('cobyla_maxiter')}",
        ]
        _render_text_page(pdf, lines)

        # Trajectory plots
        rows = payload.get("trajectory", [])
        if rows:
            times_arr = np.array([r["time"] for r in rows])
            markevery = max(1, times_arr.size // 25)
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
            ax_f, ax_e = axes[0]
            ax_n, ax_d = axes[1]

            ax_f.plot(times_arr, [r["fidelity"] for r in rows], color="#0b3d91", marker="o", markersize=3, markevery=markevery)
            ax_f.set_title(f"Fidelity(t) = |<{EXACT_LABEL}|Trotter>|²")
            ax_f.set_ylabel("F(t)")
            ax_f.grid(alpha=0.25)

            ax_e.plot(times_arr, [r["energy_exact"] for r in rows], label=EXACT_LABEL, color="#111111", linewidth=2.0, marker="s", markersize=3, markevery=markevery)
            ax_e.plot(times_arr, [r["energy_trotter"] for r in rows], label="Trotter", color="#d62728", linestyle="--", linewidth=1.4, marker="^", markersize=3, markevery=markevery)
            ax_e.set_title("Energy")
            ax_e.set_ylabel("E(t)")
            ax_e.legend(fontsize=8)
            ax_e.grid(alpha=0.25)

            ax_n.plot(times_arr, [r["n_up_site0_exact"] for r in rows], label=f"n_up0 {EXACT_LABEL}", color="#17becf", linewidth=1.8)
            ax_n.plot(times_arr, [r["n_up_site0_trotter"] for r in rows], label="n_up0 trotter", color="#0f7f8b", linestyle="--", linewidth=1.2)
            ax_n.plot(times_arr, [r["n_dn_site0_exact"] for r in rows], label=f"n_dn0 {EXACT_LABEL}", color="#9467bd", linewidth=1.8)
            ax_n.plot(times_arr, [r["n_dn_site0_trotter"] for r in rows], label="n_dn0 trotter", color="#6f4d8f", linestyle="--", linewidth=1.2)
            ax_n.set_title("Site-0 Occupations")
            ax_n.set_xlabel("Time")
            ax_n.legend(fontsize=8)
            ax_n.grid(alpha=0.25)

            ax_d.plot(times_arr, [r["doublon_exact"] for r in rows], label=f"doublon {EXACT_LABEL}", color="#8c564b", linewidth=1.8)
            ax_d.plot(times_arr, [r["doublon_trotter"] for r in rows], label="doublon trotter", color="#c251a1", linestyle="--", linewidth=1.2)
            ax_d.set_title("Total Doublon")
            ax_d.set_xlabel("Time")
            ax_d.legend(fontsize=8)
            ax_d.grid(alpha=0.25)

            fig.suptitle(f"Qiskit ADAPT-VQE Pipeline L={settings.get('L')}", fontsize=13)
            fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

        # VQE energy bar chart
        gs_exact = float(payload["ground_state"]["exact_energy"])
        adapt_e = adapt.get("energy")
        adapt_val = float(adapt_e) if adapt_e is not None else np.nan

        figv, axesv = plt.subplots(1, 2, figsize=(11.0, 5.0))
        vx0, vx1 = axesv[0], axesv[1]
        vx0.bar([0, 1], [gs_exact, adapt_val], color=["#111111", "#ff7f0e"], edgecolor="black", linewidth=0.4)
        vx0.set_xticks([0, 1])
        vx0.set_xticklabels(["Exact GS", "ADAPT-VQE"])
        vx0.set_ylabel("Energy")
        vx0.set_title("ADAPT-VQE Energy vs Exact GS")
        vx0.grid(axis="y", alpha=0.25)

        err = abs(adapt_val - gs_exact) if np.isfinite(adapt_val) else np.nan
        vx1.bar([0], [err], color="#ff7f0e", edgecolor="black", linewidth=0.4)
        vx1.set_xticks([0])
        vx1.set_xticklabels(["|ADAPT-Exact|"])
        vx1.set_ylabel("Absolute Error")
        vx1.set_title("ADAPT-VQE Absolute Error")
        vx1.grid(axis="y", alpha=0.25)

        figv.suptitle(
            "When initial_state_source=adapt_vqe, Trotter E(t=0) = ⟨ψ_adapt|H|ψ_adapt⟩ = ADAPT energy.\n"
            "ADAPT energy ≠ exact GS energy unless fully converged.",
            fontsize=10,
        )
        figv.tight_layout(rect=(0.0, 0.03, 1.0, 0.91))
        pdf.savefig(figv)
        plt.close(figv)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qiskit ADAPT-VQE Hubbard pipeline.")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--term-order", choices=["qiskit", "sorted"], default="sorted")

    # ADAPT-VQE controls
    p.add_argument("--adapt-max-iterations", type=int, default=20, help="Max ADAPT iterations (operators to add)")
    p.add_argument("--adapt-gradient-threshold", type=float, default=1e-4, help="Gradient convergence threshold")
    p.add_argument("--adapt-cobyla-maxiter", type=int, default=300, help="COBYLA maxiter per inner VQE re-optimization")
    p.add_argument("--adapt-seed", type=int, default=7)

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
    _ai_log("qiskit_adapt_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (artifacts_dir / f"qiskit_adapt_pipeline_L{args.L}.json")
    output_pdf = args.output_pdf or (artifacts_dir / f"qiskit_adapt_pipeline_L{args.L}.pdf")

    # 1) Build Hamiltonian
    qop, mapper = _build_qiskit_qubit_hamiltonian(
        num_sites=int(args.L),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        boundary=str(args.boundary),
        ordering=str(args.ordering),
    )

    native_order, coeff_map_exyz = _qiskit_terms_exyz(qop)
    _ai_log("qiskit_adapt_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "qiskit":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)
    trotter_qop_ordered = _ordered_qop_from_exyz(ordered_labels_exyz, coeff_map_exyz)

    hmat = np.asarray(qop.to_matrix(sparse=False), dtype=complex)
    nq = int(2 * args.L)

    # Sector-filtered exact ground state: ADAPT-VQE preserves particle number,
    # so compare against the GS within the same (n_alpha, n_beta) sector.
    num_particles = _half_filled_particles(int(args.L))
    alpha_idx, beta_idx = _spin_orbital_index_sets(int(args.L), ordering=str(args.ordering))
    n_alpha, n_beta = int(num_particles[0]), int(num_particles[1])
    sector_basis: list[int] = []
    for idx in range(1 << nq):
        na = sum((idx >> q) & 1 for q in alpha_idx)
        nb = sum((idx >> q) & 1 for q in beta_idx)
        if na == n_alpha and nb == n_beta:
            sector_basis.append(idx)
    h_sector = hmat[np.ix_(sector_basis, sector_basis)]
    evals_sector, evecs_sector = np.linalg.eigh(h_sector)
    gs_idx_sector = int(np.argmin(evals_sector))
    gs_energy_exact = float(np.real(evals_sector[gs_idx_sector]))
    # Embed sector eigenvector back into the full Hilbert space
    psi_exact_ground = np.zeros(1 << nq, dtype=complex)
    for k, basis_idx in enumerate(sector_basis):
        psi_exact_ground[basis_idx] = evecs_sector[k, gs_idx_sector]
    psi_exact_ground = _normalize_state(psi_exact_ground)

    # 2) Run ADAPT-VQE
    adapt_payload, psi_adapt = _run_qiskit_adapt_vqe(
        num_sites=int(args.L),
        qop=qop,
        mapper=mapper,
        ordering=str(args.ordering),
        max_iterations=int(args.adapt_max_iterations),
        gradient_threshold=float(args.adapt_gradient_threshold),
        cobyla_maxiter=int(args.adapt_cobyla_maxiter),
        seed=int(args.adapt_seed),
    )

    # 3) Select initial state for dynamics
    num_particles = _half_filled_particles(int(args.L))
    psi_hf = _normalize_state(
        np.asarray(
            hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
            dtype=complex,
        ).reshape(-1)
    )

    if args.initial_state_source == "adapt_vqe" and psi_adapt is not None:
        psi0 = psi_adapt
        init_source = "adapt_vqe"
        _ai_log("qiskit_adapt_initial_state_selected", source="adapt_vqe")
    elif args.initial_state_source == "adapt_vqe":
        raise RuntimeError("Requested --initial-state-source adapt_vqe but Qiskit ADAPT-VQE statevector is unavailable.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        init_source = "hf"
        _ai_log("qiskit_adapt_initial_state_selected", source="hf")
    else:
        psi0 = psi_exact_ground
        init_source = "exact"
        _ai_log("qiskit_adapt_initial_state_selected", source="exact")

    # 4) Trajectory
    trajectory, _exact_states = _simulate_trajectory(
        num_sites=int(args.L),
        psi0=psi0,
        hmat=hmat,
        trotter_hamiltonian_qop=trotter_qop_ordered,
        trotter_steps=int(args.trotter_steps),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        suzuki_order=int(args.suzuki_order),
    )

    # 5) Emit JSON
    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "qiskit_adapt",
        "settings": {
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "term_order": str(args.term_order),
            "initial_state_source": str(init_source),
        },
        "hamiltonian": {
            "num_qubits": int(2 * args.L),
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
            "source": str(init_source),
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
        "qiskit_adapt_main_done",
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
