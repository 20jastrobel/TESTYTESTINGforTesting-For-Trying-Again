#!/usr/bin/env python3
"""
Phase-1 L=2 hardcoded-vs-frozen-Qiskit matcher.

Scope is intentionally narrow:
- L=2 only
- locked settings from frozen baseline JSON
- hardcoded VQE + hardcoded Suzuki-Trotter dynamics
- comparison metrics/reports against the frozen baseline
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian


BASELINE_PATH = ROOT / "Tests" / "phase1_qiskit_builtin_long_highfid_results_run1.json"

ITER0_METRICS_PATH = ROOT / "hardcoded_vs_qiskit_l2_match_metrics_iter0.json"
ITER0_REPORT_PDF_PATH = ROOT / "hardcoded_vs_qiskit_l2_match_report_iter0.pdf"

FINAL_METRICS_PATH = ROOT / "hardcoded_vs_qiskit_l2_match_final_metrics.json"
FINAL_REPORT_PDF_PATH = ROOT / "hardcoded_vs_qiskit_l2_match_final_report.pdf"

AUDIT_PATH = ROOT / "trotter_term_audit_l2.json"

MAX_ITERS = 5


THRESHOLDS = {
    "ground_state_energy_abs_delta": 1e-8,
    "fidelity_max_abs_delta": 1e-4,
    "energy_trotter_max_abs_delta": 1e-3,
    "n_up_site0_trotter_max_abs_delta": 5e-3,
    "n_dn_site0_trotter_max_abs_delta": 5e-3,
    "doublon_trotter_max_abs_delta": 1e-3,
}

TARGET_FIELDS = [
    "fidelity",
    "energy_trotter",
    "n_up_site0_trotter",
    "n_dn_site0_trotter",
    "doublon_trotter",
]

PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


@dataclass
class LockedSettings:
    num_sites: int
    hopping_t: float
    onsite_u: float
    dv: float
    boundary: str
    ordering: str
    t_final: float
    num_times: int
    suzuki_order: int
    trotter_steps: int
    initial_state: str
    initial_state_label_qn_to_q0: str | None
    initial_state_amplitudes_qn_to_q0: dict[str, Any]


def _complex_from_json(val: Any) -> complex:
    if isinstance(val, dict):
        return complex(float(val.get("re", 0.0)), float(val.get("im", 0.0)))
    return complex(float(val), 0.0)


def _to_ixyz(label_exyz: str) -> str:
    return label_exyz.replace("e", "I").upper()


def _to_exyz(label_ixyz: str) -> str:
    return label_ixyz.lower().replace("i", "e")


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _state_from_amplitudes_qn_to_q0(amps: dict[str, Any], nq: int) -> np.ndarray:
    psi = np.zeros(2**nq, dtype=complex)
    for bitstring, amp in amps.items():
        if len(bitstring) != nq:
            raise ValueError(f"Amplitude key '{bitstring}' length != nq={nq}")
        idx = int(bitstring, 2)
        psi[idx] = _complex_from_json(amp)
    return _normalize_state(psi)


def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
    nq = int(round(math.log2(psi.size)))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) < cutoff:
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _load_hardcoded_vqe_namespace() -> dict[str, Any]:
    nb_path = ROOT / "pydephasing" / "quantum" / "vqe_latex_python_pairs_test.ipynb"
    payload = json.loads(nb_path.read_text())
    ns: dict[str, Any] = {}
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "Final benchmark: hardcoded VQE vs Qiskit VQE" in src:
            continue
        exec(src, ns)

    required = [
        "half_filled_num_particles",
        "hartree_fock_bitstring",
        "basis_state",
        "HardcodedUCCSDAnsatz",
        "vqe_minimize",
        "expval_pauli_polynomial",
    ]
    missing = [k for k in required if k not in ns]
    if missing:
        raise RuntimeError(f"Missing notebook symbols: {missing}")
    return ns


def _collect_hardcoded_terms(H_poly: Any, tol: float = 1e-12) -> list[tuple[str, complex]]:
    coeffs: dict[str, complex] = {}
    order: list[str] = []
    for term in H_poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if label not in coeffs:
            order.append(label)
            coeffs[label] = 0.0 + 0.0j
        coeffs[label] += coeff
    out: list[tuple[str, complex]] = []
    for label in order:
        coeff = coeffs[label]
        if abs(coeff) > tol:
            out.append((label, coeff))
    return out


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _build_hamiltonian_matrix(terms_exyz: list[tuple[str, complex]]) -> np.ndarray:
    nq = len(terms_exyz[0][0]) if terms_exyz else 1
    dim = 2**nq
    H = np.zeros((dim, dim), dtype=complex)
    for label, coeff in terms_exyz:
        H += coeff * _pauli_matrix_exyz(label)
    return H


def _expectation_hamiltonian(psi: np.ndarray, H: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, H @ psi)))


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    va = _normalize_state(np.asarray(a, dtype=complex).reshape(-1))
    vb = _normalize_state(np.asarray(b, dtype=complex).reshape(-1))
    return float(abs(np.vdot(va, vb)) ** 2)


def _state_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=complex).reshape(-1)
    vb = np.asarray(b, dtype=complex).reshape(-1)
    if va.shape != vb.shape:
        raise ValueError("State shape mismatch in L2 error.")
    return float(np.linalg.norm(va - vb))


def _occupation_from_state(psi: np.ndarray, qubit: int, nq: int) -> float:
    probs = np.abs(psi) ** 2
    out = 0.0
    for idx, p in enumerate(probs):
        bit = (idx >> qubit) & 1
        out += float(bit) * float(p)
    return float(out)


def _doublon_total_from_state(psi: np.ndarray, num_sites: int) -> float:
    """Total doublon: sum_i n_{i,up} n_{i,dn} in blocked ordering."""
    probs = np.abs(psi) ** 2
    out = 0.0
    for idx, p in enumerate(probs):
        doublon_count = 0
        for site in range(num_sites):
            up = (idx >> site) & 1
            dn = (idx >> (num_sites + site)) & 1
            doublon_count += int(up * dn)
        out += float(doublon_count) * float(p)
    return float(out)


def _apply_exp_term(psi: np.ndarray, term_matrix: np.ndarray, coeff: complex, alpha: float, tol: float = 1e-12) -> np.ndarray:
    if abs(coeff.imag) > tol:
        raise ValueError(f"Imaginary coefficient not supported in hardcoded trotter term: {coeff}")
    theta = float(alpha) * float(coeff.real)
    c = math.cos(theta)
    s = math.sin(theta)
    return c * psi - 1j * s * (term_matrix @ psi)


def _qiskit_ordered_terms_ixyz(settings: LockedSettings) -> list[tuple[str, complex]]:
    # Validation-only import path (kept out of package core).
    from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
    from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    def interleaved_to_blocked_permutation(n_sites: int) -> list[int]:
        return [idx for site in range(n_sites) for idx in (site, n_sites + site)]

    def apply_ordering(op, n_sites: int, ordering: str):
        norm = ordering.strip().lower()
        if norm == "interleaved":
            return op
        if norm == "blocked":
            return op.permute_indices(interleaved_to_blocked_permutation(n_sites))
        raise ValueError(f"Unsupported ordering: {ordering}")

    boundary = BoundaryCondition.PERIODIC if settings.boundary.strip().lower() == "periodic" else BoundaryCondition.OPEN
    lattice = LineLattice(
        num_nodes=settings.num_sites,
        edge_parameter=-settings.hopping_t,
        onsite_parameter=0.0,
        boundary_condition=boundary,
    )
    ferm_op = FermiHubbardModel(lattice=lattice, onsite_interaction=settings.onsite_u).second_q_op()
    ferm_op = apply_ordering(ferm_op, settings.num_sites, settings.ordering)
    qop = JordanWignerMapper().map(ferm_op).simplify(atol=1e-12)
    return [(str(label), complex(coeff)) for label, coeff in qop.to_list()]


def _term_sequence_for_mode(
    hardcoded_terms_exyz: list[tuple[str, complex]],
    qiskit_ordered_terms_ixyz: list[tuple[str, complex]],
    mode: str,
) -> tuple[list[tuple[str, complex]], dict[str, Any]]:
    coeff_map_hc = {lbl: complex(c) for lbl, c in hardcoded_terms_exyz}
    coeff_map_qk = {lbl: complex(c) for lbl, c in qiskit_ordered_terms_ixyz}

    diag: dict[str, Any] = {}

    if mode == "legacy_sorted_hardcoded":
        seq = sorted(hardcoded_terms_exyz, key=lambda x: x[0])
        diag["term_order_mode"] = mode
        diag["num_terms"] = len(seq)
        return seq, diag

    if mode == "baseline_qiskit_order_hardcoded_coeffs":
        seq: list[tuple[str, complex]] = []
        missing_from_hc: list[str] = []
        for label_ixyz, _qk_coeff in qiskit_ordered_terms_ixyz:
            label_exyz = _to_exyz(label_ixyz)
            if label_exyz not in coeff_map_hc:
                missing_from_hc.append(label_exyz)
                continue
            seq.append((label_exyz, coeff_map_hc[label_exyz]))
        diag["term_order_mode"] = mode
        diag["num_terms"] = len(seq)
        diag["missing_terms_from_hardcoded_coeff_map"] = missing_from_hc
        return seq, diag

    raise ValueError(f"Unknown term order mode: {mode}")


def _simulate_l2(
    settings: LockedSettings,
    ns: dict[str, Any],
    *,
    initial_state_mode: str,
    term_order_mode: str,
    time_semantics_mode: str,
) -> dict[str, Any]:
    if settings.num_sites != 2:
        raise ValueError("This harness is restricted to L=2.")
    if settings.ordering != "blocked":
        raise ValueError("This harness currently enforces blocked ordering.")
    if settings.suzuki_order != 2:
        raise ValueError("This harness currently enforces Suzuki order 2.")

    L = settings.num_sites
    nq = 2 * L

    H_poly = build_hubbard_hamiltonian(
        dims=L,
        t=settings.hopping_t,
        U=settings.onsite_u,
        v=settings.dv,
        repr_mode="JW",
        indexing=settings.ordering,
        pbc=(settings.boundary == "periodic"),
    )
    hardcoded_terms_exyz = _collect_hardcoded_terms(H_poly)
    H_mat = _build_hamiltonian_matrix(hardcoded_terms_exyz)

    qiskit_ordered_terms_ixyz = _qiskit_ordered_terms_ixyz(settings)
    term_seq_exyz, term_seq_diag = _term_sequence_for_mode(
        hardcoded_terms_exyz,
        qiskit_ordered_terms_ixyz,
        mode=term_order_mode,
    )
    term_mats = {label: _pauli_matrix_exyz(label) for label, _ in term_seq_exyz}

    # Hardcoded VQE (always computed for ground-state-energy comparison).
    num_particles = tuple(ns["half_filled_num_particles"](L))
    hf_bitstring = str(ns["hartree_fock_bitstring"](n_sites=L, num_particles=num_particles, indexing=settings.ordering))
    psi_ref = np.asarray(ns["basis_state"](nq, hf_bitstring), dtype=complex)
    hardcoded_ansatz = ns["HardcodedUCCSDAnsatz"](
        dims=L,
        num_particles=num_particles,
        reps=2,
        repr_mode="JW",
        indexing=settings.ordering,
        include_singles=True,
        include_doubles=True,
    )
    vqe_result = ns["vqe_minimize"](
        H_poly,
        hardcoded_ansatz,
        psi_ref,
        restarts=3,
        seed=7,
        maxiter=1800,
        method="SLSQP",
    )
    psi_hardcoded_vqe = _normalize_state(
        np.asarray(hardcoded_ansatz.prepare_state(np.asarray(vqe_result.theta, dtype=float), psi_ref), dtype=complex)
    )

    if initial_state_mode == "hardcoded_vqe":
        psi0 = psi_hardcoded_vqe
    elif initial_state_mode == "baseline_amplitudes":
        psi0 = _state_from_amplitudes_qn_to_q0(settings.initial_state_amplitudes_qn_to_q0, nq=nq)
    else:
        raise ValueError(f"Unknown initial_state_mode: {initial_state_mode}")

    # Exact propagator eigendecomposition.
    evals, evecs = np.linalg.eigh(H_mat)
    evecs_dag = np.conjugate(evecs).T

    times = np.linspace(0.0, settings.t_final, settings.num_times)
    trajectory: list[dict[str, float]] = []

    trotter_prev = np.array(psi0, copy=True)
    t_prev = 0.0

    for t in times:
        t_f = float(t)
        # exact from absolute time
        coeffs_exact = np.exp(-1j * evals * t_f)
        psi_exact = evecs @ (coeffs_exact * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)

        if time_semantics_mode == "baseline_per_time_fixed_reps":
            psi_trot = np.array(psi0, copy=True)
            if t_f > 0.0:
                dt = t_f / float(settings.trotter_steps)
                for _ in range(settings.trotter_steps):
                    half = 0.5 * dt
                    for label, coeff in term_seq_exyz:
                        psi_trot = _apply_exp_term(psi_trot, term_mats[label], coeff, half)
                    for label, coeff in reversed(term_seq_exyz):
                        psi_trot = _apply_exp_term(psi_trot, term_mats[label], coeff, half)
            psi_trot = _normalize_state(psi_trot)
        elif time_semantics_mode == "legacy_incremental_fixed_reps":
            if abs(t_f) <= 1e-15:
                trotter_prev = np.array(psi0, copy=True)
                t_prev = 0.0
            else:
                dt_segment = (t_f - t_prev) / float(settings.trotter_steps)
                for _ in range(settings.trotter_steps):
                    half = 0.5 * dt_segment
                    for label, coeff in term_seq_exyz:
                        trotter_prev = _apply_exp_term(trotter_prev, term_mats[label], coeff, half)
                    for label, coeff in reversed(term_seq_exyz):
                        trotter_prev = _apply_exp_term(trotter_prev, term_mats[label], coeff, half)
                trotter_prev = _normalize_state(trotter_prev)
                t_prev = t_f
            psi_trot = np.array(trotter_prev, copy=True)
        else:
            raise ValueError(f"Unknown time_semantics_mode: {time_semantics_mode}")

        trajectory.append(
            {
                "time": t_f,
                "fidelity": _fidelity(psi_exact, psi_trot),
                "state_l2_error": _state_l2_error(psi_exact, psi_trot),
                "energy_exact": _expectation_hamiltonian(psi_exact, H_mat),
                "energy_trotter": _expectation_hamiltonian(psi_trot, H_mat),
                "n_up_site0_exact": _occupation_from_state(psi_exact, qubit=0, nq=nq),
                "n_up_site0_trotter": _occupation_from_state(psi_trot, qubit=0, nq=nq),
                "n_dn_site0_exact": _occupation_from_state(psi_exact, qubit=L, nq=nq),
                "n_dn_site0_trotter": _occupation_from_state(psi_trot, qubit=L, nq=nq),
                "doublon_exact": _doublon_total_from_state(psi_exact, num_sites=L),
                "doublon_trotter": _doublon_total_from_state(psi_trot, num_sites=L),
            }
        )

    return {
        "settings": {
            "num_sites": settings.num_sites,
            "hopping_t": settings.hopping_t,
            "onsite_u": settings.onsite_u,
            "dv": settings.dv,
            "boundary": settings.boundary,
            "ordering": settings.ordering,
            "t_final": settings.t_final,
            "num_times": settings.num_times,
            "suzuki_order": settings.suzuki_order,
            "trotter_steps": settings.trotter_steps,
            "initial_state_mode": initial_state_mode,
            "term_order_mode": term_order_mode,
            "time_semantics_mode": time_semantics_mode,
        },
        "hardcoded_vqe": {
            "energy": float(vqe_result.energy),
            "best_restart": int(vqe_result.best_restart),
            "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
            "uccsd_reps": 2,
            "uccsd_num_parameters": int(hardcoded_ansatz.num_parameters),
            "optimal_point": [float(x) for x in np.asarray(vqe_result.theta, dtype=float).tolist()],
            "hf_bitstring": hf_bitstring,
            "initial_state_amplitudes_qn_to_q0_if_hardcoded_vqe": (
                _state_to_amplitudes_qn_to_q0(psi_hardcoded_vqe) if initial_state_mode == "hardcoded_vqe" else None
            ),
        },
        "trajectory": trajectory,
        "term_order_diagnostics": term_seq_diag,
        "hardcoded_terms_exyz_ordered_native": [{"label_exyz": lbl, "coeff": {"re": float(c.real), "im": float(c.imag)}} for lbl, c in hardcoded_terms_exyz],
        "qiskit_terms_ixyz_ordered_to_list": [{"label_ixyz": lbl, "coeff": {"re": float(c.real), "im": float(c.imag)}} for lbl, c in qiskit_ordered_terms_ixyz],
    }


def _first_crossing_time(times: np.ndarray, vals: np.ndarray, thr: float) -> float | None:
    idxs = np.where(vals > thr)[0]
    if idxs.size == 0:
        return None
    return float(times[int(idxs[0])])


def _compare_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    base_traj = baseline["trajectory"]
    cand_traj = candidate["trajectory"]
    if len(base_traj) != len(cand_traj):
        raise ValueError("Trajectory length mismatch.")

    base_times = np.array([float(r["time"]) for r in base_traj], dtype=float)
    cand_times = np.array([float(r["time"]) for r in cand_traj], dtype=float)
    if not np.allclose(base_times, cand_times, atol=1e-12, rtol=0.0):
        raise ValueError("Time grid mismatch.")

    out: dict[str, Any] = {
        "time_grid": {
            "num_times": int(base_times.size),
            "t0": float(base_times[0]),
            "t_final": float(base_times[-1]),
            "dt": float(base_times[1] - base_times[0]) if base_times.size > 1 else 0.0,
        },
        "trajectory_deltas": {},
    }

    for key in TARGET_FIELDS:
        base_vals = np.array([float(r[key]) for r in base_traj], dtype=float)
        cand_vals = np.array([float(r[key]) for r in cand_traj], dtype=float)
        delta = np.abs(cand_vals - base_vals)
        out["trajectory_deltas"][key] = {
            "max_abs_delta": float(np.max(delta)),
            "mean_abs_delta": float(np.mean(delta)),
            "final_abs_delta": float(delta[-1]),
            "first_time_abs_delta_gt_1e-4": _first_crossing_time(base_times, delta, 1e-4),
            "first_time_abs_delta_gt_1e-3": _first_crossing_time(base_times, delta, 1e-3),
        }

    base_gs = float(baseline.get("vqe_uccsd", {}).get("energy"))
    cand_gs = float(candidate["hardcoded_vqe"]["energy"])
    out["ground_state_energy"] = {
        "baseline_vqe_uccsd_energy": base_gs,
        "candidate_hardcoded_vqe_energy": cand_gs,
        "abs_delta": float(abs(cand_gs - base_gs)),
    }

    out["acceptance"] = {
        "thresholds": THRESHOLDS,
        "checks": {
            "ground_state_energy_abs_delta": float(abs(cand_gs - base_gs)) <= THRESHOLDS["ground_state_energy_abs_delta"],
            "fidelity_max_abs_delta": out["trajectory_deltas"]["fidelity"]["max_abs_delta"] <= THRESHOLDS["fidelity_max_abs_delta"],
            "energy_trotter_max_abs_delta": out["trajectory_deltas"]["energy_trotter"]["max_abs_delta"]
            <= THRESHOLDS["energy_trotter_max_abs_delta"],
            "n_up_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_up_site0_trotter"]["max_abs_delta"]
            <= THRESHOLDS["n_up_site0_trotter_max_abs_delta"],
            "n_dn_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_dn_site0_trotter"]["max_abs_delta"]
            <= THRESHOLDS["n_dn_site0_trotter_max_abs_delta"],
            "doublon_trotter_max_abs_delta": out["trajectory_deltas"]["doublon_trotter"]["max_abs_delta"]
            <= THRESHOLDS["doublon_trotter_max_abs_delta"],
        },
    }
    out["acceptance"]["pass"] = bool(all(out["acceptance"]["checks"].values()))
    return out


def _write_iter_pdf(
    pdf_path: Path,
    iter_name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    base = baseline["trajectory"]
    cand = candidate["trajectory"]
    t = np.array([float(r["time"]) for r in base], dtype=float)

    def arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows], dtype=float)

    b_fid = arr(base, "fidelity")
    c_fid = arr(cand, "fidelity")
    b_e = arr(base, "energy_trotter")
    c_e = arr(cand, "energy_trotter")
    b_nu = arr(base, "n_up_site0_trotter")
    c_nu = arr(cand, "n_up_site0_trotter")
    b_nd = arr(base, "n_dn_site0_trotter")
    c_nd = arr(cand, "n_dn_site0_trotter")
    b_d = arr(base, "doublon_trotter")
    c_d = arr(cand, "doublon_trotter")

    td = metrics["trajectory_deltas"]

    with PdfPages(str(pdf_path)) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]

        ax00.plot(t, b_fid, label="baseline fidelity", color="#1f77b4", linewidth=1.8)
        ax00.plot(t, c_fid, label="hardcoded fidelity", color="#ff7f0e", linestyle="--", linewidth=1.5)
        ax00.set_title("Fidelity")
        ax00.grid(alpha=0.25)
        ax00.legend(fontsize=8)

        ax01.plot(t, b_e, label="baseline energy_trotter", color="#2ca02c", linewidth=1.8)
        ax01.plot(t, c_e, label="hardcoded energy_trotter", color="#d62728", linestyle="--", linewidth=1.5)
        ax01.set_title("Energy Trotter")
        ax01.grid(alpha=0.25)
        ax01.legend(fontsize=8)

        ax10.plot(t, b_nu, label="baseline n_up0", color="#17becf", linewidth=1.8)
        ax10.plot(t, c_nu, label="hardcoded n_up0", color="#17becf", linestyle="--", linewidth=1.4)
        ax10.plot(t, b_nd, label="baseline n_dn0", color="#9467bd", linewidth=1.8)
        ax10.plot(t, c_nd, label="hardcoded n_dn0", color="#9467bd", linestyle="--", linewidth=1.4)
        ax10.set_xlabel("Time")
        ax10.set_title("Site-0 Occupations")
        ax10.grid(alpha=0.25)
        ax10.legend(fontsize=8)

        ax11.plot(t, b_d, label="baseline doublon", color="#8c564b", linewidth=1.8)
        ax11.plot(t, c_d, label="hardcoded doublon", color="#e377c2", linestyle="--", linewidth=1.4)
        ax11.set_xlabel("Time")
        ax11.set_title("Total Doublon")
        ax11.grid(alpha=0.25)
        ax11.legend(fontsize=8)

        fig.suptitle(f"L=2 Hardcoded vs Frozen Qiskit Time Dynamics ({iter_name})", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        fig2, axes2 = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        bx00, bx01 = axes2[0, 0], axes2[0, 1]
        bx10, bx11 = axes2[1, 0], axes2[1, 1]
        bx00.plot(t, np.abs(c_fid - b_fid), color="#1f77b4")
        bx00.set_title("|delta fidelity|")
        bx00.grid(alpha=0.25)
        bx01.plot(t, np.abs(c_e - b_e), color="#d62728")
        bx01.set_title("|delta energy_trotter|")
        bx01.grid(alpha=0.25)
        bx10.plot(t, np.abs(c_nu - b_nu), label="|delta n_up0|", color="#17becf")
        bx10.plot(t, np.abs(c_nd - b_nd), label="|delta n_dn0|", color="#9467bd")
        bx10.set_title("Occupation Deltas")
        bx10.grid(alpha=0.25)
        bx10.legend(fontsize=8)
        bx10.set_xlabel("Time")
        bx11.plot(t, np.abs(c_d - b_d), color="#8c564b")
        bx11.set_title("|delta doublon|")
        bx11.grid(alpha=0.25)
        bx11.set_xlabel("Time")
        fig2.suptitle(f"L=2 Delta Diagnostics ({iter_name})", fontsize=14)
        fig2.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3 = plt.figure(figsize=(11.0, 8.5))
        ax3 = fig3.add_subplot(111)
        ax3.axis("off")
        lines = [
            f"L=2 metrics summary ({iter_name})",
            "",
            f"ground-state energy abs delta: {metrics['ground_state_energy']['abs_delta']:.12e}",
            f"fidelity max/mean/final: {td['fidelity']['max_abs_delta']:.12e} / {td['fidelity']['mean_abs_delta']:.12e} / {td['fidelity']['final_abs_delta']:.12e}",
            f"energy_trotter max/mean/final: {td['energy_trotter']['max_abs_delta']:.12e} / {td['energy_trotter']['mean_abs_delta']:.12e} / {td['energy_trotter']['final_abs_delta']:.12e}",
            f"n_up_site0_trotter max/mean/final: {td['n_up_site0_trotter']['max_abs_delta']:.12e} / {td['n_up_site0_trotter']['mean_abs_delta']:.12e} / {td['n_up_site0_trotter']['final_abs_delta']:.12e}",
            f"n_dn_site0_trotter max/mean/final: {td['n_dn_site0_trotter']['max_abs_delta']:.12e} / {td['n_dn_site0_trotter']['mean_abs_delta']:.12e} / {td['n_dn_site0_trotter']['final_abs_delta']:.12e}",
            f"doublon_trotter max/mean/final: {td['doublon_trotter']['max_abs_delta']:.12e} / {td['doublon_trotter']['mean_abs_delta']:.12e} / {td['doublon_trotter']['final_abs_delta']:.12e}",
            "",
            f"Acceptance checks: {metrics['acceptance']['checks']}",
            f"PASS: {metrics['acceptance']['pass']}",
        ]
        ax3.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        pdf.savefig(fig3)
        plt.close(fig3)


def _load_locked_settings(baseline: dict[str, Any]) -> LockedSettings:
    s = baseline.get("settings", {})
    return LockedSettings(
        num_sites=int(s.get("num_sites", 2)),
        hopping_t=float(s.get("hopping_t", 1.0)),
        onsite_u=float(s.get("onsite_u", 4.0)),
        dv=0.0,  # explicit lock from task
        boundary=str(s.get("boundary", "periodic")),
        ordering=str(s.get("spin_orbital_ordering", "blocked")),
        t_final=float(s.get("t_final", 20.0)),
        num_times=int(s.get("num_times", 401)),
        suzuki_order=int(s.get("suzuki_order", 2)),
        trotter_steps=int(s.get("trotter_steps", 128)),
        initial_state=str(s.get("initial_state", "vqe_uccsd")),
        initial_state_label_qn_to_q0=s.get("initial_state_label_qn_to_q0"),
        initial_state_amplitudes_qn_to_q0=dict(s.get("initial_state_amplitudes_qn_to_q0", {})),
    )


def _assert_locked_requirements(locked: LockedSettings) -> None:
    expected = {
        "num_sites": 2,
        "hopping_t": 1.0,
        "onsite_u": 4.0,
        "dv": 0.0,
        "boundary": "periodic",
        "ordering": "blocked",
        "t_final": 20.0,
        "num_times": 401,
        "suzuki_order": 2,
        "trotter_steps": 128,
    }
    got = {
        "num_sites": locked.num_sites,
        "hopping_t": locked.hopping_t,
        "onsite_u": locked.onsite_u,
        "dv": locked.dv,
        "boundary": locked.boundary,
        "ordering": locked.ordering,
        "t_final": locked.t_final,
        "num_times": locked.num_times,
        "suzuki_order": locked.suzuki_order,
        "trotter_steps": locked.trotter_steps,
    }
    for key, expected_value in expected.items():
        actual = got[key]
        if isinstance(expected_value, float):
            if not math.isclose(float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Locked setting mismatch for {key}: got {actual}, expected {expected_value}")
        else:
            if actual != expected_value:
                raise ValueError(f"Locked setting mismatch for {key}: got {actual}, expected {expected_value}")


def main() -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    locked = _load_locked_settings(baseline)
    _assert_locked_requirements(locked)

    print("Locked settings extracted from baseline:")
    print(json.dumps(locked.__dict__, indent=2))

    ns = _load_hardcoded_vqe_namespace()

    # Iter0: current/legacy hardcoded conventions.
    iter0_candidate = _simulate_l2(
        locked,
        ns,
        initial_state_mode="hardcoded_vqe",
        term_order_mode="legacy_sorted_hardcoded",
        time_semantics_mode="legacy_incremental_fixed_reps",
    )
    iter0_metrics = _compare_to_baseline(iter0_candidate, baseline)
    ITER0_METRICS_PATH.write_text(json.dumps(iter0_metrics, indent=2), encoding="utf-8")
    _write_iter_pdf(ITER0_REPORT_PDF_PATH, "iter0", baseline, iter0_candidate, iter0_metrics)

    # Diagnostic A: baseline amplitudes, keep legacy ordering/semantics.
    diag_a_candidate = _simulate_l2(
        locked,
        ns,
        initial_state_mode="baseline_amplitudes",
        term_order_mode="legacy_sorted_hardcoded",
        time_semantics_mode="legacy_incremental_fixed_reps",
    )
    diag_a_metrics = _compare_to_baseline(diag_a_candidate, baseline)
    collapse_flags = {
        key: (
            float(diag_a_metrics["trajectory_deltas"][key]["max_abs_delta"])
            < float(iter0_metrics["trajectory_deltas"][key]["max_abs_delta"])
        )
        for key in TARGET_FIELDS
    }

    # Diagnostic B: ordered term and coefficient audit.
    hc_native = iter0_candidate["hardcoded_terms_exyz_ordered_native"]
    qk_order = iter0_candidate["qiskit_terms_ixyz_ordered_to_list"]
    hc_map_ixyz = { _to_ixyz(item["label_exyz"]): _complex_from_json(item["coeff"]) for item in hc_native }
    qk_map_ixyz = { str(item["label_ixyz"]): _complex_from_json(item["coeff"]) for item in qk_order }
    union = sorted(set(hc_map_ixyz.keys()) | set(qk_map_ixyz.keys()))
    coeff_delta_rows = []
    max_abs_coeff_delta = 0.0
    for label in union:
        dc = hc_map_ixyz.get(label, 0.0 + 0.0j) - qk_map_ixyz.get(label, 0.0 + 0.0j)
        abs_dc = float(abs(dc))
        max_abs_coeff_delta = max(max_abs_coeff_delta, abs_dc)
        coeff_delta_rows.append(
            {
                "label_ixyz": label,
                "hardcoded_coeff": {"re": float(np.real(hc_map_ixyz.get(label, 0.0))), "im": float(np.imag(hc_map_ixyz.get(label, 0.0)))},
                "qiskit_coeff": {"re": float(np.real(qk_map_ixyz.get(label, 0.0))), "im": float(np.imag(qk_map_ixyz.get(label, 0.0)))},
                "delta": {"re": float(np.real(dc)), "im": float(np.imag(dc)), "abs": abs_dc},
            }
        )

    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "L2 hardcoded-vs-baseline trotter order/sign/qubit-mapping audit",
        "locked_settings": locked.__dict__,
        "legacy_modes_used_iter0": iter0_candidate["settings"],
        "diagnostic_a_baseline_initial_amplitudes_legacy_modes": {
            "mismatch_reduction_per_metric": collapse_flags,
            "iter0_max_abs_deltas": {k: iter0_metrics["trajectory_deltas"][k]["max_abs_delta"] for k in TARGET_FIELDS},
            "diag_a_max_abs_deltas": {k: diag_a_metrics["trajectory_deltas"][k]["max_abs_delta"] for k in TARGET_FIELDS},
        },
        "ordered_terms": {
            "hardcoded_native_exyz_order": hc_native,
            "hardcoded_legacy_sorted_exyz_order": sorted(hc_native, key=lambda x: x["label_exyz"]),
            "qiskit_to_list_ixyz_order": qk_order,
        },
        "coefficient_audit": {
            "num_terms_hardcoded": len(hc_map_ixyz),
            "num_terms_qiskit": len(qk_map_ixyz),
            "num_terms_union": len(union),
            "max_abs_coeff_delta": max_abs_coeff_delta,
            "rows": coeff_delta_rows,
        },
        "qubit_index_mapping_notes": {
            "pauli_string_order": "left-to-right = q_(n-1)...q_0 (qubit 0 is rightmost char)",
            "blocked_mode_map_l2": {"up_site0_qubit": 0, "up_site1_qubit": 1, "dn_site0_qubit": 2, "dn_site1_qubit": 3},
            "number_operator_convention": "n_p = (I - Z_p)/2",
            "suzuki_order2_step_used": "forward half-step over ordered term list then reverse half-step",
        },
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Iterative patched run(s): baseline-locked initial amplitudes + qiskit to_list order + per-time reps semantics.
    final_iter_idx = 0
    final_candidate = iter0_candidate
    final_metrics = iter0_metrics
    per_iter_reports: list[dict[str, Any]] = []

    for k in range(1, MAX_ITERS + 1):
        cand = _simulate_l2(
            locked,
            ns,
            initial_state_mode="baseline_amplitudes",
            term_order_mode="baseline_qiskit_order_hardcoded_coeffs",
            time_semantics_mode="baseline_per_time_fixed_reps",
        )
        metrics = _compare_to_baseline(cand, baseline)
        metrics_path = ROOT / f"hardcoded_vs_qiskit_l2_match_metrics_iter{k}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        pdf_report_path = ROOT / f"hardcoded_vs_qiskit_l2_match_report_iter{k}.pdf"
        _write_iter_pdf(pdf_report_path, f"iter{k}", baseline, cand, metrics)
        per_iter_reports.append(
            {
                "iter": k,
                "metrics_json": str(metrics_path.relative_to(ROOT)),
                "report_pdf": str(pdf_report_path.relative_to(ROOT)),
                "pass": bool(metrics["acceptance"]["pass"]),
            }
        )
        final_iter_idx = k
        final_candidate = cand
        final_metrics = metrics
        if bool(metrics["acceptance"]["pass"]):
            break

    final_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase1_l2_hardcoded_vs_frozen_qiskit",
        "baseline_json": str(BASELINE_PATH.relative_to(ROOT)),
        "iter0_metrics_json": str(ITER0_METRICS_PATH.relative_to(ROOT)),
        "iter0_report_pdf": str(ITER0_REPORT_PDF_PATH.relative_to(ROOT)),
        "diagnostics": {
            "diagnostic_a_included_in": str(AUDIT_PATH.relative_to(ROOT)),
            "diagnostic_b_included_in": str(AUDIT_PATH.relative_to(ROOT)),
        },
        "iterations": per_iter_reports,
        "final_iteration_index": final_iter_idx,
        "final_metrics": final_metrics,
        "pass": bool(final_metrics["acceptance"]["pass"]),
    }
    FINAL_METRICS_PATH.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    _write_iter_pdf(FINAL_REPORT_PDF_PATH, f"iter{final_iter_idx}_final", baseline, final_candidate, final_metrics)

    print(f"Wrote iter0 metrics: {ITER0_METRICS_PATH.name}")
    print(f"Wrote iter0 PDF report: {ITER0_REPORT_PDF_PATH.name}")
    print(f"Wrote audit: {AUDIT_PATH.name}")
    print(f"Wrote final metrics: {FINAL_METRICS_PATH.name}")
    print(f"Wrote final PDF report: {FINAL_REPORT_PDF_PATH.name}")


if __name__ == "__main__":
    main()
