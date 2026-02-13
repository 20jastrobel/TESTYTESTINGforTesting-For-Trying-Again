#!/usr/bin/env python3
"""
L=3/L=4 hardcoded-vs-Qiskit time-dynamics comparison with PDF outputs.

This validation script keeps the scope narrow:
- fixed model settings aligned with the phase-1 L=2 setup
- HF initial state (blocked ordering, half-filling)
- second-order Suzuki-Trotter with fixed reps per sampled time
- side-by-side hardcoded vs Qiskit trajectories + pass/fail metrics
"""

from __future__ import annotations

import json
import math
import os
import sys
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

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian

LOCKED = {
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

TARGET_METRICS = [
    "fidelity",
    "energy_trotter",
    "n_up_site0_trotter",
    "n_dn_site0_trotter",
    "doublon_trotter",
]

THRESHOLDS = {
    "ground_state_energy_abs_delta": 1e-8,
    "fidelity_max_abs_delta": 1e-4,
    "energy_trotter_max_abs_delta": 1e-3,
    "n_up_site0_trotter_max_abs_delta": 5e-3,
    "n_dn_site0_trotter_max_abs_delta": 5e-3,
    "doublon_trotter_max_abs_delta": 1e-3,
}


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
        raise ValueError("Zero-norm state encountered.")
    return psi / nrm


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((num_sites + 1) // 2, num_sites // 2)


def _hf_state_blocked(num_sites: int) -> np.ndarray:
    n_up, n_dn = _half_filled_particles(num_sites)
    nq = 2 * num_sites
    occupied = list(range(n_up)) + list(range(num_sites, num_sites + n_dn))
    idx = 0
    for q in occupied:
        idx |= (1 << q)
    psi = np.zeros(1 << nq, dtype=complex)
    psi[idx] = 1.0 + 0.0j
    return psi


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
        raise ValueError(f"Invalid Pauli symbol '{op}' in label '{label_exyz}'")

    return CompiledPauliAction(label_exyz=label_exyz, perm=perm, phase=phase)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    out = np.empty_like(psi)
    out[action.perm] = action.phase * psi
    return out


def _apply_exp_term(psi: np.ndarray, action: CompiledPauliAction, coeff: complex, alpha: float, tol: float = 1e-12) -> np.ndarray:
    if abs(coeff.imag) > tol:
        raise ValueError(f"Non-negligible imaginary coefficient for {action.label_exyz}: {coeff}")
    theta = float(alpha) * float(coeff.real)
    ppsi = _apply_compiled_pauli(psi, action)
    return math.cos(theta) * psi - 1j * math.sin(theta) * ppsi


def _evolve_trotter_suzuki2_absolute(
    psi0: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    compiled_actions: dict[str, CompiledPauliAction],
    time_value: float,
    trotter_steps: int,
) -> np.ndarray:
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return psi
    dt = float(time_value) / float(trotter_steps)
    half = 0.5 * dt
    for _ in range(trotter_steps):
        for label in ordered_labels_exyz:
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
        for label in reversed(ordered_labels_exyz):
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
    return _normalize_state(psi)


def _qiskit_terms_ixyz_ordered(num_sites: int) -> list[tuple[str, complex]]:
    def interleaved_to_blocked_permutation(n_sites: int) -> list[int]:
        return [idx for site in range(n_sites) for idx in (site, n_sites + site)]

    boundary = BoundaryCondition.PERIODIC if LOCKED["boundary"] == "periodic" else BoundaryCondition.OPEN
    lattice = LineLattice(
        num_nodes=num_sites,
        edge_parameter=-float(LOCKED["hopping_t"]),
        onsite_parameter=0.0,
        boundary_condition=boundary,
    )
    ferm_op = FermiHubbardModel(lattice=lattice, onsite_interaction=float(LOCKED["onsite_u"])).second_q_op()
    if LOCKED["ordering"] == "blocked":
        ferm_op = ferm_op.permute_indices(interleaved_to_blocked_permutation(num_sites))
    qop = JordanWignerMapper().map(ferm_op).simplify(atol=1e-12)
    return [(str(label), complex(coeff)) for label, coeff in qop.to_list()]


def _hardcoded_terms_exyz(num_sites: int) -> list[tuple[str, complex]]:
    poly = build_hubbard_hamiltonian(
        dims=num_sites,
        t=float(LOCKED["hopping_t"]),
        U=float(LOCKED["onsite_u"]),
        v=float(LOCKED["dv"]),
        repr_mode="JW",
        indexing=str(LOCKED["ordering"]),
        pbc=(str(LOCKED["boundary"]) == "periodic"),
    )
    coeffs: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1e-12:
            continue
        if label not in coeffs:
            coeffs[label] = 0.0 + 0.0j
            order.append(label)
        coeffs[label] += coeff
    return [(label, coeffs[label]) for label in order if abs(coeffs[label]) > 1e-12]


def _hamiltonian_matrix_from_coeff_map(coeff_map_exyz: dict[str, complex], nq: int) -> np.ndarray:
    terms_ixyz = [(_to_ixyz(lbl), coeff) for lbl, coeff in coeff_map_exyz.items()]
    if not terms_ixyz:
        terms_ixyz = [("I" * nq, 0.0)]
    return np.asarray(SparsePauliOp.from_list(terms_ixyz).to_matrix(sparse=False), dtype=complex)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


def _occupation_site0(psi: np.ndarray, num_sites: int) -> tuple[float, float]:
    probs = np.abs(psi) ** 2
    n_up = 0.0
    n_dn = 0.0
    for idx, p in enumerate(probs):
        n_up += float((idx >> 0) & 1) * float(p)
        n_dn += float((idx >> num_sites) & 1) * float(p)
    return n_up, n_dn


def _doublon_total(psi: np.ndarray, num_sites: int) -> float:
    probs = np.abs(psi) ** 2
    out = 0.0
    for idx, p in enumerate(probs):
        dcount = 0
        for site in range(num_sites):
            up = (idx >> site) & 1
            dn = (idx >> (num_sites + site)) & 1
            dcount += int(up * dn)
        out += float(dcount) * float(p)
    return out


def _first_crossing(times: np.ndarray, vals: np.ndarray, thr: float) -> float | None:
    idx = np.where(vals > thr)[0]
    if idx.size == 0:
        return None
    return float(times[int(idx[0])])


def _simulate_one_path(
    *,
    num_sites: int,
    psi0: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    compiled_actions: dict[str, CompiledPauliAction],
    hmat: np.ndarray,
) -> dict[str, Any]:
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T
    times = np.linspace(0.0, float(LOCKED["t_final"]), int(LOCKED["num_times"]))

    out_rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for t in times:
        t_f = float(t)
        psi_exact = evecs @ (np.exp(-1j * evals * t_f) * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)
        psi_trot = _evolve_trotter_suzuki2_absolute(
            psi0,
            ordered_labels_exyz,
            coeff_map_exyz,
            compiled_actions,
            t_f,
            int(LOCKED["trotter_steps"]),
        )
        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up, n_dn = _occupation_site0(psi_trot, num_sites)
        n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)

        out_rows.append(
            {
                "time": t_f,
                "fidelity": fidelity,
                "energy_exact": _expectation_hamiltonian(psi_exact, hmat),
                "energy_trotter": _expectation_hamiltonian(psi_trot, hmat),
                "n_up_site0_exact": float(n_up_exact),
                "n_up_site0_trotter": float(n_up),
                "n_dn_site0_exact": float(n_dn_exact),
                "n_dn_site0_trotter": float(n_dn),
                "doublon_exact": _doublon_total(psi_exact, num_sites),
                "doublon_trotter": _doublon_total(psi_trot, num_sites),
            }
        )
        exact_states.append(psi_exact)

    return {"trajectory": out_rows, "exact_states": exact_states}


def _compare_paths(
    base: dict[str, Any],
    cand: dict[str, Any],
    gs_delta: float,
) -> dict[str, Any]:
    base_rows = base["trajectory"]
    cand_rows = cand["trajectory"]
    times = np.array([float(r["time"]) for r in base_rows], dtype=float)

    out: dict[str, Any] = {
        "ground_state_energy_abs_delta": float(gs_delta),
        "trajectory_deltas": {},
    }

    for key in TARGET_METRICS:
        b = np.array([float(r[key]) for r in base_rows], dtype=float)
        c = np.array([float(r[key]) for r in cand_rows], dtype=float)
        d = np.abs(c - b)
        out["trajectory_deltas"][key] = {
            "max_abs_delta": float(np.max(d)),
            "mean_abs_delta": float(np.mean(d)),
            "final_abs_delta": float(d[-1]),
            "first_time_abs_delta_gt_1e-4": _first_crossing(times, d, 1e-4),
            "first_time_abs_delta_gt_1e-3": _first_crossing(times, d, 1e-3),
        }

    checks = {
        "ground_state_energy_abs_delta": out["ground_state_energy_abs_delta"] <= THRESHOLDS["ground_state_energy_abs_delta"],
        "fidelity_max_abs_delta": out["trajectory_deltas"]["fidelity"]["max_abs_delta"] <= THRESHOLDS["fidelity_max_abs_delta"],
        "energy_trotter_max_abs_delta": out["trajectory_deltas"]["energy_trotter"]["max_abs_delta"] <= THRESHOLDS["energy_trotter_max_abs_delta"],
        "n_up_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_up_site0_trotter"]["max_abs_delta"] <= THRESHOLDS["n_up_site0_trotter_max_abs_delta"],
        "n_dn_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_dn_site0_trotter"]["max_abs_delta"] <= THRESHOLDS["n_dn_site0_trotter_max_abs_delta"],
        "doublon_trotter_max_abs_delta": out["trajectory_deltas"]["doublon_trotter"]["max_abs_delta"] <= THRESHOLDS["doublon_trotter_max_abs_delta"],
    }
    out["acceptance"] = {"checks": checks, "pass": bool(all(checks.values())), "thresholds": THRESHOLDS}
    return out


def _write_pdf(
    *,
    pdf_path: Path,
    num_sites: int,
    base: dict[str, Any],
    cand: dict[str, Any],
    metrics: dict[str, Any],
    phase_series: np.ndarray,
) -> None:
    times = np.array([float(r["time"]) for r in base["trajectory"]], dtype=float)
    markevery = max(1, times.size // 24)

    def arr(src: dict[str, Any], key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in src["trajectory"]], dtype=float)

    b_fid = arr(base, "fidelity")
    c_fid = arr(cand, "fidelity")
    b_e_exact = arr(base, "energy_exact")
    c_e_exact = arr(cand, "energy_exact")
    b_e = arr(base, "energy_trotter")
    c_e = arr(cand, "energy_trotter")
    b_nu_exact = arr(base, "n_up_site0_exact")
    c_nu_exact = arr(cand, "n_up_site0_exact")
    b_nu = arr(base, "n_up_site0_trotter")
    c_nu = arr(cand, "n_up_site0_trotter")
    b_nd_exact = arr(base, "n_dn_site0_exact")
    c_nd_exact = arr(cand, "n_dn_site0_exact")
    b_nd = arr(base, "n_dn_site0_trotter")
    c_nd = arr(cand, "n_dn_site0_trotter")
    b_d_exact = arr(base, "doublon_exact")
    c_d_exact = arr(cand, "doublon_exact")
    b_d = arr(base, "doublon_trotter")
    c_d = arr(cand, "doublon_trotter")

    with PdfPages(str(pdf_path)) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]

        ax00.plot(
            times, b_fid, label="qiskit fidelity",
            linewidth=1.8, color="#0b3d91", marker="o", markersize=3, markevery=markevery
        )
        ax00.plot(
            times, c_fid, label="hardcoded fidelity",
            linewidth=1.5, linestyle="--", color="#e15759", marker="^", markersize=3, markevery=markevery
        )
        ax00.set_title("Fidelity(t)")
        ax00.grid(alpha=0.25)
        ax00.legend(fontsize=8)

        ax01.plot(
            times, b_e_exact, label="qiskit <H> exact",
            linewidth=2.0, color="#111111", marker="o", markersize=3, markevery=markevery
        )
        ax01.plot(
            times, c_e_exact, label="hardcoded <H> exact",
            linewidth=1.6, linestyle=":", color="#7f7f7f", marker="s", markersize=3,
            markerfacecolor="none", markevery=markevery
        )
        ax01.plot(
            times, b_e, label="qiskit <H> trotter",
            linewidth=1.6, linestyle="--", color="#2ca02c", marker="^", markersize=3, markevery=markevery
        )
        ax01.plot(
            times, c_e, label="hardcoded <H> trotter",
            linewidth=1.4, linestyle="-.", color="#d62728", marker="v", markersize=3,
            markerfacecolor="none", markevery=markevery
        )
        ax01.set_title("Energy with Exact Overlays")
        ax01.grid(alpha=0.25)
        ax01.legend(fontsize=8)

        ax10.plot(
            times, b_nu_exact, label="qiskit n_up0 exact",
            linewidth=2.0, color="#17becf", marker="o", markersize=3, markevery=markevery
        )
        ax10.plot(
            times, c_nu_exact, label="hardcoded n_up0 exact",
            linewidth=1.4, linestyle=":", color="#0f7f8b", marker="s", markersize=3,
            markerfacecolor="none", markevery=markevery
        )
        ax10.plot(
            times, b_nu, label="qiskit n_up0 trotter",
            linewidth=1.4, linestyle="--", color="#1f78b4", marker="^", markersize=3, markevery=markevery
        )
        ax10.plot(
            times, c_nu, label="hardcoded n_up0 trotter",
            linewidth=1.2, linestyle="-.", color="#08306b", marker="v", markersize=3,
            markerfacecolor="none", markevery=markevery
        )
        ax10.plot(
            times, b_nd_exact, label="qiskit n_dn0 exact",
            linewidth=2.0, color="#9467bd", marker="o", markersize=3, markevery=markevery
        )
        ax10.plot(
            times, c_nd_exact, label="hardcoded n_dn0 exact",
            linewidth=1.4, linestyle=":", color="#6f4d8f", marker="s", markersize=3,
            markerfacecolor="none", markevery=markevery
        )
        ax10.plot(
            times, b_nd, label="qiskit n_dn0 trotter",
            linewidth=1.4, linestyle="--", color="#ff7f0e", marker="^", markersize=3, markevery=markevery
        )
        ax10.plot(
            times, c_nd, label="hardcoded n_dn0 trotter",
            linewidth=1.2, linestyle="-.", color="#8c2d04", marker="v", markersize=3,
            markerfacecolor="none", markevery=markevery
        )
        ax10.set_title("Site-0 Occupations with Exact Overlays")
        ax10.set_xlabel("Time")
        ax10.grid(alpha=0.25)
        ax10.legend(fontsize=8)

        ax11.plot(
            times, b_d_exact, label="qiskit doublon exact",
            linewidth=2.0, color="#8c564b", marker="o", markersize=3, markevery=markevery
        )
        ax11.plot(
            times, c_d_exact, label="hardcoded doublon exact",
            linewidth=1.4, linestyle=":", color="#5f3a33", marker="s", markersize=3,
            markerfacecolor="none", markevery=markevery
        )
        ax11.plot(
            times, b_d, label="qiskit doublon trotter",
            linewidth=1.4, linestyle="--", color="#e377c2", marker="^", markersize=3, markevery=markevery
        )
        ax11.plot(
            times, c_d, label="hardcoded doublon trotter",
            linewidth=1.2, linestyle="-.", color="#c251a1", marker="v", markersize=3,
            markerfacecolor="none", markevery=markevery
        )
        ax11.set_title("Total Doublon with Exact Overlays")
        ax11.set_xlabel("Time")
        ax11.grid(alpha=0.25)
        ax11.legend(fontsize=8)

        fig.suptitle(f"L={num_sites} Time Dynamics: hardcoded vs Qiskit", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        fig2, axes2 = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        bx00, bx01 = axes2[0, 0], axes2[0, 1]
        bx10, bx11 = axes2[1, 0], axes2[1, 1]

        bx00.plot(
            times, np.abs(c_fid - b_fid), color="#1f77b4",
            marker="o", markersize=3, markevery=markevery
        )
        bx00.set_title("|delta fidelity|")
        bx00.grid(alpha=0.25)

        bx01.plot(
            times, np.abs(b_e - b_e_exact), label="|qiskit trotter - qiskit exact|",
            color="#2ca02c", marker="s", markersize=3, markevery=markevery
        )
        bx01.plot(
            times, np.abs(c_e - c_e_exact), label="|hardcoded trotter - hardcoded exact|",
            color="#d62728", marker="^", markersize=3, markevery=markevery
        )
        bx01.set_title("Energy Trotter-vs-Exact")
        bx01.grid(alpha=0.25)
        bx01.legend(fontsize=8)

        bx10.plot(
            times, np.abs(b_nu - b_nu_exact), label="|qiskit n_up0 trotter-exact|",
            color="#17becf", marker="o", markersize=3, markevery=markevery
        )
        bx10.plot(
            times, np.abs(c_nu - c_nu_exact), label="|hardcoded n_up0 trotter-exact|",
            color="#0f7f8b", marker="s", markersize=3, markevery=markevery
        )
        bx10.plot(
            times, np.abs(b_nd - b_nd_exact), label="|qiskit n_dn0 trotter-exact|",
            color="#9467bd", marker="^", markersize=3, markevery=markevery
        )
        bx10.plot(
            times, np.abs(c_nd - c_nd_exact), label="|hardcoded n_dn0 trotter-exact|",
            color="#6f4d8f", marker="v", markersize=3, markevery=markevery
        )
        bx10.set_title("Occupation Trotter-vs-Exact")
        bx10.set_xlabel("Time")
        bx10.grid(alpha=0.25)
        bx10.legend(fontsize=8)

        bx11.plot(
            times, np.abs(b_d - b_d_exact), label="|qiskit doublon trotter-exact|",
            color="#8c564b", marker="o", markersize=3, markevery=markevery
        )
        bx11.plot(
            times, np.abs(c_d - c_d_exact), label="|hardcoded doublon trotter-exact|",
            color="#5f3a33", marker="s", markersize=3, markevery=markevery
        )
        bx11_t = bx11.twinx()
        bx11_t.plot(
            times, phase_series, label="phase(exact overlap)",
            color="#ff7f0e", linestyle="--", marker="D", markersize=2.5, markevery=markevery
        )
        bx11.set_title("Doublon Trotter-vs-Exact + Phase")
        bx11.set_xlabel("Time")
        bx11.grid(alpha=0.25)
        ln1, lb1 = bx11.get_legend_handles_labels()
        ln2, lb2 = bx11_t.get_legend_handles_labels()
        bx11.legend(ln1 + ln2, lb1 + lb2, fontsize=8, loc="upper right")

        fig2.suptitle(f"L={num_sites} Delta Diagnostics", fontsize=14)
        fig2.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3 = plt.figure(figsize=(11.0, 8.5))
        ax3 = fig3.add_subplot(111)
        ax3.axis("off")
        td = metrics["trajectory_deltas"]
        checks = metrics["acceptance"]["checks"]
        lines = [
            f"L={num_sites} validation summary",
            "",
            f"ground_state_energy_abs_delta = {metrics['ground_state_energy_abs_delta']:.12e}",
            f"fidelity max/mean/final delta = {td['fidelity']['max_abs_delta']:.12e} / {td['fidelity']['mean_abs_delta']:.12e} / {td['fidelity']['final_abs_delta']:.12e}",
            f"energy_trotter max/mean/final delta = {td['energy_trotter']['max_abs_delta']:.12e} / {td['energy_trotter']['mean_abs_delta']:.12e} / {td['energy_trotter']['final_abs_delta']:.12e}",
            f"n_up_site0_trotter max/mean/final delta = {td['n_up_site0_trotter']['max_abs_delta']:.12e} / {td['n_up_site0_trotter']['mean_abs_delta']:.12e} / {td['n_up_site0_trotter']['final_abs_delta']:.12e}",
            f"n_dn_site0_trotter max/mean/final delta = {td['n_dn_site0_trotter']['max_abs_delta']:.12e} / {td['n_dn_site0_trotter']['mean_abs_delta']:.12e} / {td['n_dn_site0_trotter']['final_abs_delta']:.12e}",
            f"doublon_trotter max/mean/final delta = {td['doublon_trotter']['max_abs_delta']:.12e} / {td['doublon_trotter']['mean_abs_delta']:.12e} / {td['doublon_trotter']['final_abs_delta']:.12e}",
            "",
            f"Checks: {checks}",
            f"PASS: {metrics['acceptance']['pass']}",
            "",
            "Locked settings:",
            json.dumps(LOCKED, indent=2),
        ]
        ax3.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        pdf.savefig(fig3)
        plt.close(fig3)


def _write_vqe_comparison_page(pdf_path: Path) -> None:
    rows: dict[int, dict[str, float | None]] = {
        2: {"hardcoded_vqe": None, "qiskit_vqe": None, "exact": None, "qpe": None, "hpe": None},
        3: {"hardcoded_vqe": None, "qiskit_vqe": None, "exact": None, "qpe": None, "hpe": None},
        4: {"hardcoded_vqe": None, "qiskit_vqe": None, "exact": None, "qpe": None, "hpe": None},
    }

    # Qiskit VQE for L=2,3 from committed comparison.
    cmp_path = ROOT / "Tests" / "vqe_hardcoded_vs_qiskit_comparison.json"
    if cmp_path.exists():
        cmp_obj = json.loads(cmp_path.read_text())
        for row in cmp_obj.get("comparison", []):
            L = int(row["lattice_sites"])
            if L in rows:
                rows[L]["qiskit_vqe"] = float(row["qiskit_vqe_energy"])

    # Qiskit VQE for L=4 from long-run artifact.
    l4_q_path = ROOT / "Tests" / "vqe_l4_longrun_results.json"
    if l4_q_path.exists():
        l4_q_obj = json.loads(l4_q_path.read_text())
        rows[4]["qiskit_vqe"] = float(l4_q_obj["results"]["best_vqe_energy_observed"])
        rows[4]["exact"] = float(l4_q_obj["results"]["exact_filtered_energy"])

    # Hardcoded VQE + exact + QPE + HPE from qpe hardcoded exports.
    for L in (2, 3, 4):
        p = ROOT / "pydephasing" / "quantum" / "exports" / f"qpe_hardcoded_sector_run_L{L}_blocked.json"
        if not p.exists():
            continue
        obj = json.loads(p.read_text())
        rows[L]["hardcoded_vqe"] = float(obj["reference"]["E_vqe"])
        rows[L]["qpe"] = float(obj["segment2_qpe"]["E_from_qpe"])
        rows[L]["hpe"] = float(obj["segment3_hpe"]["E_hpe"])
        if rows[L]["exact"] is None:
            rows[L]["exact"] = float(obj["reference"]["E_exact_sector"])

    Lvals = [2, 3, 4]
    x = np.arange(len(Lvals), dtype=float)

    def series(key: str) -> np.ndarray:
        vals = []
        for L in Lvals:
            v = rows[L][key]
            vals.append(np.nan if v is None else float(v))
        return np.array(vals, dtype=float)

    exact = series("exact")
    hardcoded_vqe = series("hardcoded_vqe")
    qiskit_vqe = series("qiskit_vqe")
    qpe = series("qpe")
    hpe = series("hpe")

    with PdfPages(str(pdf_path)) as pdf:
        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        ax.plot(x, exact, marker="X", markersize=9, linewidth=2.2, color="#111111", label="Exact")
        ax.plot(x, hardcoded_vqe, marker="s", markersize=7, linewidth=1.8, color="#2ca02c", label="Hardcoded VQE")
        ax.plot(x, qiskit_vqe, marker="o", markersize=7, linewidth=1.8, color="#ff7f0e", label="Qiskit VQE")
        ax.plot(x, qpe, marker="^", markersize=7, linewidth=1.8, linestyle="--", color="#1f77b4", label="QPE")
        ax.plot(x, hpe, marker="D", markersize=7, linewidth=1.8, linestyle="-.", color="#d62728", label="HPE")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L={L}" for L in Lvals])
        ax.set_ylabel("Energy")
        ax.set_title("Energy Comparison: VQE / QPE / HPE vs Exact")
        ax.grid(alpha=0.25)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(11.0, 8.5))
        err_hc_vqe = np.abs(hardcoded_vqe - exact)
        err_qk_vqe = np.abs(qiskit_vqe - exact)
        err_qpe = np.abs(qpe - exact)
        err_hpe = np.abs(hpe - exact)
        width = 0.18
        ax2.bar(x - 1.5 * width, err_hc_vqe, width=width, color="#2ca02c", label="|Hardcoded VQE - Exact|")
        ax2.bar(x - 0.5 * width, err_qk_vqe, width=width, color="#ff7f0e", label="|Qiskit VQE - Exact|")
        ax2.bar(x + 0.5 * width, err_qpe, width=width, color="#1f77b4", label="|QPE - Exact|")
        ax2.bar(x + 1.5 * width, err_hpe, width=width, color="#d62728", label="|HPE - Exact|")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"L={L}" for L in Lvals])
        ax2.set_ylabel("Absolute Error")
        ax2.set_yscale("log")
        ax2.set_title("Absolute Error vs Exact (log scale)")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(fontsize=9)
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3 = plt.figure(figsize=(11.0, 8.5))
        ax3 = fig3.add_subplot(111)
        ax3.axis("off")
        lines = [
            "Data table (energy values)",
            "",
            "Columns: L, exact, hardcoded_vqe, qiskit_vqe, qpe, hpe",
            "",
        ]
        for L in Lvals:
            r = rows[L]
            lines.append(
                f"L={L}  exact={r['exact']}  hardcoded_vqe={r['hardcoded_vqe']}  "
                f"qiskit_vqe={r['qiskit_vqe']}  qpe={r['qpe']}  hpe={r['hpe']}"
            )
        ax3.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
        pdf.savefig(fig3)
        plt.close(fig3)


def run_for_lattice(num_sites: int) -> dict[str, Any]:
    if num_sites not in (3, 4):
        raise ValueError("This script is scoped to L=3 and L=4.")
    nq = 2 * num_sites
    psi0 = _hf_state_blocked(num_sites)

    qk_terms_ixyz = _qiskit_terms_ixyz_ordered(num_sites)
    hc_terms_exyz = _hardcoded_terms_exyz(num_sites)

    qk_coeff_map_exyz = {_to_exyz(lbl): coeff for lbl, coeff in qk_terms_ixyz}
    hc_coeff_map_exyz = {lbl: coeff for lbl, coeff in hc_terms_exyz}
    ordered_labels_exyz = [_to_exyz(lbl) for lbl, _ in qk_terms_ixyz]

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in sorted(set(ordered_labels_exyz))}

    # Coefficient audit in Qiskit order.
    coeff_rows = []
    max_abs_coeff_delta = 0.0
    for lbl in ordered_labels_exyz:
        cq = qk_coeff_map_exyz.get(lbl, 0.0 + 0.0j)
        ch = hc_coeff_map_exyz.get(lbl, 0.0 + 0.0j)
        d = ch - cq
        max_abs_coeff_delta = max(max_abs_coeff_delta, float(abs(d)))
        coeff_rows.append(
            {
                "label_exyz": lbl,
                "qiskit_coeff": {"re": float(cq.real), "im": float(cq.imag)},
                "hardcoded_coeff": {"re": float(ch.real), "im": float(ch.imag)},
                "delta": {"re": float(d.real), "im": float(d.imag), "abs": float(abs(d))},
            }
        )

    h_qk = _hamiltonian_matrix_from_coeff_map(qk_coeff_map_exyz, nq)
    h_hc = _hamiltonian_matrix_from_coeff_map(hc_coeff_map_exyz, nq)
    eval_qk = np.linalg.eigvalsh(h_qk)
    eval_hc = np.linalg.eigvalsh(h_hc)
    gs_delta = float(abs(float(np.min(eval_hc)) - float(np.min(eval_qk))))

    base = _simulate_one_path(
        num_sites=num_sites,
        psi0=psi0,
        ordered_labels_exyz=ordered_labels_exyz,
        coeff_map_exyz=qk_coeff_map_exyz,
        compiled_actions=compiled,
        hmat=h_qk,
    )
    cand = _simulate_one_path(
        num_sites=num_sites,
        psi0=psi0,
        ordered_labels_exyz=ordered_labels_exyz,
        coeff_map_exyz=hc_coeff_map_exyz,
        compiled_actions=compiled,
        hmat=h_hc,
    )
    metrics = _compare_paths(base, cand, gs_delta)

    phase = []
    for a, b in zip(base["exact_states"], cand["exact_states"]):
        ov = np.vdot(a, b)
        phase.append(float(np.angle(ov)))
    phase_arr = np.asarray(phase, dtype=float)

    metrics_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lattice_size": num_sites,
        "settings": {"num_sites": num_sites, **LOCKED},
        "coefficient_audit": {
            "max_abs_coeff_delta": max_abs_coeff_delta,
            "rows": coeff_rows,
        },
        "metrics": metrics,
    }

    metrics_path = ROOT / f"hardcoded_vs_qiskit_l{num_sites}_time_dynamics_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    pdf_path = ROOT / f"hardcoded_vs_qiskit_l{num_sites}_time_dynamics_report.pdf"
    _write_pdf(
        pdf_path=pdf_path,
        num_sites=num_sites,
        base=base,
        cand=cand,
        metrics=metrics,
        phase_series=phase_arr,
    )

    return {
        "lattice_size": num_sites,
        "metrics_json": str(metrics_path.relative_to(ROOT)),
        "report_pdf": str(pdf_path.relative_to(ROOT)),
        "pass": bool(metrics["acceptance"]["pass"]),
        "max_abs_coeff_delta": max_abs_coeff_delta,
        "trajectory_max_abs_deltas": {k: metrics["trajectory_deltas"][k]["max_abs_delta"] for k in TARGET_METRICS},
    }


def main() -> None:
    results = [run_for_lattice(3), run_for_lattice(4)]

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "L=3/L=4 hardcoded-vs-Qiskit time-dynamics comparison with locked phase-1 settings "
            "and PDF trajectory reports."
        ),
        "settings": LOCKED,
        "results": results,
        "all_pass": bool(all(r["pass"] for r in results)),
    }
    summary_path = ROOT / "hardcoded_vs_qiskit_l3_l4_time_dynamics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    vqe_pdf = ROOT / "hardcoded_vs_qiskit_vqe_energy_comparison.pdf"
    _write_vqe_comparison_page(vqe_pdf)

    print(f"Wrote {summary_path.name}")
    for row in results:
        print(
            f"L={row['lattice_size']}: pass={row['pass']} "
            f"metrics={row['metrics_json']} pdf={row['report_pdf']}"
        )
    print(f"Wrote optional VQE PDF page bundle: {vqe_pdf.name}")


if __name__ == "__main__":
    main()
