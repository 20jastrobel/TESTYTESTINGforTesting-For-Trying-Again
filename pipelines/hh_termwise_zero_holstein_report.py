#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydephasing.quantum.ed_hubbard_holstein import (
    build_hh_sector_basis,
    build_hh_sector_hamiltonian_ed,
    matrix_to_dense,
)
from pydephasing.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
)
from pydephasing.quantum.vqe_latex_python_pairs_test import (
    HubbardHolsteinLayerwiseAnsatz,
    apply_exp_pauli_polynomial,
    hubbard_holstein_reference_state,
    vqe_minimize,
)


@dataclass(frozen=True)
class HHRunCase:
    case_id: str
    L: int
    ordering: str
    boundary: str
    J: float
    U: float
    dv: float
    omega0: float
    g: float
    n_ph_max: int
    boson_encoding: str
    reps: int
    restarts: int
    maxiter: int
    seed: int


class HubbardHolsteinTermwiseAnsatz:
    """
    Term-wise HH ansatz built by flattening HH layer groups into single-term evolutions.

    This reuses the validated HH layerwise builder and only changes parameter sharing:
    each term in each group gets its own parameter.
    """

    def __init__(
        self,
        *,
        dims: int,
        J: float,
        U: float,
        omega0: float,
        g: float,
        n_ph_max: int,
        boson_encoding: str = "binary",
        v: Optional[float] = None,
        v_t: Optional[float] = None,
        v0: Optional[float] = None,
        t_eval: Optional[float] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        pbc: bool = False,
        include_zero_point: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ):
        self.layerwise = HubbardHolsteinLayerwiseAnsatz(
            dims=int(dims),
            J=float(J),
            U=float(U),
            omega0=float(omega0),
            g=float(g),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            v=v,
            v_t=v_t,
            v0=v0,
            t_eval=t_eval,
            reps=1,
            repr_mode=str(repr_mode),
            indexing=str(indexing),
            pbc=bool(pbc),
            include_zero_point=bool(include_zero_point),
            coefficient_tolerance=float(coefficient_tolerance),
            sort_terms=bool(sort_terms),
        )

        self.nq = int(self.layerwise.nq)
        self.reps = int(reps)
        if self.reps <= 0:
            raise ValueError("reps must be positive")
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.sort_terms = bool(sort_terms)

        self.base_terms: List[Any] = []
        for _label, group_terms in self.layerwise.layer_term_groups:
            for term in group_terms:
                self.base_terms.append(term.polynomial)

        if not self.base_terms:
            raise ValueError("HubbardHolsteinTermwiseAnsatz produced no terms")
        self.num_parameters = int(self.reps * len(self.base_terms))

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            raise ValueError("theta has wrong length for this ansatz")
        if int(psi_ref.size) != (1 << int(self.nq)):
            raise ValueError("psi_ref length must be 2^nq")

        coeff_tol = self.coefficient_tolerance if coefficient_tolerance is None else float(coefficient_tolerance)
        sort_flag = self.sort_terms if sort_terms is None else bool(sort_terms)

        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for term_poly in self.base_terms:
                psi = apply_exp_pauli_polynomial(
                    psi,
                    term_poly,
                    float(theta[k]),
                    ignore_identity=ignore_identity,
                    coefficient_tolerance=coeff_tol,
                    sort_terms=sort_flag,
                )
                k += 1
        return psi


def _half_filling(L: int) -> Tuple[int, int]:
    return ((int(L) + 1) // 2, int(L) // 2)


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _parse_tol(raw: str) -> float:
    token = str(raw).strip().lower()
    if token == "auto":
        return -1.0
    return float(raw)


def _ham_term_count(h_poly: Any, tol: float = 1e-12) -> int:
    count = 0
    for term in h_poly.return_polynomial():
        if abs(complex(term.p_coeff)) > float(tol):
            count += 1
    return int(count)


def _exact_filtered_energy(case: HHRunCase, num_particles: Tuple[int, int]) -> float:
    basis = build_hh_sector_basis(
        dims=int(case.L),
        n_ph_max=int(case.n_ph_max),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        indexing=str(case.ordering),
        boson_encoding=str(case.boson_encoding),
    )
    h_ed = build_hh_sector_hamiltonian_ed(
        dims=int(case.L),
        J=float(case.J),
        U=float(case.U),
        omega0=float(case.omega0),
        g=float(case.g),
        n_ph_max=int(case.n_ph_max),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        indexing=str(case.ordering),
        boson_encoding=str(case.boson_encoding),
        pbc=(str(case.boundary) == "periodic"),
        delta_v=[float(case.dv)] * int(case.L),
        include_zero_point=True,
        basis=basis,
        sparse=False,
        return_basis=False,
    )
    dense = matrix_to_dense(h_ed)
    evals = np.linalg.eigvalsh(dense)
    return float(np.real(evals[0]))


def _run_case(case: HHRunCase, tol_factor: float) -> Dict[str, Any]:
    num_particles = _half_filling(case.L)
    v_t = None
    v0 = None
    if abs(float(case.dv)) > 1e-15:
        # Match CLI convention Hv = -dv * n using HH drive term +(v_t-v0)*n.
        v_t = 0.0
        v0 = float(case.dv)

    h_poly = build_hubbard_holstein_hamiltonian(
        dims=int(case.L),
        J=float(case.J),
        U=float(case.U),
        omega0=float(case.omega0),
        g=float(case.g),
        n_ph_max=int(case.n_ph_max),
        boson_encoding=str(case.boson_encoding),
        v_t=v_t,
        v0=v0,
        t_eval=None,
        repr_mode="JW",
        indexing=str(case.ordering),
        pbc=(str(case.boundary) == "periodic"),
        include_zero_point=True,
    )

    exact_e = _exact_filtered_energy(case, num_particles=num_particles)

    ansatz = HubbardHolsteinTermwiseAnsatz(
        dims=int(case.L),
        J=float(case.J),
        U=float(case.U),
        omega0=float(case.omega0),
        g=float(case.g),
        n_ph_max=int(case.n_ph_max),
        boson_encoding=str(case.boson_encoding),
        v=None,
        v_t=v_t,
        v0=v0,
        t_eval=None,
        reps=int(case.reps),
        repr_mode="JW",
        indexing=str(case.ordering),
        pbc=(str(case.boundary) == "periodic"),
        include_zero_point=True,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=int(case.L),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        n_ph_max=int(case.n_ph_max),
        boson_encoding=str(case.boson_encoding),
        indexing=str(case.ordering),
    )
    t0 = time.perf_counter()
    vqe = vqe_minimize(
        h_poly,
        ansatz,
        np.asarray(psi_ref, dtype=complex),
        restarts=int(case.restarts),
        seed=int(case.seed),
        maxiter=int(case.maxiter),
        method="SLSQP",
    )
    runtime_s = float(time.perf_counter() - t0)

    abs_err = float(abs(float(vqe.energy) - float(exact_e)))
    rel_err = float(abs_err / max(1.0, abs(float(exact_e))))
    tol = float(abs(float(exact_e)) * float(tol_factor))
    passed = bool(abs_err <= tol)

    n_qubits = int(getattr(ansatz, "nq", int(round(math.log2(len(psi_ref))))))
    return {
        "case": asdict(case),
        "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        "n_qubits": int(n_qubits),
        "ham_terms": int(_ham_term_count(h_poly)),
        "ansatz": "hh_termwise",
        "parameterization": "termwise",
        "num_params": int(ansatz.num_parameters),
        "exact_energy_filtered": float(exact_e),
        "vqe_energy": float(vqe.energy),
        "abs_err": float(abs_err),
        "rel_err": float(rel_err),
        "tol": float(tol),
        "pass": bool(passed),
        "vqe": {
            "success": bool(vqe.success),
            "message": str(vqe.message),
            "best_restart": int(vqe.best_restart),
            "nfev": int(vqe.nfev),
            "nit": int(vqe.nit),
            "runtime_s": float(runtime_s),
        },
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    cols = [
        "case_id",
        "L",
        "boundary",
        "ordering",
        "n_up",
        "n_dn",
        "J",
        "U",
        "g",
        "omega0",
        "dv",
        "n_ph_max",
        "boson_encoding",
        "n_qubits",
        "ham_terms",
        "ansatz",
        "parameterization",
        "reps",
        "num_params",
        "restarts",
        "maxiter",
        "seed",
        "best_restart",
        "nfev",
        "nit",
        "runtime_s",
        "success",
        "E_exact_filtered",
        "E_vqe_termwise",
        "abs_err",
        "rel_err",
        "tol",
        "pass",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            case = row["case"]
            npart = row["num_particles"]
            vqe = row["vqe"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "L": case["L"],
                    "boundary": case["boundary"],
                    "ordering": case["ordering"],
                    "n_up": npart["n_up"],
                    "n_dn": npart["n_dn"],
                    "J": case["J"],
                    "U": case["U"],
                    "g": case["g"],
                    "omega0": case["omega0"],
                    "dv": case["dv"],
                    "n_ph_max": case["n_ph_max"],
                    "boson_encoding": case["boson_encoding"],
                    "n_qubits": row["n_qubits"],
                    "ham_terms": row["ham_terms"],
                    "ansatz": row["ansatz"],
                    "parameterization": row["parameterization"],
                    "reps": case["reps"],
                    "num_params": row["num_params"],
                    "restarts": case["restarts"],
                    "maxiter": case["maxiter"],
                    "seed": case["seed"],
                    "best_restart": vqe["best_restart"],
                    "nfev": vqe["nfev"],
                    "nit": vqe["nit"],
                    "runtime_s": vqe["runtime_s"],
                    "success": vqe["success"],
                    "E_exact_filtered": row["exact_energy_filtered"],
                    "E_vqe_termwise": row["vqe_energy"],
                    "abs_err": row["abs_err"],
                    "rel_err": row["rel_err"],
                    "tol": row["tol"],
                    "pass": row["pass"],
                }
            )


def _table_page(pdf: PdfPages, rows: Sequence[Dict[str, Any]], title: str) -> None:
    fig = plt.figure(figsize=(17, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)

    headers = [
        "Case ID",
        "L",
        "Boundary",
        "Ordering",
        "N_up",
        "N_dn",
        "J",
        "U",
        "g",
        "omega0",
        "dv",
        "n_ph_max",
        "encoding",
        "nq",
        "terms",
        "ansatz",
        "reps",
        "params",
        "restarts",
        "maxiter",
        "seed",
        "best_r",
        "nfev",
        "nit",
        "runtime_s",
        "success",
        "E_exact",
        "E_VQE",
        "|err|",
        "rel_err",
        "tol",
        "pass",
    ]

    body: List[List[str]] = []
    for r in rows:
        c = r["case"]
        npart = r["num_particles"]
        vqe = r["vqe"]
        body.append(
            [
                str(c["case_id"]),
                str(c["L"]),
                str(c["boundary"]),
                str(c["ordering"]),
                str(npart["n_up"]),
                str(npart["n_dn"]),
                f"{float(c['J']):.3g}",
                f"{float(c['U']):.3g}",
                f"{float(c['g']):.3g}",
                f"{float(c['omega0']):.3g}",
                f"{float(c['dv']):.3g}",
                str(c["n_ph_max"]),
                str(c["boson_encoding"]),
                str(r["n_qubits"]),
                str(r["ham_terms"]),
                "hh_termwise",
                str(c["reps"]),
                str(r["num_params"]),
                str(c["restarts"]),
                str(c["maxiter"]),
                str(c["seed"]),
                str(vqe["best_restart"]),
                str(vqe["nfev"]),
                str(vqe["nit"]),
                f"{float(vqe['runtime_s']):.2f}",
                str(vqe["success"]),
                f"{float(r['exact_energy_filtered']): .12f}",
                f"{float(r['vqe_energy']): .12f}",
                f"{float(r['abs_err']):.3e}",
                f"{float(r['rel_err']):.3e}",
                f"{float(r['tol']):.3e}",
                str(bool(r["pass"])),
            ]
        )

    table = ax.table(
        cellText=body,
        colLabels=headers,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.3)
    table.scale(1.0, 1.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_pages(pdf: PdfPages, rows: Sequence[Dict[str, Any]]) -> None:
    labels = [str(r["case"]["case_id"]) for r in rows]
    x = np.arange(len(labels), dtype=float)
    e_exact = np.array([float(r["exact_energy_filtered"]) for r in rows], dtype=float)
    e_vqe = np.array([float(r["vqe_energy"]) for r in rows], dtype=float)
    abs_err = np.array([float(r["abs_err"]) for r in rows], dtype=float)
    runtime = np.array([float(r["vqe"]["runtime_s"]) for r in rows], dtype=float)

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(x, e_exact, marker="o", linewidth=2.0, color="#111111", label="E_exact_filtered")
    ax1.plot(x, e_vqe, marker="s", linewidth=1.8, linestyle="--", color="#d62728", label="E_VQE_termwise")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("Energy")
    ax1.set_title("HH Hubbard-Limit Energy Comparison (g=0, omega0=0)")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)

    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    bars = ax2.bar(x, abs_err, color="#1f77b4", alpha=0.9)
    ax2.set_ylabel("|E_VQE - E_exact|")
    ax2.set_title("Absolute Error by Case")
    ax2.grid(axis="y", alpha=0.25)
    for i, b in enumerate(bars):
        ax2.text(float(b.get_x() + b.get_width() / 2.0), float(b.get_height()), f"{abs_err[i]:.3e}", ha="center", va="bottom", fontsize=8)

    ax3.plot(x, runtime, marker="^", color="#2ca02c", linewidth=1.8)
    ax3.set_ylabel("Runtime (s)")
    ax3.set_xlabel("Case")
    ax3.grid(alpha=0.25)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=25, ha="right")
    fig2.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Heavy HH term-wise VQE benchmark in the Hubbard limit (g=0, omega0=0), "
            "with exact filtered ED comparison and PDF report."
        )
    )
    parser.add_argument("--L", type=str, default="2,3,4", help="Comma-separated system sizes.")
    parser.add_argument("--orderings", type=str, default="blocked")
    parser.add_argument("--boundaries", type=str, default="open,periodic")
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--omega0", type=float, default=0.0)
    parser.add_argument("--g", type=float, default=0.0)
    parser.add_argument("--n-ph-max", type=int, default=1)
    parser.add_argument("--boson-encoding", choices=["binary"], default="binary")
    parser.add_argument("--vqe-reps", type=int, default=2)
    parser.add_argument("--vqe-restarts", type=int, default=20)
    parser.add_argument("--vqe-maxiter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tol", type=str, default="auto", help="Absolute tolerance or 'auto' (|E_exact|*1e-3).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hh_vqe_zero_holstein_L2_L4_termwise_heavy"),
    )
    parser.add_argument("--output-prefix", type=str, default="hh_zero_holstein_L2_L4_termwise_heavy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)

    sizes = _parse_int_list(args.L)
    orderings = [x.strip() for x in str(args.orderings).split(",") if x.strip()]
    boundaries = [x.strip() for x in str(args.boundaries).split(",") if x.strip()]

    tol_raw = _parse_tol(args.tol)

    cases: List[HHRunCase] = []
    case_counter = 0
    for L in sizes:
        for ordering in orderings:
            for boundary in boundaries:
                case_counter += 1
                case_id = f"L{int(L)}_{ordering}_{boundary}_dv{str(float(args.dv)).replace('.', 'p')}"
                cases.append(
                    HHRunCase(
                        case_id=case_id,
                        L=int(L),
                        ordering=str(ordering),
                        boundary=str(boundary),
                        J=float(args.J),
                        U=float(args.U),
                        dv=float(args.dv),
                        omega0=float(args.omega0),
                        g=float(args.g),
                        n_ph_max=int(args.n_ph_max),
                        boson_encoding=str(args.boson_encoding),
                        reps=int(args.vqe_reps),
                        restarts=int(args.vqe_restarts),
                        maxiter=int(args.vqe_maxiter),
                        seed=int(args.seed + 97 * case_counter),
                    )
                )

    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] {case.case_id}: "
            f"L={case.L}, ordering={case.ordering}, boundary={case.boundary}, "
            f"J={case.J}, U={case.U}, dv={case.dv}, g={case.g}, omega0={case.omega0}, "
            f"reps={case.reps}, restarts={case.restarts}, maxiter={case.maxiter}"
        )
        row = _run_case(case, tol_factor=1e-3 if tol_raw < 0 else 0.0)
        if tol_raw >= 0:
            row["tol"] = float(tol_raw)
            row["pass"] = bool(float(row["abs_err"]) <= float(tol_raw))
        rows.append(row)
        print(
            f"  E_exact={row['exact_energy_filtered']:.12f} "
            f"E_vqe={row['vqe_energy']:.12f} "
            f"|err|={row['abs_err']:.3e} "
            f"success={row['vqe']['success']} runtime={row['vqe']['runtime_s']:.2f}s"
        )

    rows_sorted = sorted(rows, key=lambda r: (int(r["case"]["L"]), str(r["case"]["ordering"]), str(r["case"]["boundary"])))
    pass_count = sum(1 for r in rows_sorted if bool(r["pass"]))
    max_abs_err = float(max(float(r["abs_err"]) for r in rows_sorted)) if rows_sorted else 0.0
    summary = {
        "num_cases": int(len(rows_sorted)),
        "pass_count": int(pass_count),
        "fail_count": int(len(rows_sorted) - pass_count),
        "max_abs_err": float(max_abs_err),
        "settings": {
            "L": str(args.L),
            "orderings": str(args.orderings),
            "boundaries": str(args.boundaries),
            "J": float(args.J),
            "U": float(args.U),
            "dv": float(args.dv),
            "g": float(args.g),
            "omega0": float(args.omega0),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "vqe_reps": int(args.vqe_reps),
            "vqe_restarts": int(args.vqe_restarts),
            "vqe_maxiter": int(args.vqe_maxiter),
            "seed": int(args.seed),
            "tol": "auto(|E_exact|*1e-3)" if tol_raw < 0 else float(tol_raw),
        },
    }

    json_path = out_dir / f"{prefix}_results.json"
    csv_path = out_dir / f"{prefix}_summary.csv"
    pdf_path = out_dir / f"{prefix}_report.pdf"

    payload = {
        "summary": summary,
        "cases": rows_sorted,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows_sorted)

    title = "HH Term-wise VQE vs Exact (Hubbard limit: g=0, omega0=0)"
    with PdfPages(str(pdf_path)) as pdf:
        _table_page(pdf, rows_sorted, title=title + " | Full Parameters")
        _plot_pages(pdf, rows_sorted)

    print("\nFinished.")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote PDF:  {pdf_path}")
    print(
        f"Pass count: {summary['pass_count']}/{summary['num_cases']} | "
        f"max |err| = {summary['max_abs_err']:.6e}"
    )


if __name__ == "__main__":
    main()
