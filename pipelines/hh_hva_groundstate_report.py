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
from datetime import datetime, timezone
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

from pipelines.hh_termwise_zero_holstein_report import HubbardHolsteinTermwiseAnsatz
from pydephasing.quantum.ed_hubbard_holstein import (
    build_hh_sector_basis,
    build_hh_sector_hamiltonian_ed,
    matrix_to_dense,
)
from pydephasing.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from pydephasing.quantum.vqe_latex_python_pairs_test import (
    HubbardHolsteinLayerwiseAnsatz,
    exact_ground_energy_sector,
    hubbard_holstein_reference_state,
    vqe_minimize,
)


@dataclass(frozen=True)
class DepthConfig:
    reps: int
    restarts: int
    maxiter: int


@dataclass(frozen=True)
class HHCase:
    case_id: str
    section: str
    L: int
    boundary: str
    ordering: str
    J: float
    U: float
    dv: float
    g: float
    omega0: float
    n_ph_max: int
    boson_encoding: str
    include_zero_point: bool


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for tok in str(raw).split(","):
        token = tok.strip()
        if token:
            out.append(int(token))
    return out


def _parse_boundary_orderings(raw: str) -> List[Tuple[str, str]]:
    token = str(raw).strip()
    if not token:
        return []
    out: List[Tuple[str, str]] = []
    for part in token.split(","):
        cur = part.strip()
        if not cur:
            continue
        if ":" not in cur:
            raise ValueError(f"invalid boundary:ordering token '{cur}'")
        boundary, ordering = [x.strip().lower() for x in cur.split(":", 1)]
        if boundary not in {"periodic", "open"}:
            raise ValueError(f"unsupported boundary '{boundary}'")
        if ordering not in {"blocked", "interleaved"}:
            raise ValueError(f"unsupported ordering '{ordering}'")
        pair = (boundary, ordering)
        if pair not in out:
            out.append(pair)
    return out


def _half_filling(L: int) -> Tuple[int, int]:
    return ((int(L) + 1) // 2, int(L) // 2)


def _drive_terms_from_dv(dv: float) -> Tuple[Optional[float], Optional[float]]:
    if abs(float(dv)) <= 1e-15:
        return None, None
    # HH drive convention: H_drive = (v_t - v0) * n ; we want Hv = -dv * n.
    return 0.0, float(dv)


def _fmt_float_token(x: float) -> str:
    s = f"{float(x):.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _make_case_id(L: int, boundary: str, ordering: str, g: float, omega0: float) -> str:
    return f"L{int(L)}_{boundary}_{ordering}_g{_fmt_float_token(g)}_w{_fmt_float_token(omega0)}"


def _ham_term_count(h_poly: Any, tol: float = 1e-12) -> int:
    cnt = 0
    for term in h_poly.return_polynomial():
        if abs(complex(term.p_coeff)) > float(tol):
            cnt += 1
    return int(cnt)


def _ham_num_qubits(h_poly: Any) -> int:
    terms = list(h_poly.return_polynomial())
    if not terms:
        return 0
    return int(terms[0].nqubit())


def _build_hh_hamiltonian(case: HHCase) -> Any:
    v_t, v0 = _drive_terms_from_dv(case.dv)
    return build_hubbard_holstein_hamiltonian(
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
        include_zero_point=bool(case.include_zero_point),
    )


def _exact_hh_filtered_energy(case: HHCase, num_particles: Tuple[int, int]) -> float:
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
        include_zero_point=bool(case.include_zero_point),
        basis=basis,
        sparse=False,
        return_basis=False,
    )
    dense = matrix_to_dense(h_ed)
    evals = np.linalg.eigvalsh(dense)
    return float(np.real(evals[0]))


def _exact_hubbard_embedded_energy(case: HHCase, num_particles: Tuple[int, int]) -> float:
    qpb = int(boson_qubits_per_site(int(case.n_ph_max), str(case.boson_encoding)))
    nq_total = int(2 * int(case.L) + int(case.L) * qpb)

    h_hubbard_emb = build_hubbard_hamiltonian(
        dims=int(case.L),
        t=float(case.J),
        U=float(case.U),
        v=float(case.dv),
        repr_mode="JW",
        indexing=str(case.ordering),
        pbc=(str(case.boundary) == "periodic"),
        nq_override=nq_total,
    )
    return float(
        exact_ground_energy_sector(
            h_hubbard_emb,
            num_sites=int(case.L),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            indexing=str(case.ordering),
        )
    )


def _stage_seed(base_seed: int, case_idx: int, ansatz_name: str, stage: str) -> int:
    ansatz_offset = 0 if str(ansatz_name) == "termwise" else 250_000
    stage_offset = 0 if str(stage) == "baseline" else 90_000
    return int(base_seed + case_idx * 10_000 + ansatz_offset + stage_offset)


def _run_single_vqe_stage(
    *,
    case: HHCase,
    h_poly: Any,
    exact_energy: float,
    num_particles: Tuple[int, int],
    ansatz_name: str,
    stage_name: str,
    depth: DepthConfig,
    seed: int,
    tol: float,
) -> Dict[str, Any]:
    v_t, v0 = _drive_terms_from_dv(case.dv)
    stage: Dict[str, Any] = {
        "ran": True,
        "stage": str(stage_name),
        "ansatz": str(ansatz_name),
        "reps": int(depth.reps),
        "restarts": int(depth.restarts),
        "maxiter": int(depth.maxiter),
        "seed": int(seed),
        "tol": float(tol),
    }

    try:
        if str(ansatz_name) == "termwise":
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
                reps=int(depth.reps),
                repr_mode="JW",
                indexing=str(case.ordering),
                pbc=(str(case.boundary) == "periodic"),
                include_zero_point=bool(case.include_zero_point),
            )
        elif str(ansatz_name) == "layerwise":
            ansatz = HubbardHolsteinLayerwiseAnsatz(
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
                reps=int(depth.reps),
                repr_mode="JW",
                indexing=str(case.ordering),
                pbc=(str(case.boundary) == "periodic"),
                include_zero_point=bool(case.include_zero_point),
            )
        else:
            raise ValueError(f"unsupported ansatz '{ansatz_name}'")

        psi_ref = hubbard_holstein_reference_state(
            dims=int(case.L),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            n_ph_max=int(case.n_ph_max),
            boson_encoding=str(case.boson_encoding),
            indexing=str(case.ordering),
        )

        stage["num_params"] = int(ansatz.num_parameters)
        stage["n_qubits"] = int(getattr(ansatz, "nq", int(round(math.log2(int(psi_ref.size))))))

        t0 = time.perf_counter()
        res = vqe_minimize(
            h_poly,
            ansatz,
            np.asarray(psi_ref, dtype=complex),
            restarts=int(depth.restarts),
            seed=int(seed),
            maxiter=int(depth.maxiter),
            method="SLSQP",
        )
        runtime_s = float(time.perf_counter() - t0)

        energy = float(res.energy)
        abs_err = float(abs(float(energy) - float(exact_energy)))
        passed = bool(bool(res.success) and abs_err <= float(tol))

        stage.update(
            {
                "success": bool(res.success),
                "message": str(res.message),
                "energy": float(energy),
                "abs_err": float(abs_err),
                "pass": bool(passed),
                "nfev": int(getattr(res, "nfev", 0)),
                "nit": int(getattr(res, "nit", 0)),
                "best_restart": int(getattr(res, "best_restart", 0)),
                "runtime_s": float(runtime_s),
            }
        )
    except Exception as exc:
        stage.update(
            {
                "success": False,
                "message": str(exc),
                "energy": None,
                "abs_err": None,
                "pass": False,
                "nfev": 0,
                "nit": 0,
                "best_restart": -1,
                "runtime_s": 0.0,
                "num_params": None,
                "n_qubits": None,
            }
        )
    return stage


def _skipped_stage(*, ansatz_name: str, stage_name: str, depth: DepthConfig, seed: int, reason: str) -> Dict[str, Any]:
    return {
        "ran": False,
        "stage": str(stage_name),
        "ansatz": str(ansatz_name),
        "reps": int(depth.reps),
        "restarts": int(depth.restarts),
        "maxiter": int(depth.maxiter),
        "seed": int(seed),
        "success": False,
        "message": str(reason),
        "energy": None,
        "abs_err": None,
        "pass": False,
        "nfev": 0,
        "nit": 0,
        "best_restart": -1,
        "runtime_s": 0.0,
        "num_params": None,
        "n_qubits": None,
    }


def _is_finite_number(x: Any) -> bool:
    return x is not None and np.isfinite(float(x))


def _select_best_stage(baseline: Dict[str, Any], heavy: Dict[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = [baseline]
    if bool(heavy.get("ran", False)):
        candidates.append(heavy)

    finite = [c for c in candidates if _is_finite_number(c.get("abs_err"))]
    if finite:
        best = min(finite, key=lambda c: float(c["abs_err"]))
    else:
        best = baseline

    return {
        "stage": str(best.get("stage", "baseline")),
        "energy": best.get("energy"),
        "abs_err": best.get("abs_err"),
        "pass": bool(best.get("pass", False)),
        "success": bool(best.get("success", False)),
        "runtime_s": best.get("runtime_s"),
        "num_params": best.get("num_params"),
    }


def _should_run_heavy(
    case: HHCase,
    *,
    baseline_pass: bool,
    baseline_abs_err: Any,
    heavy_skip_err_threshold: float,
) -> Tuple[bool, str]:
    if float(heavy_skip_err_threshold) > 0.0 and _is_finite_number(baseline_abs_err):
        baseline_err_f = float(baseline_abs_err)
        if baseline_err_f < float(heavy_skip_err_threshold):
            return (
                False,
                f"baseline abs_err={baseline_err_f:.3e} < heavy-skip threshold={float(heavy_skip_err_threshold):.3e}",
            )

    if str(case.ordering) == "blocked":
        return True, "blocked ordering requires heavy depth run"
    if str(case.boundary) == "periodic" and str(case.ordering) == "interleaved":
        if bool(baseline_pass):
            return False, "periodic-interleaved baseline already converged within tolerance"
        return True, "periodic-interleaved baseline not converged; running heavy"
    return False, "heavy policy disabled for this boundary/ordering"


def _run_ansatz_with_depth_policy(
    *,
    case: HHCase,
    case_idx: int,
    h_poly: Any,
    exact_energy: float,
    num_particles: Tuple[int, int],
    ansatz_name: str,
    baseline_depth: DepthConfig,
    heavy_depth: DepthConfig,
    base_seed: int,
    tol: float,
    heavy_skip_err_threshold: float,
) -> Dict[str, Any]:
    seed_baseline = _stage_seed(base_seed, case_idx, ansatz_name, "baseline")
    baseline = _run_single_vqe_stage(
        case=case,
        h_poly=h_poly,
        exact_energy=float(exact_energy),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        ansatz_name=str(ansatz_name),
        stage_name="baseline",
        depth=baseline_depth,
        seed=seed_baseline,
        tol=float(tol),
    )

    run_heavy, heavy_reason = _should_run_heavy(
        case,
        baseline_pass=bool(baseline.get("pass", False)),
        baseline_abs_err=baseline.get("abs_err"),
        heavy_skip_err_threshold=float(heavy_skip_err_threshold),
    )
    seed_heavy = _stage_seed(base_seed, case_idx, ansatz_name, "heavy")

    if run_heavy:
        heavy = _run_single_vqe_stage(
            case=case,
            h_poly=h_poly,
            exact_energy=float(exact_energy),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            ansatz_name=str(ansatz_name),
            stage_name="heavy",
            depth=heavy_depth,
            seed=seed_heavy,
            tol=float(tol),
        )
    else:
        heavy = _skipped_stage(
            ansatz_name=str(ansatz_name),
            stage_name="heavy",
            depth=heavy_depth,
            seed=seed_heavy,
            reason=heavy_reason,
        )

    best = _select_best_stage(baseline, heavy)

    improvement: Optional[float] = None
    if bool(heavy.get("ran", False)) and _is_finite_number(baseline.get("abs_err")) and _is_finite_number(heavy.get("abs_err")):
        improvement = float(float(baseline["abs_err"]) - float(heavy["abs_err"]))

    expressivity_suspected = bool(
        bool(heavy.get("ran", False)) and bool(heavy.get("success", False)) and (not bool(heavy.get("pass", False)))
    )

    return {
        "baseline": baseline,
        "heavy": heavy,
        "best": best,
        "heavy_decision_reason": str(heavy_reason),
        "depth_improvement": improvement,
        "expressivity_suspected": bool(expressivity_suspected),
    }


def _safe_float(x: Any, default: float = float("nan")) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _case_label_short(case_row: Dict[str, Any]) -> str:
    m = case_row["case_meta"]
    return f"L{m['L']}|{m['boundary'][0]}|{m['ordering'][0]}|g={float(m['g']):.2g}|w={float(m['omega0']):.2g}"


def _ansatz_csv_prefix(ansatz: str) -> str:
    return str(ansatz).lower().strip()


def _write_csv(path: Path, case_rows: Sequence[Dict[str, Any]]) -> None:
    cols = [
        "case_id",
        "section",
        "L",
        "boundary",
        "ordering",
        "J",
        "U",
        "dv",
        "g",
        "omega0",
        "n_ph_max",
        "boson_encoding",
        "n_up",
        "n_dn",
        "n_qubits",
        "ham_terms",
        "tol",
        "E_exact_hh",
        "E_exact_hubbard_embedded",
        "delta_exact",
        "zero_limit_consistency_pass",
        "termwise_baseline_energy",
        "termwise_baseline_abs_err",
        "termwise_baseline_pass",
        "termwise_heavy_ran",
        "termwise_heavy_energy",
        "termwise_heavy_abs_err",
        "termwise_heavy_pass",
        "termwise_best_stage",
        "termwise_best_energy",
        "termwise_best_abs_err",
        "termwise_best_pass",
        "termwise_depth_improvement",
        "termwise_expressivity_suspected",
        "layerwise_baseline_energy",
        "layerwise_baseline_abs_err",
        "layerwise_baseline_pass",
        "layerwise_heavy_ran",
        "layerwise_heavy_energy",
        "layerwise_heavy_abs_err",
        "layerwise_heavy_pass",
        "layerwise_best_stage",
        "layerwise_best_energy",
        "layerwise_best_abs_err",
        "layerwise_best_pass",
        "layerwise_depth_improvement",
        "layerwise_expressivity_suspected",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for row in case_rows:
            meta = row["case_meta"]
            exact = row["exact"]
            ham = row["hamiltonian"]
            num_particles = meta["num_particles"]

            out: Dict[str, Any] = {
                "case_id": meta["case_id"],
                "section": meta["section"],
                "L": meta["L"],
                "boundary": meta["boundary"],
                "ordering": meta["ordering"],
                "J": meta["J"],
                "U": meta["U"],
                "dv": meta["dv"],
                "g": meta["g"],
                "omega0": meta["omega0"],
                "n_ph_max": meta["n_ph_max"],
                "boson_encoding": meta["boson_encoding"],
                "n_up": num_particles["n_up"],
                "n_dn": num_particles["n_dn"],
                "n_qubits": ham["num_qubits"],
                "ham_terms": ham["num_terms"],
                "tol": exact["tol"],
                "E_exact_hh": exact["hh_filtered"],
                "E_exact_hubbard_embedded": exact.get("hubbard_embedded_filtered"),
                "delta_exact": exact.get("delta_exact"),
                "zero_limit_consistency_pass": exact.get("zero_limit_consistency_pass"),
            }

            for ansatz in ("termwise", "layerwise"):
                pref = _ansatz_csv_prefix(ansatz)
                block = row["results"][ansatz]
                base = block["baseline"]
                heavy = block["heavy"]
                best = block["best"]

                out[f"{pref}_baseline_energy"] = base.get("energy")
                out[f"{pref}_baseline_abs_err"] = base.get("abs_err")
                out[f"{pref}_baseline_pass"] = base.get("pass")
                out[f"{pref}_heavy_ran"] = heavy.get("ran")
                out[f"{pref}_heavy_energy"] = heavy.get("energy")
                out[f"{pref}_heavy_abs_err"] = heavy.get("abs_err")
                out[f"{pref}_heavy_pass"] = heavy.get("pass")
                out[f"{pref}_best_stage"] = best.get("stage")
                out[f"{pref}_best_energy"] = best.get("energy")
                out[f"{pref}_best_abs_err"] = best.get("abs_err")
                out[f"{pref}_best_pass"] = best.get("pass")
                out[f"{pref}_depth_improvement"] = block.get("depth_improvement")
                out[f"{pref}_expressivity_suspected"] = block.get("expressivity_suspected")

            writer.writerow(out)


def _render_text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(13.5, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(str(title), fontsize=16, pad=16)
    ax.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        linespacing=1.45,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_pages(
    pdf: PdfPages,
    *,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    rows_per_page: int = 16,
    fontsize: float = 8.2,
) -> None:
    if not rows:
        _render_text_page(pdf, title, ["No rows to display."])
        return

    for page_idx in range(0, len(rows), int(rows_per_page)):
        chunk = rows[page_idx : page_idx + int(rows_per_page)]
        fig = plt.figure(figsize=(18, 10))
        ax = fig.add_subplot(111)
        ax.axis("off")
        page_no = page_idx // int(rows_per_page) + 1
        page_title = f"{title} (page {page_no})" if len(rows) > int(rows_per_page) else title
        ax.set_title(page_title, fontsize=14, pad=12)

        table = ax.table(
            cellText=chunk,
            colLabels=list(headers),
            cellLoc="center",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(float(fontsize))
        table.scale(1.0, 1.25)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _main_comparison_rows(main_rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "Case",
        "L",
        "Boundary",
        "Ordering",
        "g",
        "omega0",
        "E_exact",
        "Term best",
        "E_term",
        "|err|_term",
        "pass_t",
        "Layer best",
        "E_layer",
        "|err|_layer",
        "pass_l",
    ]
    rows: List[List[str]] = []
    for row in main_rows:
        meta = row["case_meta"]
        exact = row["exact"]
        tbest = row["results"]["termwise"]["best"]
        lbest = row["results"]["layerwise"]["best"]
        rows.append(
            [
                str(meta["case_id"]),
                str(meta["L"]),
                str(meta["boundary"]),
                str(meta["ordering"]),
                f"{float(meta['g']):.3g}",
                f"{float(meta['omega0']):.3g}",
                f"{float(exact['hh_filtered']): .12f}",
                str(tbest.get("stage", "-")),
                f"{_safe_float(tbest.get('energy')): .12f}" if _is_finite_number(tbest.get("energy")) else "nan",
                f"{_safe_float(tbest.get('abs_err')):.3e}" if _is_finite_number(tbest.get("abs_err")) else "nan",
                str(bool(tbest.get("pass", False))),
                str(lbest.get("stage", "-")),
                f"{_safe_float(lbest.get('energy')): .12f}" if _is_finite_number(lbest.get("energy")) else "nan",
                f"{_safe_float(lbest.get('abs_err')):.3e}" if _is_finite_number(lbest.get("abs_err")) else "nan",
                str(bool(lbest.get("pass", False))),
            ]
        )
    return headers, rows


def _appendix_detail_rows(case_rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "Case",
        "Ansatz",
        "E_exact",
        "E_base",
        "|err|_base",
        "pass_b",
        "E_heavy",
        "|err|_heavy",
        "pass_h",
        "heavy?",
        "best",
        "|err|_best",
        "pass_best",
        "improve",
        "expr?",
    ]

    out_rows: List[List[str]] = []
    for row in case_rows:
        exact_e = float(row["exact"]["hh_filtered"])
        for ansatz in ("termwise", "layerwise"):
            block = row["results"][ansatz]
            base = block["baseline"]
            heavy = block["heavy"]
            best = block["best"]
            improve = block.get("depth_improvement")
            out_rows.append(
                [
                    str(row["case_meta"]["case_id"]),
                    str(ansatz),
                    f"{exact_e: .10f}",
                    f"{_safe_float(base.get('energy')): .10f}" if _is_finite_number(base.get("energy")) else "nan",
                    f"{_safe_float(base.get('abs_err')):.3e}" if _is_finite_number(base.get("abs_err")) else "nan",
                    str(bool(base.get("pass", False))),
                    f"{_safe_float(heavy.get('energy')): .10f}" if _is_finite_number(heavy.get("energy")) else "nan",
                    f"{_safe_float(heavy.get('abs_err')):.3e}" if _is_finite_number(heavy.get("abs_err")) else "nan",
                    str(bool(heavy.get("pass", False))),
                    str(bool(heavy.get("ran", False))),
                    str(best.get("stage", "-")),
                    f"{_safe_float(best.get('abs_err')):.3e}" if _is_finite_number(best.get("abs_err")) else "nan",
                    str(bool(best.get("pass", False))),
                    f"{float(improve):+.3e}" if improve is not None and np.isfinite(float(improve)) else "nan",
                    str(bool(block.get("expressivity_suspected", False))),
                ]
            )
    return headers, out_rows


def _to_lines_summary(summary: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append("HH HVA Ground-State Report")
    lines.append("")
    lines.append(f"generated_utc: {summary['generated_utc']}")
    lines.append(f"num_cases: {summary['num_cases']} (main={summary['num_main_cases']}, appendix={summary['num_appendix_cases']})")
    lines.append(f"tol_rule: tol = max({config['tol_floor']}, {config['tol_scale']} * |E_exact|)")
    lines.append(f"heavy_skip_rule: skip heavy when baseline abs_err < {config['heavy_skip_err_threshold']}")
    lines.append(f"overall_max_abs_err_best: {summary['overall_max_abs_err_best']:.6e}")
    lines.append("")

    for ansatz in ("termwise", "layerwise"):
        block = summary["ansatz_summary"][ansatz]
        lines.append(f"[{ansatz}] baseline_pass={block['baseline_pass_count']}/{summary['num_cases']}")
        lines.append(f"[{ansatz}] heavy_ran={block['heavy_ran_count']}/{summary['num_cases']}")
        lines.append(f"[{ansatz}] heavy_pass={block['heavy_pass_count']}/{max(1, block['heavy_ran_count'])}")
        lines.append(f"[{ansatz}] best_pass={block['best_pass_count']}/{summary['num_cases']}")
        lines.append(f"[{ansatz}] heavy_improved_count={block['heavy_improved_count']}")
        lines.append(f"[{ansatz}] expressivity_suspected_count={block['expressivity_suspected_count']}")
        lines.append(f"[{ansatz}] max_abs_err_best={block['max_abs_err_best']:.6e}")
        lines.append("")

    zl = summary["zero_limit_consistency"]
    lines.append("[zero-limit HH vs embedded Hubbard exact check]")
    lines.append(f"cases={zl['num_cases']} pass={zl['pass_count']} fail={zl['fail_count']}")
    lines.append(f"max_delta_exact={zl['max_delta_exact']:.6e} (threshold={zl['threshold']:.1e})")
    return lines


def _plot_energy_comparison(pdf: PdfPages, main_rows: Sequence[Dict[str, Any]]) -> None:
    if not main_rows:
        _render_text_page(pdf, "Energy Comparison", ["No main rows to plot."])
        return

    labels = [_case_label_short(r) for r in main_rows]
    x = np.arange(len(labels), dtype=float)

    e_exact = np.array([float(r["exact"]["hh_filtered"]) for r in main_rows], dtype=float)
    e_term = np.array([_safe_float(r["results"]["termwise"]["best"].get("energy")) for r in main_rows], dtype=float)
    e_layer = np.array([_safe_float(r["results"]["layerwise"]["best"].get("energy")) for r in main_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    width = 0.34
    ax.bar(x - width / 2.0, e_term, width=width, color="#0072B2", alpha=0.90, label="Term-wise (best)")
    ax.bar(x + width / 2.0, e_layer, width=width, color="#D55E00", alpha=0.90, label="Layer-wise (best)")
    ax.plot(x, e_exact, color="#111111", marker="o", linewidth=2.2, label="Exact (HH filtered)")

    ax.set_title("Main Cases: Energy Comparison", fontsize=14)
    ax.set_ylabel("Energy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_abs_error_log(pdf: PdfPages, main_rows: Sequence[Dict[str, Any]]) -> None:
    if not main_rows:
        _render_text_page(pdf, "Absolute Error (log)", ["No main rows to plot."])
        return

    labels = [_case_label_short(r) for r in main_rows]
    x = np.arange(len(labels), dtype=float)
    err_term = np.array([
        max(1e-16, _safe_float(r["results"]["termwise"]["best"].get("abs_err"), default=1e6)) for r in main_rows
    ])
    err_layer = np.array([
        max(1e-16, _safe_float(r["results"]["layerwise"]["best"].get("abs_err"), default=1e6)) for r in main_rows
    ])

    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    width = 0.34
    ax.bar(x - width / 2.0, err_term, width=width, color="#0072B2", alpha=0.95, label="Term-wise |err|")
    ax.bar(x + width / 2.0, err_layer, width=width, color="#D55E00", alpha=0.95, label="Layer-wise |err|")
    ax.set_yscale("log")
    ax.set_title("Main Cases: Absolute Error vs Exact (log scale)", fontsize=14)
    ax.set_ylabel("|E_vqe - E_exact| (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", which="both", alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_depth_improvement(pdf: PdfPages, case_rows: Sequence[Dict[str, Any]]) -> None:
    if not case_rows:
        _render_text_page(pdf, "Depth Improvement", ["No rows to plot."])
        return

    labels = [_case_label_short(r) for r in case_rows]
    x = np.arange(len(labels), dtype=float)

    term_impr = np.array([
        _safe_float(r["results"]["termwise"].get("depth_improvement"), default=float("nan")) for r in case_rows
    ])
    layer_impr = np.array([
        _safe_float(r["results"]["layerwise"].get("depth_improvement"), default=float("nan")) for r in case_rows
    ])

    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    width = 0.35
    ax.bar(x - width / 2.0, term_impr, width=width, color="#009E73", alpha=0.90, label="Term-wise improvement")
    ax.bar(x + width / 2.0, layer_impr, width=width, color="#CC79A7", alpha=0.90, label="Layer-wise improvement")
    ax.axhline(0.0, color="#111111", linewidth=1.2)

    ax.set_title("Depth Effect: baseline |err| - heavy |err|", fontsize=14)
    ax.set_ylabel("Positive values mean heavy depth improved convergence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _zero_limit_table_and_plot(pdf: PdfPages, zero_rows: Sequence[Dict[str, Any]], threshold: float) -> None:
    headers = ["Case", "Boundary", "Ordering", "L", "E_exact_HH", "E_exact_Hubbard_emb", "delta_exact", "pass"]
    rows: List[List[str]] = []
    for r in zero_rows:
        meta = r["case_meta"]
        ex = r["exact"]
        rows.append(
            [
                str(meta["case_id"]),
                str(meta["boundary"]),
                str(meta["ordering"]),
                str(meta["L"]),
                f"{float(ex['hh_filtered']): .12f}",
                f"{_safe_float(ex.get('hubbard_embedded_filtered')): .12f}" if _is_finite_number(ex.get("hubbard_embedded_filtered")) else "nan",
                f"{_safe_float(ex.get('delta_exact')):.3e}" if _is_finite_number(ex.get("delta_exact")) else "nan",
                str(bool(ex.get("zero_limit_consistency_pass", False))),
            ]
        )
    _table_pages(
        pdf,
        title="Zero-limit Consistency: HH(g=0,omega0=0) vs Embedded Hubbard exact",
        headers=headers,
        rows=rows,
        rows_per_page=14,
        fontsize=8.5,
    )

    if not zero_rows:
        return

    labels = [_case_label_short(r) for r in zero_rows]
    deltas = np.array([max(1e-16, _safe_float(r["exact"].get("delta_exact"), default=1e6)) for r in zero_rows], dtype=float)
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    ax.bar(x, deltas, color="#56B4E9", alpha=0.92)
    ax.axhline(float(threshold), color="#111111", linestyle="--", linewidth=1.5, label=f"threshold = {threshold:.1e}")
    ax.set_yscale("log")
    ax.set_title("Zero-limit exact-energy consistency (log scale)", fontsize=14)
    ax.set_ylabel("|E_exact_HH - E_exact_Hubbard_embedded|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", which="both", alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _write_pdf(
    path: Path,
    *,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    case_rows: Sequence[Dict[str, Any]],
) -> None:
    main_rows = [r for r in case_rows if str(r["case_meta"]["section"]) == "main"]
    appendix_open_blocked = [
        r
        for r in case_rows
        if str(r["case_meta"]["boundary"]) == "open" and str(r["case_meta"]["ordering"]) == "blocked"
    ]
    appendix_periodic_interleaved = [
        r
        for r in case_rows
        if str(r["case_meta"]["boundary"]) == "periodic" and str(r["case_meta"]["ordering"]) == "interleaved"
    ]
    zero_rows = [
        r
        for r in case_rows
        if abs(float(r["case_meta"]["g"])) <= 1e-15 and abs(float(r["case_meta"]["omega0"])) <= 1e-15
    ]

    with PdfPages(str(path)) as pdf:
        _render_text_page(pdf, "Executive Summary", _to_lines_summary(summary, config))

        headers, rows = _main_comparison_rows(main_rows)
        _table_pages(
            pdf,
            title="Main Cases: Best-per-ansatz Ground-State Results",
            headers=headers,
            rows=rows,
            rows_per_page=14,
            fontsize=8.4,
        )

        _plot_energy_comparison(pdf, main_rows)
        _plot_abs_error_log(pdf, main_rows)
        _plot_depth_improvement(pdf, case_rows)
        _zero_limit_table_and_plot(pdf, zero_rows, threshold=float(summary["zero_limit_consistency"]["threshold"]))

        headers_a, rows_a = _appendix_detail_rows(appendix_open_blocked)
        _table_pages(
            pdf,
            title="Appendix A: Open + Blocked full stage details",
            headers=headers_a,
            rows=rows_a,
            rows_per_page=13,
            fontsize=8.0,
        )

        headers_b, rows_b = _appendix_detail_rows(appendix_periodic_interleaved)
        _table_pages(
            pdf,
            title="Appendix B: Periodic + Interleaved detailed rows",
            headers=headers_b,
            rows=rows_b,
            rows_per_page=13,
            fontsize=8.0,
        )


def _summarize(case_rows: Sequence[Dict[str, Any]], zero_threshold: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "num_cases": int(len(case_rows)),
        "num_main_cases": int(sum(1 for r in case_rows if str(r["case_meta"]["section"]) == "main")),
        "num_appendix_cases": int(sum(1 for r in case_rows if str(r["case_meta"]["section"]) == "appendix")),
        "ansatz_summary": {},
    }

    best_abs_all: List[float] = []

    for ansatz in ("termwise", "layerwise"):
        baseline_pass = 0
        heavy_ran = 0
        heavy_pass = 0
        best_pass = 0
        heavy_improved = 0
        expressivity = 0
        best_abs: List[float] = []

        for row in case_rows:
            block = row["results"][ansatz]
            base = block["baseline"]
            heavy = block["heavy"]
            best = block["best"]

            if bool(base.get("pass", False)):
                baseline_pass += 1

            if bool(heavy.get("ran", False)):
                heavy_ran += 1
                if bool(heavy.get("pass", False)):
                    heavy_pass += 1

            if bool(best.get("pass", False)):
                best_pass += 1

            if block.get("depth_improvement") is not None and _is_finite_number(block.get("depth_improvement")):
                if float(block["depth_improvement"]) > 0.0:
                    heavy_improved += 1

            if bool(block.get("expressivity_suspected", False)):
                expressivity += 1

            if _is_finite_number(best.get("abs_err")):
                best_abs.append(float(best["abs_err"]))
                best_abs_all.append(float(best["abs_err"]))

        out["ansatz_summary"][ansatz] = {
            "baseline_pass_count": int(baseline_pass),
            "heavy_ran_count": int(heavy_ran),
            "heavy_pass_count": int(heavy_pass),
            "best_pass_count": int(best_pass),
            "heavy_improved_count": int(heavy_improved),
            "expressivity_suspected_count": int(expressivity),
            "max_abs_err_best": float(max(best_abs) if best_abs else float("nan")),
        }

    out["overall_max_abs_err_best"] = float(max(best_abs_all) if best_abs_all else float("nan"))

    zero_rows = [
        r
        for r in case_rows
        if abs(float(r["case_meta"]["g"])) <= 1e-15 and abs(float(r["case_meta"]["omega0"])) <= 1e-15
    ]
    deltas = [float(r["exact"]["delta_exact"]) for r in zero_rows if _is_finite_number(r["exact"].get("delta_exact"))]
    pass_count = sum(1 for r in zero_rows if bool(r["exact"].get("zero_limit_consistency_pass", False)))
    out["zero_limit_consistency"] = {
        "num_cases": int(len(zero_rows)),
        "pass_count": int(pass_count),
        "fail_count": int(len(zero_rows) - pass_count),
        "max_delta_exact": float(max(deltas) if deltas else float("nan")),
        "threshold": float(zero_threshold),
    }

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HH ground-state benchmark report: term-wise + layer-wise HVA with "
            "variable optimization depth, zero-limit exact consistency checks, and PDF/CSV/JSON outputs."
        )
    )

    parser.add_argument("--l-values", type=str, default="2,3")
    parser.add_argument("--nonzero-g", type=float, default=0.15)
    parser.add_argument("--nonzero-omega0", type=float, default=0.4)
    parser.add_argument(
        "--physics-mode",
        choices=["both", "zero", "nonzero"],
        default="both",
        help="Which physics points to run: both={(0,0),(nonzero)}, zero={(0,0)}, nonzero={(nonzero)}.",
    )

    parser.add_argument("--main-boundary-orderings", type=str, default="periodic:blocked,periodic:interleaved")
    parser.add_argument("--appendix-boundary-orderings", type=str, default="open:blocked")

    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--n-ph-max", type=int, default=1)
    parser.add_argument("--boson-encoding", choices=["binary"], default="binary")
    parser.add_argument("--include-zero-point", action="store_true", default=True)

    parser.add_argument("--layerwise-baseline-reps", type=int, default=2)
    parser.add_argument("--layerwise-baseline-restarts", type=int, default=6)
    parser.add_argument("--layerwise-baseline-maxiter", type=int, default=800)

    parser.add_argument("--layerwise-heavy-reps", type=int, default=4)
    parser.add_argument("--layerwise-heavy-restarts", type=int, default=16)
    parser.add_argument("--layerwise-heavy-maxiter", type=int, default=2500)

    parser.add_argument("--termwise-baseline-reps", type=int, default=2)
    parser.add_argument("--termwise-baseline-restarts", type=int, default=10)
    parser.add_argument("--termwise-baseline-maxiter", type=int, default=1200)

    parser.add_argument("--termwise-heavy-reps", type=int, default=3)
    parser.add_argument("--termwise-heavy-restarts", type=int, default=24)
    parser.add_argument("--termwise-heavy-maxiter", type=int, default=3200)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tol-scale", type=float, default=1e-3)
    parser.add_argument("--tol-floor", type=float, default=1e-3)
    parser.add_argument(
        "--heavy-skip-err-threshold",
        type=float,
        default=1e-5,
        help="If baseline abs_err is below this threshold, heavy stage is skipped.",
    )
    parser.add_argument("--zero-limit-threshold", type=float, default=1e-10)

    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hh_hva_groundstate_report"))
    parser.add_argument("--output-prefix", type=str, default="hh_hva_groundstate")
    parser.add_argument("--skip-pdf", action="store_true")

    parser.add_argument("--enforce-zero-limit-check", dest="enforce_zero_limit_check", action="store_true")
    parser.add_argument("--no-enforce-zero-limit-check", dest="enforce_zero_limit_check", action="store_false")
    parser.set_defaults(enforce_zero_limit_check=True)

    return parser.parse_args()


def _physics_points_from_args(args: argparse.Namespace) -> List[Tuple[float, float]]:
    mode = str(args.physics_mode).strip().lower()
    if mode == "zero":
        return [(0.0, 0.0)]
    if mode == "nonzero":
        return [(float(args.nonzero_g), float(args.nonzero_omega0))]
    return [(0.0, 0.0), (float(args.nonzero_g), float(args.nonzero_omega0))]


def _build_cases(args: argparse.Namespace) -> List[HHCase]:
    l_values = _parse_int_list(args.l_values)
    physics_points = _physics_points_from_args(args)

    main_pairs = _parse_boundary_orderings(str(args.main_boundary_orderings))
    appendix_pairs = _parse_boundary_orderings(str(args.appendix_boundary_orderings))

    out: List[HHCase] = []
    seen: set[Tuple[int, str, str, float, float]] = set()

    for section, pairs in (("main", main_pairs), ("appendix", appendix_pairs)):
        for L in l_values:
            for (g, omega0) in physics_points:
                for (boundary, ordering) in pairs:
                    key = (int(L), str(boundary), str(ordering), float(g), float(omega0))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        HHCase(
                            case_id=_make_case_id(int(L), str(boundary), str(ordering), float(g), float(omega0)),
                            section=str(section),
                            L=int(L),
                            boundary=str(boundary),
                            ordering=str(ordering),
                            J=float(args.J),
                            U=float(args.U),
                            dv=float(args.dv),
                            g=float(g),
                            omega0=float(omega0),
                            n_ph_max=int(args.n_ph_max),
                            boson_encoding=str(args.boson_encoding),
                            include_zero_point=bool(args.include_zero_point),
                        )
                    )

    return out


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{str(args.output_prefix)}_results.json"
    csv_path = out_dir / f"{str(args.output_prefix)}_summary.csv"
    pdf_path = out_dir / f"{str(args.output_prefix)}_report.pdf"

    layerwise_baseline = DepthConfig(
        reps=int(args.layerwise_baseline_reps),
        restarts=int(args.layerwise_baseline_restarts),
        maxiter=int(args.layerwise_baseline_maxiter),
    )
    layerwise_heavy = DepthConfig(
        reps=int(args.layerwise_heavy_reps),
        restarts=int(args.layerwise_heavy_restarts),
        maxiter=int(args.layerwise_heavy_maxiter),
    )
    termwise_baseline = DepthConfig(
        reps=int(args.termwise_baseline_reps),
        restarts=int(args.termwise_baseline_restarts),
        maxiter=int(args.termwise_baseline_maxiter),
    )
    termwise_heavy = DepthConfig(
        reps=int(args.termwise_heavy_reps),
        restarts=int(args.termwise_heavy_restarts),
        maxiter=int(args.termwise_heavy_maxiter),
    )

    cases = _build_cases(args)
    if not cases:
        raise RuntimeError("no cases generated; check boundary-ordering arguments")

    config: Dict[str, Any] = {
        "l_values": str(args.l_values),
        "physics_points": [{"g": float(g), "omega0": float(w)} for (g, w) in _physics_points_from_args(args)],
        "physics_mode": str(args.physics_mode),
        "main_boundary_orderings": str(args.main_boundary_orderings),
        "appendix_boundary_orderings": str(args.appendix_boundary_orderings),
        "J": float(args.J),
        "U": float(args.U),
        "dv": float(args.dv),
        "n_ph_max": int(args.n_ph_max),
        "boson_encoding": str(args.boson_encoding),
        "include_zero_point": bool(args.include_zero_point),
        "layerwise": {
            "baseline": asdict(layerwise_baseline),
            "heavy": asdict(layerwise_heavy),
        },
        "termwise": {
            "baseline": asdict(termwise_baseline),
            "heavy": asdict(termwise_heavy),
        },
        "seed": int(args.seed),
        "tol_scale": float(args.tol_scale),
        "tol_floor": float(args.tol_floor),
        "heavy_skip_err_threshold": float(args.heavy_skip_err_threshold),
        "zero_limit_threshold": float(args.zero_limit_threshold),
    }

    case_rows: List[Dict[str, Any]] = []

    print(f"Running {len(cases)} HH cases...")
    for idx, case in enumerate(cases, start=1):
        num_particles = _half_filling(case.L)
        h_poly = _build_hh_hamiltonian(case)
        exact_hh = _exact_hh_filtered_energy(case, num_particles=num_particles)
        tol = float(max(float(args.tol_floor), float(args.tol_scale) * abs(float(exact_hh))))

        print(
            f"[{idx}/{len(cases)}] {case.case_id} section={case.section} "
            f"L={case.L} boundary={case.boundary} ordering={case.ordering} "
            f"g={case.g} omega0={case.omega0} tol={tol:.3e}"
        )

        exact_hubbard_emb: Optional[float] = None
        delta_exact: Optional[float] = None
        zero_pass: Optional[bool] = None
        if abs(float(case.g)) <= 1e-15 and abs(float(case.omega0)) <= 1e-15:
            exact_hubbard_emb = _exact_hubbard_embedded_energy(case, num_particles=num_particles)
            delta_exact = float(abs(float(exact_hh) - float(exact_hubbard_emb)))
            zero_pass = bool(delta_exact <= float(args.zero_limit_threshold))

        termwise = _run_ansatz_with_depth_policy(
            case=case,
            case_idx=idx,
            h_poly=h_poly,
            exact_energy=float(exact_hh),
            num_particles=num_particles,
            ansatz_name="termwise",
            baseline_depth=termwise_baseline,
            heavy_depth=termwise_heavy,
            base_seed=int(args.seed),
            tol=float(tol),
            heavy_skip_err_threshold=float(args.heavy_skip_err_threshold),
        )
        layerwise = _run_ansatz_with_depth_policy(
            case=case,
            case_idx=idx,
            h_poly=h_poly,
            exact_energy=float(exact_hh),
            num_particles=num_particles,
            ansatz_name="layerwise",
            baseline_depth=layerwise_baseline,
            heavy_depth=layerwise_heavy,
            base_seed=int(args.seed),
            tol=float(tol),
            heavy_skip_err_threshold=float(args.heavy_skip_err_threshold),
        )

        case_row: Dict[str, Any] = {
            "case_meta": {
                "case_id": str(case.case_id),
                "section": str(case.section),
                "L": int(case.L),
                "boundary": str(case.boundary),
                "ordering": str(case.ordering),
                "J": float(case.J),
                "U": float(case.U),
                "dv": float(case.dv),
                "g": float(case.g),
                "omega0": float(case.omega0),
                "n_ph_max": int(case.n_ph_max),
                "boson_encoding": str(case.boson_encoding),
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
            },
            "hamiltonian": {
                "num_terms": int(_ham_term_count(h_poly)),
                "num_qubits": int(_ham_num_qubits(h_poly)),
            },
            "exact": {
                "hh_filtered": float(exact_hh),
                "hubbard_embedded_filtered": float(exact_hubbard_emb) if exact_hubbard_emb is not None else None,
                "delta_exact": float(delta_exact) if delta_exact is not None else None,
                "zero_limit_consistency_pass": bool(zero_pass) if zero_pass is not None else None,
                "tol": float(tol),
            },
            "results": {
                "termwise": termwise,
                "layerwise": layerwise,
            },
            "diagnostics": {
                "depth_improvement": {
                    "termwise": termwise.get("depth_improvement"),
                    "layerwise": layerwise.get("depth_improvement"),
                },
                "expressivity_suspected": {
                    "termwise": bool(termwise.get("expressivity_suspected", False)),
                    "layerwise": bool(layerwise.get("expressivity_suspected", False)),
                },
            },
        }
        case_rows.append(case_row)

        print(
            f"  termwise(best={termwise['best']['stage']}, |err|={_safe_float(termwise['best'].get('abs_err')):.3e}, pass={termwise['best']['pass']}) "
            f"layerwise(best={layerwise['best']['stage']}, |err|={_safe_float(layerwise['best'].get('abs_err')):.3e}, pass={layerwise['best']['pass']})"
        )
        if delta_exact is not None:
            print(f"  zero-limit delta_exact={delta_exact:.3e}, pass={zero_pass}")

    summary = _summarize(case_rows, zero_threshold=float(args.zero_limit_threshold))

    payload = {
        "config": config,
        "summary": summary,
        "cases": case_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, case_rows)

    if not bool(args.skip_pdf):
        _write_pdf(pdf_path, config=config, summary=summary, case_rows=case_rows)

    print("\nFinished HH HVA ground-state report generation.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    if not bool(args.skip_pdf):
        print(f"PDF:  {pdf_path}")

    if bool(args.enforce_zero_limit_check):
        zl = summary["zero_limit_consistency"]
        if int(zl["fail_count"]) > 0:
            raise AssertionError(
                "Zero-limit exact consistency check failed: "
                f"fail_count={zl['fail_count']}, max_delta_exact={zl['max_delta_exact']}"
            )


if __name__ == "__main__":
    main()
