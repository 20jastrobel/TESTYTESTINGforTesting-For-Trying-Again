#!/usr/bin/env python3
"""Unified energy comparison for ADAPT-VQE, VQE, and exact-filtered reference.

This script compares, per lattice size L:
- hardcoded ADAPT-VQE energy
- Qiskit ADAPT-VQE energy
- hardcoded VQE-UCCSD energy
- Qiskit VQE-UCCSD energy
- exact filtered ground-state energy (canonical reference from qiskit VQE payload)

Outputs:
- per-L metrics JSON
- overall summary JSON (`all_pass`)
- flat CSV table
- command log (if child pipelines were executed)
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
VQE_ROOT = REPO_ROOT / "Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"


THRESHOLDS_DEFAULT = {
    "exact_filtered_cross_source_abs_delta": 1e-8,
    "adapt_hc_vs_qk_abs_delta": 1.0,
    "adapt_hc_vs_exact_filtered_abs_delta": 1.0,
    "adapt_qk_vs_exact_filtered_abs_delta": 1.0,
    "vqe_hc_vs_qk_abs_delta": 1.0,
    "vqe_hc_vs_exact_filtered_abs_delta": 1.0,
    "vqe_qk_vs_exact_filtered_abs_delta": 1.0,
}


@dataclass
class RunArtifacts:
    L: int
    hardcoded_adapt_json: Path
    qiskit_adapt_json: Path
    hardcoded_vqe_json: Path
    qiskit_vqe_json: Path
    per_l_metrics_json: Path


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _run_command(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    t0 = time.perf_counter()
    _ai_log("subprocess_start", cmd=cmd, cwd=str(cwd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    _ai_log(
        "subprocess_done",
        cmd=cmd,
        cwd=str(cwd),
        returncode=int(proc.returncode),
        elapsed_sec=round(time.perf_counter() - t0, 6),
        stdout_lines=int(len(proc.stdout.splitlines())),
        stderr_lines=int(len(proc.stderr.splitlines())),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _current_command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _require_path(path: Path, *, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    _require_path(path, what="JSON artifact")
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing key path: {'.'.join(keys)}")
        cur = cur[key]
    return cur


def _require_finite_float(value: Any, *, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required value: {name}")
    val = float(value)
    if not np.isfinite(val):
        raise ValueError(f"Non-finite value for {name}: {val!r}")
    return val


def _safe_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    val = float(value)
    if not np.isfinite(val):
        return None
    return val


def _abs_delta(a: float, b: float) -> float:
    return float(abs(float(a) - float(b)))


def _energy_metrics_for_l(
    *,
    L: int,
    hardcoded_adapt: dict[str, Any],
    qiskit_adapt: dict[str, Any],
    hardcoded_vqe: dict[str, Any],
    qiskit_vqe: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    e_adapt_hc = _require_finite_float(
        _nested_get(hardcoded_adapt, "adapt_vqe", "energy"),
        name=f"L={L} hardcoded_adapt.adapt_vqe.energy",
    )
    e_adapt_qk = _require_finite_float(
        _nested_get(qiskit_adapt, "adapt_vqe", "energy"),
        name=f"L={L} qiskit_adapt.adapt_vqe.energy",
    )
    e_vqe_hc = _require_finite_float(
        _nested_get(hardcoded_vqe, "vqe", "energy"),
        name=f"L={L} hardcoded_vqe.vqe.energy",
    )
    e_vqe_qk = _require_finite_float(
        _nested_get(qiskit_vqe, "vqe", "energy"),
        name=f"L={L} qiskit_vqe.vqe.energy",
    )

    e_exact_filtered_vqe = _require_finite_float(
        _nested_get(qiskit_vqe, "vqe", "exact_filtered_energy"),
        name=f"L={L} qiskit_vqe.vqe.exact_filtered_energy",
    )
    e_exact_filtered_adapt = _safe_optional_float(
        _nested_get(qiskit_adapt, "adapt_vqe", "exact_filtered_energy")
    )

    # Canonical exact-filtered reference for this report.
    e_exact_filtered_ref = e_exact_filtered_vqe

    exact_cross_delta = (
        _abs_delta(e_exact_filtered_vqe, e_exact_filtered_adapt)
        if e_exact_filtered_adapt is not None
        else None
    )

    deltas = {
        "adapt_hc_vs_qk_abs_delta": _abs_delta(e_adapt_hc, e_adapt_qk),
        "adapt_hc_vs_exact_filtered_abs_delta": _abs_delta(e_adapt_hc, e_exact_filtered_ref),
        "adapt_qk_vs_exact_filtered_abs_delta": _abs_delta(e_adapt_qk, e_exact_filtered_ref),
        "adapt_hc_vs_vqe_hc_abs_delta": _abs_delta(e_adapt_hc, e_vqe_hc),
        "adapt_hc_vs_vqe_qk_abs_delta": _abs_delta(e_adapt_hc, e_vqe_qk),
        "adapt_qk_vs_vqe_hc_abs_delta": _abs_delta(e_adapt_qk, e_vqe_hc),
        "adapt_qk_vs_vqe_qk_abs_delta": _abs_delta(e_adapt_qk, e_vqe_qk),
        "vqe_hc_vs_qk_abs_delta": _abs_delta(e_vqe_hc, e_vqe_qk),
        "vqe_hc_vs_exact_filtered_abs_delta": _abs_delta(e_vqe_hc, e_exact_filtered_ref),
        "vqe_qk_vs_exact_filtered_abs_delta": _abs_delta(e_vqe_qk, e_exact_filtered_ref),
    }
    if exact_cross_delta is not None:
        deltas["exact_filtered_cross_source_abs_delta"] = exact_cross_delta

    checks = {
        "finite_energies": True,
        "adapt_hc_vs_qk_abs_delta": (
            deltas["adapt_hc_vs_qk_abs_delta"] <= float(thresholds["adapt_hc_vs_qk_abs_delta"])
        ),
        "adapt_hc_vs_exact_filtered_abs_delta": (
            deltas["adapt_hc_vs_exact_filtered_abs_delta"]
            <= float(thresholds["adapt_hc_vs_exact_filtered_abs_delta"])
        ),
        "adapt_qk_vs_exact_filtered_abs_delta": (
            deltas["adapt_qk_vs_exact_filtered_abs_delta"]
            <= float(thresholds["adapt_qk_vs_exact_filtered_abs_delta"])
        ),
        "vqe_hc_vs_qk_abs_delta": (
            deltas["vqe_hc_vs_qk_abs_delta"] <= float(thresholds["vqe_hc_vs_qk_abs_delta"])
        ),
        "vqe_hc_vs_exact_filtered_abs_delta": (
            deltas["vqe_hc_vs_exact_filtered_abs_delta"]
            <= float(thresholds["vqe_hc_vs_exact_filtered_abs_delta"])
        ),
        "vqe_qk_vs_exact_filtered_abs_delta": (
            deltas["vqe_qk_vs_exact_filtered_abs_delta"]
            <= float(thresholds["vqe_qk_vs_exact_filtered_abs_delta"])
        ),
        "exact_filtered_cross_source_abs_delta": (
            exact_cross_delta is None
            or exact_cross_delta <= float(thresholds["exact_filtered_cross_source_abs_delta"])
        ),
    }

    return {
        "energies": {
            "hardcoded_adapt_vqe": e_adapt_hc,
            "qiskit_adapt_vqe": e_adapt_qk,
            "hardcoded_vqe_uccsd": e_vqe_hc,
            "qiskit_vqe_uccsd": e_vqe_qk,
            "exact_filtered_reference": e_exact_filtered_ref,
            "exact_filtered_source_qiskit_vqe": e_exact_filtered_vqe,
            "exact_filtered_source_qiskit_adapt": e_exact_filtered_adapt,
        },
        "deltas": deltas,
        "acceptance": {
            "thresholds": thresholds,
            "checks": checks,
            "pass": bool(all(checks.values())),
        },
    }


def _append_csv_row(
    *,
    rows: list[dict[str, Any]],
    L: int,
    metrics: dict[str, Any],
    files: dict[str, str],
) -> None:
    energies = metrics["energies"]
    deltas = metrics["deltas"]
    checks = metrics["acceptance"]["checks"]
    rows.append(
        {
            "L": int(L),
            "pass": bool(metrics["acceptance"]["pass"]),
            "hardcoded_adapt_vqe": energies["hardcoded_adapt_vqe"],
            "qiskit_adapt_vqe": energies["qiskit_adapt_vqe"],
            "hardcoded_vqe_uccsd": energies["hardcoded_vqe_uccsd"],
            "qiskit_vqe_uccsd": energies["qiskit_vqe_uccsd"],
            "exact_filtered_reference": energies["exact_filtered_reference"],
            "exact_filtered_source_qiskit_adapt": energies["exact_filtered_source_qiskit_adapt"],
            "delta_adapt_hc_vs_qk": deltas["adapt_hc_vs_qk_abs_delta"],
            "delta_adapt_hc_vs_exact": deltas["adapt_hc_vs_exact_filtered_abs_delta"],
            "delta_adapt_qk_vs_exact": deltas["adapt_qk_vs_exact_filtered_abs_delta"],
            "delta_vqe_hc_vs_qk": deltas["vqe_hc_vs_qk_abs_delta"],
            "delta_vqe_hc_vs_exact": deltas["vqe_hc_vs_exact_filtered_abs_delta"],
            "delta_vqe_qk_vs_exact": deltas["vqe_qk_vs_exact_filtered_abs_delta"],
            "delta_exact_cross_source": deltas.get("exact_filtered_cross_source_abs_delta"),
            "check_exact_cross_source": checks["exact_filtered_cross_source_abs_delta"],
            "hardcoded_adapt_json": files["hardcoded_adapt_json"],
            "qiskit_adapt_json": files["qiskit_adapt_json"],
            "hardcoded_vqe_json": files["hardcoded_vqe_json"],
            "qiskit_vqe_json": files["qiskit_vqe_json"],
            "per_l_metrics_json": files["per_l_metrics_json"],
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "L",
        "pass",
        "hardcoded_adapt_vqe",
        "qiskit_adapt_vqe",
        "hardcoded_vqe_uccsd",
        "qiskit_vqe_uccsd",
        "exact_filtered_reference",
        "exact_filtered_source_qiskit_adapt",
        "delta_adapt_hc_vs_qk",
        "delta_adapt_hc_vs_exact",
        "delta_adapt_qk_vs_exact",
        "delta_vqe_hc_vs_qk",
        "delta_vqe_hc_vs_exact",
        "delta_vqe_qk_vs_exact",
        "delta_exact_cross_source",
        "check_exact_cross_source",
        "hardcoded_adapt_json",
        "qiskit_adapt_json",
        "hardcoded_vqe_json",
        "qiskit_vqe_json",
        "per_l_metrics_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare hardcoded/qiskit ADAPT-VQE and hardcoded/qiskit VQE "
            "against exact filtered ground-state energy."
        )
    )
    p.add_argument("--l-values", type=str, default="2,3")
    p.add_argument("--run-pipelines", action="store_true", default=True)
    p.add_argument("--no-run-pipelines", dest="run_pipelines", action="store_false")

    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")

    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--num-times", type=int, default=201)
    p.add_argument("--suzuki-order", type=int, default=2)
    p.add_argument("--trotter-steps", type=int, default=64)

    p.add_argument("--hardcoded-vqe-reps", type=int, default=2)
    p.add_argument("--hardcoded-vqe-restarts", type=int, default=3)
    p.add_argument("--hardcoded-vqe-seed", type=int, default=7)
    p.add_argument("--hardcoded-vqe-maxiter", type=int, default=600)

    p.add_argument("--qiskit-vqe-reps", type=int, default=2)
    p.add_argument("--qiskit-vqe-restarts", type=int, default=3)
    p.add_argument("--qiskit-vqe-seed", type=int, default=7)
    p.add_argument("--qiskit-vqe-maxiter", type=int, default=600)

    p.add_argument("--hc-adapt-max-depth", type=int, default=20)
    p.add_argument("--hc-adapt-eps-grad", type=float, default=1e-4)
    p.add_argument("--hc-adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--hc-adapt-maxiter", type=int, default=300)
    p.add_argument("--hc-adapt-seed", type=int, default=7)
    p.add_argument("--hc-adapt-allow-repeats", action="store_true", default=True)
    p.add_argument("--hc-adapt-no-repeats", dest="hc_adapt_allow_repeats", action="store_false")

    p.add_argument("--qk-adapt-max-iterations", type=int, default=20)
    p.add_argument("--qk-adapt-gradient-threshold", type=float, default=1e-4)
    p.add_argument("--qk-adapt-cobyla-maxiter", type=int, default=300)
    p.add_argument("--qk-adapt-seed", type=int, default=7)

    p.add_argument("--vqe-initial-state-source", choices=["exact", "vqe", "hf"], default="vqe")
    p.add_argument("--adapt-initial-state-source", choices=["exact", "adapt_vqe", "hf"], default="adapt_vqe")

    p.add_argument(
        "--thr-exact-filtered-cross-source-abs-delta",
        type=float,
        default=THRESHOLDS_DEFAULT["exact_filtered_cross_source_abs_delta"],
    )
    p.add_argument(
        "--thr-adapt-hc-vs-qk-abs-delta",
        type=float,
        default=THRESHOLDS_DEFAULT["adapt_hc_vs_qk_abs_delta"],
    )
    p.add_argument(
        "--thr-adapt-hc-vs-exact-filtered-abs-delta",
        type=float,
        default=THRESHOLDS_DEFAULT["adapt_hc_vs_exact_filtered_abs_delta"],
    )
    p.add_argument(
        "--thr-adapt-qk-vs-exact-filtered-abs-delta",
        type=float,
        default=THRESHOLDS_DEFAULT["adapt_qk_vs_exact_filtered_abs_delta"],
    )
    p.add_argument(
        "--thr-vqe-hc-vs-qk-abs-delta",
        type=float,
        default=THRESHOLDS_DEFAULT["vqe_hc_vs_qk_abs_delta"],
    )
    p.add_argument(
        "--thr-vqe-hc-vs-exact-filtered-abs-delta",
        type=float,
        default=THRESHOLDS_DEFAULT["vqe_hc_vs_exact_filtered_abs_delta"],
    )
    p.add_argument(
        "--thr-vqe-qk-vs-exact-filtered-abs-delta",
        type=float,
        default=THRESHOLDS_DEFAULT["vqe_qk_vs_exact_filtered_abs_delta"],
    )

    p.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("main_start", settings=vars(args))
    run_command = _current_command_string()

    if not VQE_ROOT.exists():
        raise FileNotFoundError(f"Expected VQE pipeline repo directory not found: {VQE_ROOT}")

    l_values = [int(x.strip()) for x in str(args.l_values).split(",") if x.strip()]
    if not l_values:
        raise ValueError("No L values parsed from --l-values.")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "exact_filtered_cross_source_abs_delta": float(args.thr_exact_filtered_cross_source_abs_delta),
        "adapt_hc_vs_qk_abs_delta": float(args.thr_adapt_hc_vs_qk_abs_delta),
        "adapt_hc_vs_exact_filtered_abs_delta": float(args.thr_adapt_hc_vs_exact_filtered_abs_delta),
        "adapt_qk_vs_exact_filtered_abs_delta": float(args.thr_adapt_qk_vs_exact_filtered_abs_delta),
        "vqe_hc_vs_qk_abs_delta": float(args.thr_vqe_hc_vs_qk_abs_delta),
        "vqe_hc_vs_exact_filtered_abs_delta": float(args.thr_vqe_hc_vs_exact_filtered_abs_delta),
        "vqe_qk_vs_exact_filtered_abs_delta": float(args.thr_vqe_qk_vs_exact_filtered_abs_delta),
    }

    command_log_path = artifacts_dir / "adapt_vqe_vqe_exact_filtered_commands_run.txt"
    command_log: list[str] = []
    run_artifacts: list[RunArtifacts] = []

    for L in l_values:
        hardcoded_adapt_json = artifacts_dir / f"hardcoded_adapt_pipeline_L{L}.json"
        qiskit_adapt_json = artifacts_dir / f"qiskit_adapt_pipeline_L{L}.json"
        hardcoded_vqe_json = artifacts_dir / f"hardcoded_pipeline_L{L}.json"
        qiskit_vqe_json = artifacts_dir / f"qiskit_pipeline_L{L}.json"
        per_l_metrics_json = artifacts_dir / f"adapt_vqe_vqe_exact_filtered_L{L}_metrics.json"

        hardcoded_adapt_pdf = artifacts_dir / f"hardcoded_adapt_pipeline_L{L}.pdf"
        qiskit_adapt_pdf = artifacts_dir / f"qiskit_adapt_pipeline_L{L}.pdf"
        hardcoded_vqe_pdf = artifacts_dir / f"hardcoded_pipeline_L{L}.pdf"
        qiskit_vqe_pdf = artifacts_dir / f"qiskit_pipeline_L{L}.pdf"

        if args.run_pipelines:
            hc_adapt_cmd = [
                sys.executable,
                str(ROOT / "pipelines" / "hardcoded_adapt_pipeline.py"),
                "--L",
                str(L),
                "--t",
                str(args.t),
                "--u",
                str(args.u),
                "--dv",
                str(args.dv),
                "--boundary",
                str(args.boundary),
                "--ordering",
                str(args.ordering),
                "--t-final",
                str(args.t_final),
                "--num-times",
                str(args.num_times),
                "--suzuki-order",
                str(args.suzuki_order),
                "--trotter-steps",
                str(args.trotter_steps),
                "--term-order",
                "sorted",
                "--adapt-max-depth",
                str(args.hc_adapt_max_depth),
                "--adapt-eps-grad",
                str(args.hc_adapt_eps_grad),
                "--adapt-eps-energy",
                str(args.hc_adapt_eps_energy),
                "--adapt-maxiter",
                str(args.hc_adapt_maxiter),
                "--adapt-seed",
                str(args.hc_adapt_seed),
                "--initial-state-source",
                str(args.adapt_initial_state_source),
                "--output-json",
                str(hardcoded_adapt_json),
                "--output-pdf",
                str(hardcoded_adapt_pdf),
                "--skip-pdf",
            ]
            hc_adapt_cmd.append("--adapt-allow-repeats" if args.hc_adapt_allow_repeats else "--adapt-no-repeats")
            command_log.append(" ".join(shlex.quote(x) for x in hc_adapt_cmd))
            code, out, err = _run_command(hc_adapt_cmd, cwd=REPO_ROOT)
            if code != 0:
                raise RuntimeError(
                    f"Hardcoded ADAPT pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )

            qk_adapt_cmd = [
                sys.executable,
                str(ROOT / "pipelines" / "qiskit_adapt_pipeline.py"),
                "--L",
                str(L),
                "--t",
                str(args.t),
                "--u",
                str(args.u),
                "--dv",
                str(args.dv),
                "--boundary",
                str(args.boundary),
                "--ordering",
                str(args.ordering),
                "--t-final",
                str(args.t_final),
                "--num-times",
                str(args.num_times),
                "--suzuki-order",
                str(args.suzuki_order),
                "--trotter-steps",
                str(args.trotter_steps),
                "--term-order",
                "sorted",
                "--adapt-max-iterations",
                str(args.qk_adapt_max_iterations),
                "--adapt-gradient-threshold",
                str(args.qk_adapt_gradient_threshold),
                "--adapt-cobyla-maxiter",
                str(args.qk_adapt_cobyla_maxiter),
                "--adapt-seed",
                str(args.qk_adapt_seed),
                "--initial-state-source",
                str(args.adapt_initial_state_source),
                "--output-json",
                str(qiskit_adapt_json),
                "--output-pdf",
                str(qiskit_adapt_pdf),
                "--skip-pdf",
            ]
            command_log.append(" ".join(shlex.quote(x) for x in qk_adapt_cmd))
            code, out, err = _run_command(qk_adapt_cmd, cwd=REPO_ROOT)
            if code != 0:
                raise RuntimeError(
                    f"Qiskit ADAPT pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )

            hc_vqe_cmd = [
                sys.executable,
                str(VQE_ROOT / "pipelines" / "hardcoded_hubbard_pipeline.py"),
                "--L",
                str(L),
                "--t",
                str(args.t),
                "--u",
                str(args.u),
                "--dv",
                str(args.dv),
                "--boundary",
                str(args.boundary),
                "--ordering",
                str(args.ordering),
                "--t-final",
                str(args.t_final),
                "--num-times",
                str(args.num_times),
                "--suzuki-order",
                str(args.suzuki_order),
                "--trotter-steps",
                str(args.trotter_steps),
                "--term-order",
                "sorted",
                "--vqe-reps",
                str(args.hardcoded_vqe_reps),
                "--vqe-restarts",
                str(args.hardcoded_vqe_restarts),
                "--vqe-seed",
                str(args.hardcoded_vqe_seed),
                "--vqe-maxiter",
                str(args.hardcoded_vqe_maxiter),
                "--initial-state-source",
                str(args.vqe_initial_state_source),
                "--skip-qpe",
                "--output-json",
                str(hardcoded_vqe_json),
                "--output-pdf",
                str(hardcoded_vqe_pdf),
                "--skip-pdf",
            ]
            command_log.append(" ".join(shlex.quote(x) for x in hc_vqe_cmd))
            code, out, err = _run_command(hc_vqe_cmd, cwd=REPO_ROOT)
            if code != 0:
                raise RuntimeError(
                    f"Hardcoded VQE pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )

            qk_vqe_cmd = [
                sys.executable,
                str(VQE_ROOT / "pipelines" / "qiskit_hubbard_baseline_pipeline.py"),
                "--L",
                str(L),
                "--t",
                str(args.t),
                "--u",
                str(args.u),
                "--dv",
                str(args.dv),
                "--boundary",
                str(args.boundary),
                "--ordering",
                str(args.ordering),
                "--t-final",
                str(args.t_final),
                "--num-times",
                str(args.num_times),
                "--suzuki-order",
                str(args.suzuki_order),
                "--trotter-steps",
                str(args.trotter_steps),
                "--term-order",
                "sorted",
                "--vqe-reps",
                str(args.qiskit_vqe_reps),
                "--vqe-restarts",
                str(args.qiskit_vqe_restarts),
                "--vqe-seed",
                str(args.qiskit_vqe_seed),
                "--vqe-maxiter",
                str(args.qiskit_vqe_maxiter),
                "--initial-state-source",
                str(args.vqe_initial_state_source),
                "--skip-qpe",
                "--output-json",
                str(qiskit_vqe_json),
                "--output-pdf",
                str(qiskit_vqe_pdf),
                "--skip-pdf",
            ]
            command_log.append(" ".join(shlex.quote(x) for x in qk_vqe_cmd))
            code, out, err = _run_command(qk_vqe_cmd, cwd=REPO_ROOT)
            if code != 0:
                raise RuntimeError(
                    f"Qiskit VQE pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )

        run_artifacts.append(
            RunArtifacts(
                L=int(L),
                hardcoded_adapt_json=hardcoded_adapt_json,
                qiskit_adapt_json=qiskit_adapt_json,
                hardcoded_vqe_json=hardcoded_vqe_json,
                qiskit_vqe_json=qiskit_vqe_json,
                per_l_metrics_json=per_l_metrics_json,
            )
        )

    if command_log:
        with command_log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(command_log) + "\n")
        _ai_log("command_log_written", path=str(command_log_path), commands=len(command_log))

    results_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for row in run_artifacts:
        hardcoded_adapt = _read_json(row.hardcoded_adapt_json)
        qiskit_adapt = _read_json(row.qiskit_adapt_json)
        hardcoded_vqe = _read_json(row.hardcoded_vqe_json)
        qiskit_vqe = _read_json(row.qiskit_vqe_json)

        metrics = _energy_metrics_for_l(
            L=row.L,
            hardcoded_adapt=hardcoded_adapt,
            qiskit_adapt=qiskit_adapt,
            hardcoded_vqe=hardcoded_vqe,
            qiskit_vqe=qiskit_vqe,
            thresholds=thresholds,
        )

        per_l_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "L": int(row.L),
            "sources": {
                "hardcoded_adapt_json": str(row.hardcoded_adapt_json),
                "qiskit_adapt_json": str(row.qiskit_adapt_json),
                "hardcoded_vqe_json": str(row.hardcoded_vqe_json),
                "qiskit_vqe_json": str(row.qiskit_vqe_json),
            },
            "metrics": metrics,
        }
        row.per_l_metrics_json.write_text(json.dumps(per_l_payload, indent=2), encoding="utf-8")

        result_row = {
            "L": int(row.L),
            "pass": bool(metrics["acceptance"]["pass"]),
            "metrics_json": str(row.per_l_metrics_json),
            "energies": metrics["energies"],
            "deltas": metrics["deltas"],
            "checks": metrics["acceptance"]["checks"],
        }
        results_rows.append(result_row)

        _append_csv_row(
            rows=csv_rows,
            L=row.L,
            metrics=metrics,
            files={
                "hardcoded_adapt_json": str(row.hardcoded_adapt_json),
                "qiskit_adapt_json": str(row.qiskit_adapt_json),
                "hardcoded_vqe_json": str(row.hardcoded_vqe_json),
                "qiskit_vqe_json": str(row.qiskit_vqe_json),
                "per_l_metrics_json": str(row.per_l_metrics_json),
            },
        )
        _ai_log(
            "l_metrics_done",
            L=int(row.L),
            passed=bool(metrics["acceptance"]["pass"]),
            delta_adapt_hc_vs_qk=float(metrics["deltas"]["adapt_hc_vs_qk_abs_delta"]),
            delta_vqe_hc_vs_qk=float(metrics["deltas"]["vqe_hc_vs_qk_abs_delta"]),
            delta_exact_cross_source=metrics["deltas"].get("exact_filtered_cross_source_abs_delta"),
        )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Unified comparison: hardcoded/qiskit ADAPT-VQE and hardcoded/qiskit "
            "VQE energies against exact filtered ground-state reference."
        ),
        "script": "pipelines/compare_adapt_vqe_vqe_exact_filtered.py",
        "run_command": run_command,
        "l_values": l_values,
        "thresholds": thresholds,
        "requested_run_settings": {
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "vqe_initial_state_source": str(args.vqe_initial_state_source),
            "adapt_initial_state_source": str(args.adapt_initial_state_source),
            "run_pipelines": bool(args.run_pipelines),
            "skip_qpe_for_vqe_runs": True,
        },
        "results": results_rows,
        "all_pass": bool(all(r["pass"] for r in results_rows)),
    }

    summary_json = artifacts_dir / "adapt_vqe_vqe_exact_filtered_summary.json"
    csv_path = artifacts_dir / "adapt_vqe_vqe_exact_filtered_table.csv"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, csv_rows)

    _ai_log(
        "main_done",
        summary_json=str(summary_json),
        csv_path=str(csv_path),
        all_pass=bool(summary["all_pass"]),
    )
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote CSV table:    {csv_path}")
    if command_log:
        print(f"Wrote command log:  {command_log_path}")
    for row in results_rows:
        print(f"L={row['L']}: pass={row['pass']} metrics={row['metrics_json']}")


if __name__ == "__main__":
    main()
