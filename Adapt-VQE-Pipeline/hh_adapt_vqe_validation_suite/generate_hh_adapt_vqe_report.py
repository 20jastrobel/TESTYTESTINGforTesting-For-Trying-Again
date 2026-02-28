#!/usr/bin/env python3
"""Generate a PDF report for HH ADAPT-VQE validation artifacts."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

SUITE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SUITE_DIR.parent
ARTIFACT_DIR = SUITE_DIR / "artifacts"
DEFAULT_OUTPUT = ARTIFACT_DIR / "hh_adapt_vqe_validation_report.pdf"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pipelines import hardcoded_adapt_pipeline as hardcoded_adapt
except Exception as exc:  # pragma: no cover - defensive fallback for report-only utility
    hardcoded_adapt = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _parse_attempt(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"attempt_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _run_id(run: dict[str, Any]) -> str:
    attempt = run.get("attempt")
    attempt_str = f"a{attempt}" if attempt is not None else "a-"
    seed = run.get("seed")
    seed_str = f"s{seed}" if seed is not None else "s-"
    return f"L{run.get('L')}|{seed_str}|{attempt_str}"


def _discover_runs() -> list[dict[str, Any]]:
    paths = sorted(p for p in ARTIFACT_DIR.rglob("*.json") if p.is_file())
    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = payload.get("settings", {})
        adapt = payload.get("adapt_vqe", {})
        trajectory = payload.get("trajectory", [])
        final_fidelity = float("nan")
        min_fidelity = float("nan")
        if trajectory:
            try:
                final_fidelity = _safe_float(trajectory[-1].get("fidelity"))
                min_fidelity = float(min(_safe_float(t.get("fidelity")) for t in trajectory))
            except Exception:
                pass

        operators = [str(x) for x in adapt.get("operators", [])]
        history = adapt.get("history", []) if isinstance(adapt.get("history", []), list) else []
        runs.append({
            "path": path,
            "payload": payload,
            "settings": settings,
            "adapt": adapt,
            "L": int(settings.get("L", -1)),
            "seed": _parse_seed(path),
            "attempt": _parse_attempt(path),
            "success": bool(adapt.get("success", False)),
            "energy": _safe_float(adapt.get("energy")),
            "exact_energy": _safe_float(adapt.get("exact_gs_energy", payload.get("ground_state", {}).get("exact_energy"))),
            "abs_delta_e": _safe_float(adapt.get("abs_delta_e")),
            "ansatz_depth": int(adapt.get("ansatz_depth", len(operators) if operators else 0)),
            "nfev_total": int(adapt.get("nfev_total", 0)),
            "elapsed_s": _safe_float(adapt.get("elapsed_s")),
            "pool_type": str(adapt.get("pool_type", settings.get("adapt_pool", "unknown"))),
            "method": str(adapt.get("method", "unknown")),
            "operators": operators,
            "optimal_point": [float(x) for x in adapt.get("optimal_point", [])],
            "history": history,
            "ham_num_qubits": int(payload.get("hamiltonian", {}).get("num_qubits", 0)),
            "ham_num_terms": int(payload.get("hamiltonian", {}).get("num_terms", 0)),
            "final_fidelity": final_fidelity,
            "min_fidelity": min_fidelity,
        })

    runs.sort(key=lambda x: (x["L"], -1 if x["seed"] is None else x["seed"], str(x["path"])))
    return runs


def _generator_breakdown(labels: list[str]) -> dict[str, int]:
    counts = Counter()
    for label in labels:
        if label.startswith("uccsd_sing"):
            counts["uccsd_sing"] += 1
        elif label.startswith("uccsd_dbl"):
            counts["uccsd_dbl"] += 1
        elif "hva" in label.lower():
            counts["hva_layerwise"] += 1
        else:
            counts["other"] += 1
    return {
        "uccsd_sing": int(counts.get("uccsd_sing", 0)),
        "uccsd_dbl": int(counts.get("uccsd_dbl", 0)),
        "hva_layerwise": int(counts.get("hva_layerwise", 0)),
        "other": int(counts.get("other", 0)),
    }


def _pool_lookup(settings: dict[str, Any], cache: dict[tuple[Any, ...], dict[str, Any] | None]) -> dict[str, Any] | None:
    if hardcoded_adapt is None:
        return None

    key = (
        int(settings.get("L", 0)),
        float(settings.get("t", 0.0)),
        float(settings.get("u", 0.0)),
        float(settings.get("omega0", 0.0)),
        float(settings.get("g_ep", 0.0)),
        float(settings.get("dv", 0.0)),
        int(settings.get("n_ph_max", 1)),
        str(settings.get("boson_encoding", "binary")),
        str(settings.get("ordering", "blocked")),
        str(settings.get("boundary", "open")),
    )
    if key in cache:
        return cache[key]

    try:
        pool = hardcoded_adapt._build_hva_pool(
            num_sites=int(settings.get("L")),
            t=float(settings.get("t")),
            u=float(settings.get("u")),
            omega0=float(settings.get("omega0")),
            g_ep=float(settings.get("g_ep")),
            dv=float(settings.get("dv")),
            n_ph_max=int(settings.get("n_ph_max")),
            boson_encoding=str(settings.get("boson_encoding")),
            ordering=str(settings.get("ordering")),
            boundary=str(settings.get("boundary")),
        )
    except Exception:
        cache[key] = None
        return None

    cache[key] = {str(term.label): term for term in pool}
    return cache[key]


def _estimate_gate_counts(run: dict[str, Any], cache: dict[tuple[Any, ...], dict[str, Any] | None]) -> dict[str, Any]:
    lookup = _pool_lookup(run["settings"], cache)
    labels = run["operators"]
    totals = {
        "exp_terms": 0,
        "cnot": 0,
        "rz": 0,
        "h": 0,
        "s_or_sdg": 0,
        "max_pauli_support": 0,
        "mean_pauli_support": 0.0,
        "unresolved_labels": 0,
    }
    if not labels:
        return totals
    if lookup is None:
        totals["unresolved_labels"] = len(labels)
        return totals

    support_values: list[int] = []
    missing = 0
    for op_label in labels:
        term = lookup.get(op_label)
        if term is None:
            missing += 1
            continue
        for pterm in term.polynomial.return_polynomial():
            word = str(pterm.pw2strng())
            support = sum(1 for ch in word if ch != "e")
            if support == 0:
                continue
            x_count = word.count("x")
            y_count = word.count("y")
            totals["exp_terms"] += 1
            totals["rz"] += 1
            totals["cnot"] += 2 * (support - 1)
            totals["h"] += 2 * (x_count + y_count)
            totals["s_or_sdg"] += 2 * y_count
            totals["max_pauli_support"] = max(totals["max_pauli_support"], support)
            support_values.append(support)

    totals["unresolved_labels"] = int(missing)
    if support_values:
        totals["mean_pauli_support"] = float(np.mean(np.asarray(support_values, dtype=float)))
    return totals


def _fmt(x: Any, digits: int = 6) -> str:
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, int):
        return str(x)
    xf = _safe_float(x, default=float("nan"))
    if not math.isfinite(xf):
        return "nan"
    return f"{xf:.{digits}g}"


def _render_text_pages(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 8) -> None:
    line_spacing = 0.028
    cursor = 0
    while cursor < len(lines):
        fig = plt.figure(figsize=(11.0, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            title,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )
        y = 0.93
        while cursor < len(lines) and y > 0.03:
            ax.text(
                0.02,
                y,
                lines[cursor],
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=fontsize,
                family="monospace",
            )
            y -= line_spacing
            cursor += 1
        pdf.savefig(fig)
        plt.close(fig)


def _make_report(runs: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gate_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}

    for run in runs:
        run["generator_counts"] = _generator_breakdown(run["operators"])
        run["gate_counts"] = _estimate_gate_counts(run, gate_cache)

    with PdfPages(str(output_path)) as pdf:
        now = datetime.now(timezone.utc).isoformat()
        pass_count = sum(1 for r in runs if r["success"])
        lines = [
            f"Generated UTC: {now}",
            f"Artifacts scanned: {len(runs)}",
            f"Passing runs: {pass_count}/{len(runs)}",
            "",
            "Algorithm:",
            "  hardcoded ADAPT-VQE on the Hubbard-Holstein model (problem=hh)",
            "  pool type: hva (HH layerwise + lifted UCCSD termwise generators)",
            "  optimizer: COBYLA over full selected-operator parameter vector per depth",
            "  convergence target: |Delta E| < 1e-4 vs exact_ground_energy_sector",
            "",
            "Gate model used in this report (term-wise Pauli exponential estimate):",
            "  For exp(-i theta P) with Pauli support k:",
            "  CNOT = 2*(k-1), RZ = 1, H = 2*(#X + #Y), S/Sdg = 2*(#Y)",
            "",
            "Notes:",
            "  These are decomposition estimates inferred from selected generators.",
            "  If a generator contains multiple Pauli terms, this is a term-wise upper-bound style estimate.",
        ]
        if _IMPORT_ERROR:
            lines.extend([
                "",
                "Pool import warning:",
                f"  { _IMPORT_ERROR }",
                "  Gate estimates for unresolved generators are marked in tables.",
            ])
        _render_text_pages(pdf, "HH ADAPT-VQE Validation Report", lines, fontsize=9)

        headers = [
            "run_id", "success", "L", "seed", "method", "pool",
            "energy", "exact", "abs_delta_e", "depth", "nfev", "elapsed_s",
            "ham_q", "ham_terms", "traj_f_final", "traj_f_min",
        ]
        rows = []
        for run in runs:
            rows.append([
                _run_id(run),
                str(run["success"]),
                str(run["L"]),
                str(run["seed"]),
                run["method"],
                run["pool_type"],
                _fmt(run["energy"], 9),
                _fmt(run["exact_energy"], 9),
                _fmt(run["abs_delta_e"], 3),
                str(run["ansatz_depth"]),
                str(run["nfev_total"]),
                _fmt(run["elapsed_s"], 4),
                str(run["ham_num_qubits"]),
                str(run["ham_num_terms"]),
                _fmt(run["final_fidelity"], 4),
                _fmt(run["min_fidelity"], 4),
            ])

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.axis("off")
        tbl = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.3)
        ax.set_title("Run Summary Table", fontsize=14, pad=10)
        pdf.savefig(fig)
        plt.close(fig)

        labels = [_run_id(r) for r in runs]
        x = np.arange(len(runs))
        abs_delta = np.asarray([max(_safe_float(r["abs_delta_e"], 1e-20), 1e-20) for r in runs], dtype=float)
        energies = np.asarray([_safe_float(r["energy"]) for r in runs], dtype=float)
        exact = np.asarray([_safe_float(r["exact_energy"]) for r in runs], dtype=float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.bar(x, abs_delta, color="#1f77b4")
        ax1.axhline(1e-4, color="#d62728", linestyle="--", linewidth=1.5, label="target 1e-4")
        ax1.set_yscale("log")
        ax1.set_xticks(x, labels, rotation=35, ha="right")
        ax1.set_ylabel("|Delta E|")
        ax1.set_title("Energy Error by Run")
        ax1.grid(alpha=0.25, axis="y")
        ax1.legend(fontsize=8)

        ax2.scatter(x, exact, color="#111111", label="Exact", marker="o", s=48)
        ax2.scatter(x, energies, color="#ff7f0e", label="ADAPT-VQE", marker="x", s=64)
        for i in range(len(runs)):
            ax2.plot([x[i], x[i]], [exact[i], energies[i]], color="#999999", linewidth=0.8)
        ax2.set_xticks(x, labels, rotation=35, ha="right")
        ax2.set_ylabel("Energy")
        ax2.set_title("ADAPT-VQE Energy vs Exact")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8)

        fig.suptitle("Accuracy Plots", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(13, 7))
        ax = fig.add_subplot(111)
        has_curve = False
        for run in runs:
            history = run["history"]
            if not history:
                continue
            depths = np.asarray([int(h.get("depth", i + 1)) for i, h in enumerate(history)], dtype=int)
            exact_e = _safe_float(run["exact_energy"])
            err_vals = []
            for h in history:
                err_vals.append(abs(_safe_float(h.get("energy_after_opt")) - exact_e))
            err = np.asarray([max(v, 1e-20) for v in err_vals], dtype=float)
            ax.plot(depths, err, marker="o", markersize=2.5, linewidth=1.0, label=_run_id(run))
            has_curve = True
        if has_curve:
            ax.set_yscale("log")
            ax.axhline(1e-4, color="#d62728", linestyle="--", linewidth=1.2, label="target 1e-4")
            ax.set_xlabel("ADAPT Depth")
            ax.set_ylabel("|E_depth - E_exact|")
            ax.set_title("Convergence History")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No convergence history available in artifacts.", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)

        gate_headers = [
            "run_id", "depth", "sing", "dbl", "hva/other",
            "exp_terms", "CNOT", "RZ", "H", "S/Sdg",
            "max_supp", "mean_supp", "unresolved",
        ]
        gate_rows = []
        for run in runs:
            g = run["generator_counts"]
            c = run["gate_counts"]
            gate_rows.append([
                _run_id(run),
                str(run["ansatz_depth"]),
                str(g["uccsd_sing"]),
                str(g["uccsd_dbl"]),
                str(g["hva_layerwise"] + g["other"]),
                str(c["exp_terms"]),
                str(c["cnot"]),
                str(c["rz"]),
                str(c["h"]),
                str(c["s_or_sdg"]),
                str(c["max_pauli_support"]),
                _fmt(c["mean_pauli_support"], 3),
                str(c["unresolved_labels"]),
            ])

        fig = plt.figure(figsize=(13, 7))
        ax = fig.add_subplot(111)
        ax.axis("off")
        gate_tbl = ax.table(cellText=gate_rows, colLabels=gate_headers, loc="center", cellLoc="center")
        gate_tbl.auto_set_font_size(False)
        gate_tbl.set_fontsize(8)
        gate_tbl.scale(1.0, 1.4)
        ax.set_title("Generator Type and Estimated Gate Counts", fontsize=14, pad=10)
        pdf.savefig(fig)
        plt.close(fig)

        op_lines: list[str] = []
        for run in runs:
            rid = _run_id(run)
            op_lines.append(f"{rid}  method={run['method']}  pool={run['pool_type']}  depth={run['ansatz_depth']}")
            for idx, op in enumerate(run["operators"], start=1):
                theta = "nan"
                if idx - 1 < len(run["optimal_point"]):
                    theta = _fmt(run["optimal_point"][idx - 1], 7)
                op_lines.append(f"  {idx:02d}. {op:<42} theta={theta}")
            op_lines.append("")
            op_lines.append(f"  artifact: {run['path']}")
            op_lines.append("")
        _render_text_pages(pdf, "Selected Generators and Parameters", op_lines, fontsize=8)


def main() -> None:
    output_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT.resolve()
    runs = _discover_runs()
    if not runs:
        raise RuntimeError(f"No JSON artifacts found under {ARTIFACT_DIR}")
    _make_report(runs, output_path)
    print(f"Wrote PDF report: {output_path}")


if __name__ == "__main__":
    main()
