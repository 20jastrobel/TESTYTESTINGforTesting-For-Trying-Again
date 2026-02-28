#!/usr/bin/env python3
"""Compare hardcoded ADAPT-VQE vs Qiskit ADAPT-VQE pipelines for L=2,3,...

Outputs:
- per-L metrics JSON
- per-L comparison PDFs (optional)
- overall summary JSON (`all_pass`)
- bundled PDF report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pprint
import shlex
import subprocess
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

# --- Acceptance thresholds ---
# ADAPT-VQE comparison: both pipelines use the same Hamiltonian, same Trotter
# dynamics from their respective ADAPT ground states.  The ADAPT energies
# themselves may differ because the algorithms are distinct; the *trajectory*
# comparison is the apples-to-apples check.
THRESHOLDS = {
    "ground_state_energy_abs_delta": 1e-8,
    "fidelity_max_abs_delta": 1e-4,
    "energy_trotter_max_abs_delta": 1e-3,
    "n_up_site0_trotter_max_abs_delta": 5e-3,
    "n_dn_site0_trotter_max_abs_delta": 5e-3,
    "doublon_trotter_max_abs_delta": 1e-3,
}

TARGET_METRICS = [
    "fidelity",
    "energy_trotter",
    "n_up_site0_trotter",
    "n_dn_site0_trotter",
    "doublon_trotter",
]

EXACT_LABEL_HARDCODE = "Exact_Hardcode"
EXACT_LABEL_QISKIT = "Exact_Qiskit"
EXACT_METHOD = "python_matrix_eigendecomposition"


@dataclass
class RunArtifacts:
    L: int
    hardcoded_json: Path
    hardcoded_pdf: Path
    qiskit_json: Path
    qiskit_pdf: Path
    compare_metrics_json: Path
    compare_pdf: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _first_crossing(times: np.ndarray, vals: np.ndarray, thr: float) -> float | None:
    idx = np.where(vals > thr)[0]
    if idx.size == 0:
        return None
    return float(times[int(idx[0])])


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def _fp(x: float) -> str:
    return repr(float(x))


def _sci(x: float) -> str:
    return f"{x:.2e}"


def _fmt_obj(obj: Any, *, width: int = 90) -> str:
    formatted = pprint.pformat(obj, width=width, compact=True, sort_dicts=True)
    wrapped_lines: list[str] = []
    for line in formatted.splitlines():
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=width, subsequent_indent="  "))
    return "\n".join(wrapped_lines)


def _delta_metric_definition_text() -> str:
    return (
        "ΔF(t)      = |F_hc(t) - F_qk(t)|\n"
        "ΔE_trot(t) = |E_trot_hc(t) - E_trot_qk(t)|\n"
        "Δn_up0(t)  = |n_up0_hc(t) - n_up0_qk(t)|\n"
        "Δn_dn0(t)  = |n_dn0_hc(t) - n_dn0_qk(t)|\n"
        "ΔD(t)      = |D_hc(t) - D_qk(t)|\n"
        "F_pipeline(t) is the pipeline's stored trajectory fidelity value "
        "(as computed internally vs that pipeline's exact evolution)."
    )


# ---------------------------------------------------------------------------
# Payload comparison
# ---------------------------------------------------------------------------

def _compare_payloads(hardcoded: dict[str, Any], qiskit: dict[str, Any]) -> dict[str, Any]:
    h_rows = hardcoded["trajectory"]
    q_rows = qiskit["trajectory"]

    if len(h_rows) != len(q_rows):
        raise ValueError("Trajectory length mismatch between hardcoded and qiskit outputs.")

    t_h = _arr(h_rows, "time")
    t_q = _arr(q_rows, "time")
    if not np.allclose(t_h, t_q, atol=1e-12, rtol=0.0):
        raise ValueError("Time-grid mismatch between hardcoded and qiskit outputs.")

    out: dict[str, Any] = {
        "time_grid": {
            "num_times": int(t_h.size),
            "t0": float(t_h[0]),
            "t_final": float(t_h[-1]),
            "dt": float(t_h[1] - t_h[0]) if t_h.size > 1 else 0.0,
        },
        "trajectory_deltas": {},
    }

    for key in TARGET_METRICS:
        h = _arr(h_rows, key)
        q = _arr(q_rows, key)
        d = np.abs(h - q)
        out["trajectory_deltas"][key] = {
            "max_abs_delta": float(np.max(d)),
            "mean_abs_delta": float(np.mean(d)),
            "final_abs_delta": float(d[-1]),
            "first_time_abs_delta_gt_1e-4": _first_crossing(t_h, d, 1e-4),
            "first_time_abs_delta_gt_1e-3": _first_crossing(t_h, d, 1e-3),
        }

    gs_h = float(hardcoded["ground_state"]["exact_energy"])
    gs_q = float(qiskit["ground_state"]["exact_energy"])
    out["ground_state_energy"] = {
        "hardcoded_exact_energy": gs_h,
        "qiskit_exact_energy": gs_q,
        "abs_delta": float(abs(gs_h - gs_q)),
    }

    # ADAPT-VQE energy comparison
    hc_adapt = hardcoded.get("adapt_vqe", {})
    qk_adapt = qiskit.get("adapt_vqe", {})
    hc_adapt_e = hc_adapt.get("energy")
    qk_adapt_e = qk_adapt.get("energy")
    out["adapt_vqe_energy"] = {
        "hardcoded_adapt_energy": float(hc_adapt_e) if hc_adapt_e is not None else None,
        "qiskit_adapt_energy": float(qk_adapt_e) if qk_adapt_e is not None else None,
        "abs_delta": float(abs(float(hc_adapt_e) - float(qk_adapt_e)))
        if hc_adapt_e is not None and qk_adapt_e is not None
        else None,
    }

    checks = {
        "ground_state_energy_abs_delta": out["ground_state_energy"]["abs_delta"] <= THRESHOLDS["ground_state_energy_abs_delta"],
        "fidelity_max_abs_delta": out["trajectory_deltas"]["fidelity"]["max_abs_delta"] <= THRESHOLDS["fidelity_max_abs_delta"],
        "energy_trotter_max_abs_delta": out["trajectory_deltas"]["energy_trotter"]["max_abs_delta"] <= THRESHOLDS["energy_trotter_max_abs_delta"],
        "n_up_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_up_site0_trotter"]["max_abs_delta"] <= THRESHOLDS["n_up_site0_trotter_max_abs_delta"],
        "n_dn_site0_trotter_max_abs_delta": out["trajectory_deltas"]["n_dn_site0_trotter"]["max_abs_delta"] <= THRESHOLDS["n_dn_site0_trotter_max_abs_delta"],
        "doublon_trotter_max_abs_delta": out["trajectory_deltas"]["doublon_trotter"]["max_abs_delta"] <= THRESHOLDS["doublon_trotter_max_abs_delta"],
    }
    out["acceptance"] = {
        "thresholds": THRESHOLDS,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    return out


# ---------------------------------------------------------------------------
# Info-box helpers
# ---------------------------------------------------------------------------

_INFO_BOX_SETTINGS_KEYS = [
    "L", "t", "u", "dv", "boundary", "ordering",
    "initial_state_source", "t_final", "num_times",
    "suzuki_order", "trotter_steps",
]


def _build_info_box_text(
    settings: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    parts: list[str] = []
    for k in _INFO_BOX_SETTINGS_KEYS:
        v = settings.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    settings_line = "  ".join(parts)

    thr = metrics["acceptance"]["thresholds"]
    thr_lines = [f"  {k}: {_sci(v)}" for k, v in sorted(thr.items())]

    td = metrics["trajectory_deltas"]
    delta_lines = []
    for key in sorted(td.keys()):
        delta_lines.append(f"  {key}: {_sci(td[key]['max_abs_delta'])}")
    gs_delta = metrics["ground_state_energy"]["abs_delta"]
    delta_lines.insert(0, f"  gs_energy: {_sci(gs_delta)}")
    verdict = "PASS" if metrics["acceptance"]["pass"] else "FAIL"

    return "\n".join([
        settings_line,
        "",
        "thresholds:",
        *thr_lines,
        "",
        "max |Δ|:",
        *delta_lines,
        f"result: {verdict}",
    ])


_INFO_BBOX = dict(
    boxstyle="round,pad=0.4",
    facecolor="white",
    edgecolor="#888888",
    alpha=0.80,
)


# ---------------------------------------------------------------------------
# PDF rendering helpers
# ---------------------------------------------------------------------------

def _render_text_page(
    pdf: PdfPages,
    lines: list[str],
    *,
    fontsize: int = 9,
    line_spacing: float = 0.028,
    max_line_width: int = 115,
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


def _current_command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _render_command_page(pdf: PdfPages, command: str) -> None:
    lines = [
        "Executed Command",
        "",
        "Script: pipelines/compare_adapt_pipelines.py",
        "",
        command,
    ]
    _render_text_page(pdf, lines, fontsize=10, line_spacing=0.03, max_line_width=110)


def _render_info_page(pdf: PdfPages, info_text: str, title: str = "") -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    ax.text(
        0.05, 0.92, info_text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        family="monospace",
        bbox=_INFO_BBOX,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _autozoom(ax: Any, *arrays: np.ndarray, pad_frac: float = 0.05) -> None:
    combined = np.concatenate([a for a in arrays if a.size > 0])
    lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
    span = hi - lo
    pad = span * pad_frac if span > 0 else 1e-8
    ax.set_ylim(lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# Per-L comparison PDF
# ---------------------------------------------------------------------------

def _write_comparison_pdf(
    *,
    pdf_path: Path,
    L: int,
    hardcoded: dict[str, Any],
    qiskit: dict[str, Any],
    metrics: dict[str, Any],
    run_command: str,
) -> None:
    h_rows = hardcoded["trajectory"]
    q_rows = qiskit["trajectory"]
    times = _arr(h_rows, "time")
    markevery = max(1, times.size // 25)

    def h(key: str) -> np.ndarray:
        return _arr(h_rows, key)

    def q(key: str) -> np.ndarray:
        return _arr(q_rows, key)

    with PdfPages(str(pdf_path)) as pdf:
        _render_command_page(pdf, run_command)
        _info_text = _build_info_box_text(hardcoded.get("settings", {}), metrics)
        _render_info_page(pdf, _info_text, title=f"L={L} ADAPT-VQE Comparison: Settings & Metrics")

        # --- Page A: Fidelity + Energy ---
        figA, (axF, axE) = plt.subplots(1, 2, figsize=(11.0, 8.5), sharex=True)

        axF.plot(times, q("fidelity"), label="Qiskit fidelity", color="#0b3d91", marker="o", markersize=3, markevery=markevery)
        axF.plot(times, h("fidelity"), label="Hardcoded fidelity", color="#e15759", linestyle="--", marker="^", markersize=3, markevery=markevery)
        axF.set_title("Fidelity")
        axF.set_xlabel("Time")
        axF.grid(alpha=0.25)
        axF.legend(fontsize=8)
        _autozoom(axF, h("fidelity"), q("fidelity"))

        axE.plot(times, q("energy_trotter"), label="Qiskit trotter", color="#2ca02c", marker="s", markersize=3, markevery=markevery)
        axE.plot(times, h("energy_trotter"), label="Hardcoded trotter", color="#d62728", linestyle="--", marker="v", markersize=3, markevery=markevery)
        axE.plot(times, q("energy_exact"), label=EXACT_LABEL_QISKIT, color="#111111", linewidth=1.4)
        axE.plot(times, h("energy_exact"), label=EXACT_LABEL_HARDCODE, color="#7f7f7f", linestyle=":", linewidth=1.2)
        axE.set_title("Energy")
        axE.set_xlabel("Time")
        axE.grid(alpha=0.25)
        axE.legend(fontsize=8)
        _autozoom(axE, h("energy_trotter"), q("energy_trotter"), h("energy_exact"), q("energy_exact"))

        figA.suptitle(f"ADAPT-VQE Comparison L={L}: Fidelity & Energy", fontsize=13)
        figA.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(figA)
        plt.close(figA)

        # --- Page B: Occupations + Doublon ---
        figB, (axUp, axDn, axD) = plt.subplots(1, 3, figsize=(11.0, 8.5), sharex=True)

        axUp.plot(times, q("n_up_site0_trotter"), label="Qiskit trotter", color="#17becf")
        axUp.plot(times, h("n_up_site0_trotter"), label="Hardcoded trotter", color="#0f7f8b", linestyle="--")
        axUp.set_title("Site-0 n_up")
        axUp.set_xlabel("Time")
        axUp.grid(alpha=0.25)
        axUp.legend(fontsize=7)
        _autozoom(axUp, h("n_up_site0_trotter"), q("n_up_site0_trotter"))

        axDn.plot(times, q("n_dn_site0_trotter"), label="Qiskit trotter", color="#9467bd")
        axDn.plot(times, h("n_dn_site0_trotter"), label="Hardcoded trotter", color="#6f4d8f", linestyle="--")
        axDn.set_title("Site-0 n_dn")
        axDn.set_xlabel("Time")
        axDn.grid(alpha=0.25)
        axDn.legend(fontsize=7)
        _autozoom(axDn, h("n_dn_site0_trotter"), q("n_dn_site0_trotter"))

        axD.plot(times, q("doublon_trotter"), label="Qiskit trotter", color="#e377c2")
        axD.plot(times, h("doublon_trotter"), label="Hardcoded trotter", color="#c251a1", linestyle="--")
        axD.set_title("Doublon")
        axD.set_xlabel("Time")
        axD.grid(alpha=0.25)
        axD.legend(fontsize=7)
        _autozoom(axD, h("doublon_trotter"), q("doublon_trotter"))

        figB.suptitle(f"ADAPT-VQE Comparison L={L}: Occupations & Doublon", fontsize=13)
        figB.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(figB)
        plt.close(figB)

        # --- ADAPT-VQE energy bar chart ---
        gs_exact = float(hardcoded["ground_state"]["exact_energy"])
        hc_adapt_e = hardcoded.get("adapt_vqe", {}).get("energy")
        qk_adapt_e = qiskit.get("adapt_vqe", {}).get("energy")
        hc_val = float(hc_adapt_e) if hc_adapt_e is not None else np.nan
        qk_val = float(qk_adapt_e) if qk_adapt_e is not None else np.nan

        figv, axesv = plt.subplots(1, 2, figsize=(11.0, 5.0))
        vx0, vx1 = axesv[0], axesv[1]
        labels = ["Exact GS", "Hardcoded ADAPT", "Qiskit ADAPT"]
        vals = [gs_exact, hc_val, qk_val]
        colors = ["#111111", "#2ca02c", "#ff7f0e"]
        vx0.bar(np.arange(3), vals, color=colors, edgecolor="black", linewidth=0.4)
        vx0.set_xticks(np.arange(3))
        vx0.set_xticklabels(labels, rotation=18, ha="right")
        vx0.set_ylabel("Energy")
        vx0.set_title(f"L={L} ADAPT-VQE Energy")
        vx0.grid(axis="y", alpha=0.25)

        err_h = abs(hc_val - gs_exact) if np.isfinite(hc_val) else np.nan
        err_q = abs(qk_val - gs_exact) if np.isfinite(qk_val) else np.nan
        vx1.bar([0, 1], [err_h, err_q], color=["#2ca02c", "#ff7f0e"], edgecolor="black", linewidth=0.4)
        vx1.set_xticks([0, 1])
        vx1.set_xticklabels(["|HC ADAPT-Exact|", "|QK ADAPT-Exact|"], rotation=18, ha="right")
        vx1.set_ylabel("Absolute Error")
        vx1.set_title(f"L={L} ADAPT-VQE Absolute Error")
        vx1.grid(axis="y", alpha=0.25)

        figv.suptitle(
            "ADAPT-VQE converged energy vs exact ground state energy.\n"
            "Trotter dynamics start from each pipeline's ADAPT ground state.",
            fontsize=10,
        )
        figv.tight_layout(rect=(0.0, 0.03, 1.0, 0.91))
        pdf.savefig(figv)
        plt.close(figv)

        # --- Delta diagnostics ---
        fig2, axes2 = plt.subplots(2, 2, figsize=(11.0, 8.5))
        bx00, bx01 = axes2[0, 0], axes2[0, 1]
        bx10, bx11 = axes2[1, 0], axes2[1, 1]

        bx00.plot(times, np.abs(h("fidelity") - q("fidelity")), color="#1f77b4")
        bx00.set_title("|ΔF(t)|")
        bx00.set_xlabel("Time")
        bx00.grid(alpha=0.25)

        bx01.plot(times, np.abs(h("energy_trotter") - q("energy_trotter")), color="#d62728")
        bx01.set_title("|ΔE_trot(t)|")
        bx01.set_xlabel("Time")
        bx01.grid(alpha=0.25)

        bx10.plot(times, np.abs(h("n_up_site0_trotter") - q("n_up_site0_trotter")), label="|Δn_up0|", color="#17becf")
        bx10.plot(times, np.abs(h("n_dn_site0_trotter") - q("n_dn_site0_trotter")), label="|Δn_dn0|", color="#9467bd")
        bx10.set_title("|Δn_up0(t)| and |Δn_dn0(t)|")
        bx10.set_xlabel("Time")
        bx10.grid(alpha=0.25)
        bx10.legend(fontsize=8)

        bx11.plot(times, np.abs(h("doublon_trotter") - q("doublon_trotter")), color="#8c564b")
        bx11.set_title("|ΔD(t)|")
        bx11.set_xlabel("Time")
        bx11.grid(alpha=0.25)

        fig2.suptitle(f"Delta Diagnostics L={L}: Hardcoded ADAPT vs Qiskit ADAPT", fontsize=14)
        fig2.text(
            0.5, 0.93,
            "ΔX(t) = |X_hc(t) − X_qk(t)|, where X_pipeline(t) is that pipeline's stored trajectory value.",
            ha="center", fontsize=8, style="italic",
        )
        fig2.tight_layout(rect=(0.0, 0.02, 1.0, 0.91))
        pdf.savefig(fig2)
        plt.close(fig2)

        # Metrics text page
        td = metrics["trajectory_deltas"]
        lines = [
            f"L={L} ADAPT-VQE metrics summary",
            "",
            "Delta metric definitions:",
            *_delta_metric_definition_text().splitlines(),
            "",
            f"ground_state_energy_abs_delta = {_fp(metrics['ground_state_energy']['abs_delta'])}",
            f"fidelity max/mean/final = {_fp(td['fidelity']['max_abs_delta'])} / {_fp(td['fidelity']['mean_abs_delta'])} / {_fp(td['fidelity']['final_abs_delta'])}",
            f"energy_trotter max/mean/final = {_fp(td['energy_trotter']['max_abs_delta'])} / {_fp(td['energy_trotter']['mean_abs_delta'])} / {_fp(td['energy_trotter']['final_abs_delta'])}",
            f"n_up_site0_trotter max/mean/final = {_fp(td['n_up_site0_trotter']['max_abs_delta'])} / {_fp(td['n_up_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_up_site0_trotter']['final_abs_delta'])}",
            f"n_dn_site0_trotter max/mean/final = {_fp(td['n_dn_site0_trotter']['max_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['final_abs_delta'])}",
            f"doublon_trotter max/mean/final = {_fp(td['doublon_trotter']['max_abs_delta'])} / {_fp(td['doublon_trotter']['mean_abs_delta'])} / {_fp(td['doublon_trotter']['final_abs_delta'])}",
            "",
            "checks:",
            *_fmt_obj(metrics["acceptance"]["checks"]).splitlines(),
            "",
            f"PASS = {metrics['acceptance']['pass']}",
        ]
        _render_text_page(pdf, lines)


# ---------------------------------------------------------------------------
# Bundle PDF
# ---------------------------------------------------------------------------

def _write_bundle_pdf(
    *,
    bundle_path: Path,
    per_l_data: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]],
    overall_summary: dict[str, Any],
    include_per_l_pages: bool,
    run_command: str,
) -> None:
    lvals = [L for L, _h, _q, _m in per_l_data]
    exact_gs = np.array([float(h["ground_state"]["exact_energy"]) for _L, h, _q, _m in per_l_data], dtype=float)
    hc_adapt = np.array(
        [float(h["adapt_vqe"].get("energy", np.nan)) if h["adapt_vqe"].get("energy") is not None else np.nan for _L, h, _q, _m in per_l_data],
        dtype=float,
    )
    qk_adapt = np.array(
        [float(q["adapt_vqe"].get("energy", np.nan)) if q["adapt_vqe"].get("energy") is not None else np.nan for _L, _h, q, _m in per_l_data],
        dtype=float,
    )

    with PdfPages(str(bundle_path)) as pdf:
        _render_command_page(pdf, run_command)

        # Summary page
        lines = [
            "Hardcoded ADAPT-VQE vs Qiskit ADAPT-VQE Comparison Summary",
            "",
            f"generated_utc: {overall_summary['generated_utc']}",
            f"all_pass: {overall_summary['all_pass']}",
            f"l_values: {overall_summary['l_values']}",
            "",
            "trajectory_comparison_basis: trotter trajectories start from",
            "  each pipeline's ADAPT-VQE ground state (default: adapt_vqe)",
            f"exact_trajectory_labels: {EXACT_LABEL_HARDCODE}, {EXACT_LABEL_QISKIT}",
            f"exact_trajectory_method: {EXACT_METHOD}",
            "",
            "thresholds:",
            *_fmt_obj(THRESHOLDS).splitlines(),
            "",
            "Delta metric definitions:",
            *_delta_metric_definition_text().splitlines(),
            "",
            "Per-L pass flags:",
        ]
        for row in overall_summary["results"]:
            lines.append(f"L={row['L']} pass={row['pass']} metrics_json={row['metrics_json']}")
        _render_text_page(pdf, lines)

        # ADAPT-VQE energy comparison across L values
        x = np.arange(len(lvals), dtype=float)

        fig1, ax1 = plt.subplots(figsize=(11.0, 8.5))
        ax1.plot(x, exact_gs, marker="D", linewidth=2.0, color="#111111", label="Exact GS")
        ax1.plot(x, hc_adapt, marker="s", linewidth=1.8, color="#2ca02c", label="Hardcoded ADAPT-VQE")
        ax1.plot(x, qk_adapt, marker="o", linewidth=1.8, color="#ff7f0e", label="Qiskit ADAPT-VQE")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"L={L}" for L in lvals])
        ax1.set_ylabel("Energy")
        ax1.set_title("ADAPT-VQE Energy Comparison")
        ax1.grid(alpha=0.25)
        ax1.legend()
        pdf.savefig(fig1)
        plt.close(fig1)

        # Error bar chart
        fig2, axes2 = plt.subplots(1, 2, figsize=(11.0, 8.5))
        ax2, ax3 = axes2[0], axes2[1]
        width = 0.25
        ax2.bar(x - 0.5 * width, np.abs(hc_adapt - exact_gs), width=width, color="#2ca02c", label="|HC ADAPT - Exact|")
        ax2.bar(x + 0.5 * width, np.abs(qk_adapt - exact_gs), width=width, color="#ff7f0e", label="|QK ADAPT - Exact|")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"L={L}" for L in lvals])
        ax2.set_ylabel("Absolute Error")
        ax2.set_title("ADAPT-VQE Absolute Error (linear)")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(fontsize=8)

        ax3.plot(x, np.abs(hc_adapt - qk_adapt), marker="o", linewidth=1.8, color="#9467bd", label="|HC ADAPT - QK ADAPT|")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"L={L}" for L in lvals])
        ax3.set_ylabel("Absolute Delta")
        ax3.set_title("ADAPT-VQE Cross-Implementation Delta")
        ax3.grid(alpha=0.25)
        ax3.legend(fontsize=8)
        pdf.savefig(fig2)
        plt.close(fig2)

        # Include per-L trajectory pages
        if include_per_l_pages:
            for L, hardcoded, qiskit, metrics in per_l_data:
                _write_comparison_pages_into_pdf(pdf, L, hardcoded, qiskit, metrics)


def _write_comparison_pages_into_pdf(
    pdf: PdfPages,
    L: int,
    hardcoded: dict[str, Any],
    qiskit: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    h_rows = hardcoded["trajectory"]
    q_rows = qiskit["trajectory"]
    times = _arr(h_rows, "time")
    markevery = max(1, times.size // 25)

    def h(key: str) -> np.ndarray:
        return _arr(h_rows, key)

    def q(key: str) -> np.ndarray:
        return _arr(q_rows, key)

    _info_text = _build_info_box_text(hardcoded.get("settings", {}), metrics)
    _render_info_page(pdf, _info_text, title=f"Bundle L={L}: ADAPT-VQE Settings & Metrics")

    # Fidelity + Energy
    figA, (axF, axE) = plt.subplots(1, 2, figsize=(11.0, 8.5), sharex=True)
    axF.plot(times, q("fidelity"), label="Qiskit fidelity", color="#0b3d91", marker="o", markersize=3, markevery=markevery)
    axF.plot(times, h("fidelity"), label="Hardcoded fidelity", color="#e15759", linestyle="--", marker="^", markersize=3, markevery=markevery)
    axF.set_title(f"L={L} Fidelity")
    axF.set_xlabel("Time")
    axF.grid(alpha=0.25)
    axF.legend(fontsize=8)
    _autozoom(axF, h("fidelity"), q("fidelity"))

    axE.plot(times, q("energy_trotter"), label="Qiskit trotter", color="#2ca02c")
    axE.plot(times, h("energy_trotter"), label="Hardcoded trotter", color="#d62728", linestyle="--")
    axE.plot(times, q("energy_exact"), label=EXACT_LABEL_QISKIT, color="#111111", linewidth=1.2)
    axE.plot(times, h("energy_exact"), label=EXACT_LABEL_HARDCODE, color="#7f7f7f", linestyle=":", linewidth=1.2)
    axE.set_title(f"L={L} Energy")
    axE.set_xlabel("Time")
    axE.grid(alpha=0.25)
    axE.legend(fontsize=8)
    _autozoom(axE, h("energy_trotter"), q("energy_trotter"), h("energy_exact"), q("energy_exact"))

    figA.suptitle(f"Bundle: L={L} ADAPT-VQE Fidelity & Energy", fontsize=14)
    figA.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    pdf.savefig(figA)
    plt.close(figA)

    # Occupations + Doublon
    figB, (axUp, axDn, axD) = plt.subplots(1, 3, figsize=(11.0, 8.5), sharex=True)
    axUp.plot(times, q("n_up_site0_trotter"), label="Qiskit trotter", color="#17becf")
    axUp.plot(times, h("n_up_site0_trotter"), label="Hardcoded trotter", color="#0f7f8b", linestyle="--")
    axUp.set_title(f"L={L} Site-0 n_up")
    axUp.set_xlabel("Time")
    axUp.grid(alpha=0.25)
    axUp.legend(fontsize=7)
    _autozoom(axUp, h("n_up_site0_trotter"), q("n_up_site0_trotter"))

    axDn.plot(times, q("n_dn_site0_trotter"), label="Qiskit trotter", color="#9467bd")
    axDn.plot(times, h("n_dn_site0_trotter"), label="Hardcoded trotter", color="#6f4d8f", linestyle="--")
    axDn.set_title(f"L={L} Site-0 n_dn")
    axDn.set_xlabel("Time")
    axDn.grid(alpha=0.25)
    axDn.legend(fontsize=7)
    _autozoom(axDn, h("n_dn_site0_trotter"), q("n_dn_site0_trotter"))

    axD.plot(times, q("doublon_trotter"), label="Qiskit trotter", color="#e377c2")
    axD.plot(times, h("doublon_trotter"), label="Hardcoded trotter", color="#c251a1", linestyle="--")
    axD.set_title(f"L={L} Doublon")
    axD.set_xlabel("Time")
    axD.grid(alpha=0.25)
    axD.legend(fontsize=7)
    _autozoom(axD, h("doublon_trotter"), q("doublon_trotter"))

    figB.suptitle(f"Bundle: L={L} Occupations & Doublon", fontsize=13)
    figB.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    pdf.savefig(figB)
    plt.close(figB)

    # Delta diagnostics
    fig2, axes2 = plt.subplots(2, 2, figsize=(11.0, 8.5))
    bx00, bx01 = axes2[0, 0], axes2[0, 1]
    bx10, bx11 = axes2[1, 0], axes2[1, 1]

    bx00.plot(times, np.abs(h("fidelity") - q("fidelity")), color="#1f77b4")
    bx00.set_title("|ΔF(t)|")
    bx00.set_xlabel("Time")
    bx00.grid(alpha=0.25)

    bx01.plot(times, np.abs(h("energy_trotter") - q("energy_trotter")), color="#d62728")
    bx01.set_title("|ΔE_trot(t)|")
    bx01.set_xlabel("Time")
    bx01.grid(alpha=0.25)

    bx10.plot(times, np.abs(h("n_up_site0_trotter") - q("n_up_site0_trotter")), label="|Δn_up0|", color="#17becf")
    bx10.plot(times, np.abs(h("n_dn_site0_trotter") - q("n_dn_site0_trotter")), label="|Δn_dn0|", color="#9467bd")
    bx10.set_title("|Δn_up0(t)| and |Δn_dn0(t)|")
    bx10.set_xlabel("Time")
    bx10.grid(alpha=0.25)
    bx10.legend(fontsize=8)

    bx11.plot(times, np.abs(h("doublon_trotter") - q("doublon_trotter")), color="#8c564b")
    bx11.set_title("|ΔD(t)|")
    bx11.set_xlabel("Time")
    bx11.grid(alpha=0.25)

    fig2.suptitle(f"Bundle Delta Diagnostics L={L}", fontsize=14)
    fig2.text(
        0.5, 0.93,
        "ΔX(t) = |X_hc(t) − X_qk(t)|",
        ha="center", fontsize=8, style="italic",
    )
    fig2.tight_layout(rect=(0.0, 0.02, 1.0, 0.91))
    pdf.savefig(fig2)
    plt.close(fig2)

    # Metrics text page
    td = metrics["trajectory_deltas"]
    lines = [
        f"Bundle metrics page L={L}",
        "",
        f"ground_state_energy_abs_delta = {_fp(metrics['ground_state_energy']['abs_delta'])}",
        f"fidelity max/mean/final = {_fp(td['fidelity']['max_abs_delta'])} / {_fp(td['fidelity']['mean_abs_delta'])} / {_fp(td['fidelity']['final_abs_delta'])}",
        f"energy_trotter max/mean/final = {_fp(td['energy_trotter']['max_abs_delta'])} / {_fp(td['energy_trotter']['mean_abs_delta'])} / {_fp(td['energy_trotter']['final_abs_delta'])}",
        f"n_up_site0_trotter max/mean/final = {_fp(td['n_up_site0_trotter']['max_abs_delta'])} / {_fp(td['n_up_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_up_site0_trotter']['final_abs_delta'])}",
        f"n_dn_site0_trotter max/mean/final = {_fp(td['n_dn_site0_trotter']['max_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['mean_abs_delta'])} / {_fp(td['n_dn_site0_trotter']['final_abs_delta'])}",
        f"doublon_trotter max/mean/final = {_fp(td['doublon_trotter']['max_abs_delta'])} / {_fp(td['doublon_trotter']['mean_abs_delta'])} / {_fp(td['doublon_trotter']['final_abs_delta'])}",
        "",
        "checks:",
        *_fmt_obj(metrics["acceptance"]["checks"]).splitlines(),
        "",
        f"PASS = {metrics['acceptance']['pass']}",
    ]
    _render_text_page(pdf, lines)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_command(cmd: list[str]) -> tuple[int, str, str]:
    t0 = time.perf_counter()
    _ai_log("compare_subprocess_start", cmd=cmd)
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    _ai_log(
        "compare_subprocess_done",
        cmd=cmd,
        returncode=int(proc.returncode),
        elapsed_sec=round(time.perf_counter() - t0, 6),
        stdout_lines=int(len(proc.stdout.splitlines())),
        stderr_lines=int(len(proc.stderr.splitlines())),
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare hardcoded vs Qiskit ADAPT-VQE pipelines.")
    p.add_argument("--l-values", type=str, default="2,3")
    p.add_argument("--run-pipelines", action="store_true", default=True)
    p.add_argument("--no-run-pipelines", dest="run_pipelines", action="store_false")
    p.add_argument("--with-per-l-pdfs", action="store_true",
                    help="Emit standalone per-L comparison PDFs in addition to the bundle.")

    # Physics / lattice
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")

    # Trotter dynamics
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--num-times", type=int, default=201)
    p.add_argument("--suzuki-order", type=int, default=2)
    p.add_argument("--trotter-steps", type=int, default=64)

    # Hardcoded ADAPT controls
    p.add_argument("--hc-adapt-max-depth", type=int, default=20)
    p.add_argument("--hc-adapt-eps-grad", type=float, default=1e-4)
    p.add_argument("--hc-adapt-eps-energy", type=float, default=1e-8)
    p.add_argument("--hc-adapt-maxiter", type=int, default=300)
    p.add_argument("--hc-adapt-seed", type=int, default=7)
    p.add_argument("--hc-adapt-allow-repeats", action="store_true", default=True)
    p.add_argument("--hc-adapt-no-repeats", dest="hc_adapt_allow_repeats", action="store_false")

    # Qiskit ADAPT controls
    p.add_argument("--qk-adapt-max-iterations", type=int, default=20)
    p.add_argument("--qk-adapt-gradient-threshold", type=float, default=1e-4)
    p.add_argument("--qk-adapt-cobyla-maxiter", type=int, default=300)
    p.add_argument("--qk-adapt-seed", type=int, default=7)

    p.add_argument("--initial-state-source", choices=["exact", "adapt_vqe", "hf"], default="adapt_vqe")
    p.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("compare_adapt_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    l_values = [int(x.strip()) for x in str(args.l_values).split(",") if x.strip()]
    _ai_log("compare_adapt_l_values", l_values=l_values)

    command_log_path = artifacts_dir / "adapt_pipeline_commands_run.txt"
    command_log: list[str] = []
    run_artifacts: list[RunArtifacts] = []

    for L in l_values:
        _ai_log("compare_adapt_l_start", L=int(L))
        hardcoded_json = artifacts_dir / f"hardcoded_adapt_pipeline_L{L}.json"
        hardcoded_pdf = artifacts_dir / f"hardcoded_adapt_pipeline_L{L}.pdf"
        qiskit_json = artifacts_dir / f"qiskit_adapt_pipeline_L{L}.json"
        qiskit_pdf = artifacts_dir / f"qiskit_adapt_pipeline_L{L}.pdf"
        compare_metrics_json = artifacts_dir / f"hardcoded_vs_qiskit_adapt_L{L}_metrics.json"
        compare_pdf = artifacts_dir / f"hardcoded_vs_qiskit_adapt_L{L}_comparison.pdf"

        if args.run_pipelines:
            # --- Run hardcoded ADAPT pipeline ---
            hc_cmd = [
                sys.executable,
                "pipelines/hardcoded_adapt_pipeline.py",
                "--L", str(L),
                "--t", str(args.t),
                "--u", str(args.u),
                "--dv", str(args.dv),
                "--boundary", str(args.boundary),
                "--ordering", str(args.ordering),
                "--t-final", str(args.t_final),
                "--num-times", str(args.num_times),
                "--suzuki-order", str(args.suzuki_order),
                "--trotter-steps", str(args.trotter_steps),
                "--term-order", "sorted",
                "--adapt-max-depth", str(args.hc_adapt_max_depth),
                "--adapt-eps-grad", str(args.hc_adapt_eps_grad),
                "--adapt-eps-energy", str(args.hc_adapt_eps_energy),
                "--adapt-maxiter", str(args.hc_adapt_maxiter),
                "--adapt-seed", str(args.hc_adapt_seed),
                "--initial-state-source", str(args.initial_state_source),
                "--output-json", str(hardcoded_json),
                "--output-pdf", str(hardcoded_pdf),
                "--skip-pdf",
            ]
            if args.hc_adapt_allow_repeats:
                hc_cmd.append("--adapt-allow-repeats")
            else:
                hc_cmd.append("--adapt-no-repeats")
            command_log.append(" ".join(hc_cmd))
            code, out, err = _run_command(hc_cmd)
            if code != 0:
                raise RuntimeError(
                    f"Hardcoded ADAPT pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )
            _ai_log("compare_adapt_l_hardcoded_done", L=int(L), json_path=str(hardcoded_json))

            # --- Run Qiskit ADAPT pipeline ---
            qk_cmd = [
                sys.executable,
                "pipelines/qiskit_adapt_pipeline.py",
                "--L", str(L),
                "--t", str(args.t),
                "--u", str(args.u),
                "--dv", str(args.dv),
                "--boundary", str(args.boundary),
                "--ordering", str(args.ordering),
                "--t-final", str(args.t_final),
                "--num-times", str(args.num_times),
                "--suzuki-order", str(args.suzuki_order),
                "--trotter-steps", str(args.trotter_steps),
                "--term-order", "sorted",
                "--adapt-max-iterations", str(args.qk_adapt_max_iterations),
                "--adapt-gradient-threshold", str(args.qk_adapt_gradient_threshold),
                "--adapt-cobyla-maxiter", str(args.qk_adapt_cobyla_maxiter),
                "--adapt-seed", str(args.qk_adapt_seed),
                "--initial-state-source", str(args.initial_state_source),
                "--output-json", str(qiskit_json),
                "--output-pdf", str(qiskit_pdf),
                "--skip-pdf",
            ]
            command_log.append(" ".join(qk_cmd))
            code, out, err = _run_command(qk_cmd)
            if code != 0:
                raise RuntimeError(
                    f"Qiskit ADAPT pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )
            _ai_log("compare_adapt_l_qiskit_done", L=int(L), json_path=str(qiskit_json))

        run_artifacts.append(
            RunArtifacts(
                L=L,
                hardcoded_json=hardcoded_json,
                hardcoded_pdf=hardcoded_pdf,
                qiskit_json=qiskit_json,
                qiskit_pdf=qiskit_pdf,
                compare_metrics_json=compare_metrics_json,
                compare_pdf=compare_pdf,
            )
        )

    # Write command log
    if command_log:
        with command_log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(command_log) + "\n")
        _ai_log("compare_adapt_command_log_written", path=str(command_log_path))

    # Compare payloads
    per_l_data: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    results_rows: list[dict[str, Any]] = []

    for row in run_artifacts:
        if not row.hardcoded_json.exists():
            raise FileNotFoundError(f"Missing hardcoded ADAPT JSON for L={row.L}: {row.hardcoded_json}")
        if not row.qiskit_json.exists():
            raise FileNotFoundError(f"Missing Qiskit ADAPT JSON for L={row.L}: {row.qiskit_json}")

        hardcoded = json.loads(row.hardcoded_json.read_text(encoding="utf-8"))
        qiskit = json.loads(row.qiskit_json.read_text(encoding="utf-8"))
        metrics = _compare_payloads(hardcoded, qiskit)
        _ai_log(
            "compare_adapt_l_metrics",
            L=int(row.L),
            passed=bool(metrics["acceptance"]["pass"]),
            gs_abs_delta=float(metrics["ground_state_energy"]["abs_delta"]),
            fidelity_max_abs_delta=float(metrics["trajectory_deltas"]["fidelity"]["max_abs_delta"]),
            energy_trotter_max_abs_delta=float(metrics["trajectory_deltas"]["energy_trotter"]["max_abs_delta"]),
        )

        metrics_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "L": int(row.L),
            "hardcoded_json": str(row.hardcoded_json),
            "qiskit_json": str(row.qiskit_json),
            "metrics": metrics,
        }
        row.compare_metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        if args.with_per_l_pdfs:
            _write_comparison_pdf(
                pdf_path=row.compare_pdf,
                L=row.L,
                hardcoded=hardcoded,
                qiskit=qiskit,
                metrics=metrics,
                run_command=run_command,
            )

        per_l_data.append((row.L, hardcoded, qiskit, metrics))
        results_rows.append({
            "L": int(row.L),
            "pass": bool(metrics["acceptance"]["pass"]),
            "metrics_json": str(row.compare_metrics_json),
            "comparison_pdf": (str(row.compare_pdf) if args.with_per_l_pdfs else None),
            "hardcoded_settings": hardcoded.get("settings", {}),
            "qiskit_settings": qiskit.get("settings", {}),
            "ground_state_energy_abs_delta": float(metrics["ground_state_energy"]["abs_delta"]),
            "adapt_vqe_energy_delta": metrics.get("adapt_vqe_energy", {}).get("abs_delta"),
            "trajectory_max_abs_deltas": {
                key: float(metrics["trajectory_deltas"][key]["max_abs_delta"]) for key in TARGET_METRICS
            },
        })

    # Overall summary
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Hardcoded ADAPT-VQE vs Qiskit ADAPT-VQE pipeline comparison.",
        "l_values": l_values,
        "thresholds": THRESHOLDS,
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
            "initial_state_source": str(args.initial_state_source),
        },
        "results": results_rows,
        "all_pass": bool(all(r["pass"] for r in results_rows)),
    }

    summary_json = artifacts_dir / "hardcoded_vs_qiskit_adapt_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bundle_pdf = artifacts_dir / "hardcoded_vs_qiskit_adapt_bundle.pdf"
    _write_bundle_pdf(
        bundle_path=bundle_pdf,
        per_l_data=per_l_data,
        overall_summary=summary,
        include_per_l_pages=True,
        run_command=run_command,
    )

    _ai_log(
        "compare_adapt_main_done",
        summary_json=str(summary_json),
        bundle_pdf=str(bundle_pdf),
        all_pass=bool(summary["all_pass"]),
    )

    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote bundle PDF:   {bundle_pdf}")
    if command_log:
        print(f"Wrote command log:  {command_log_path}")
    for row in results_rows:
        print(f"L={row['L']}: pass={row['pass']} metrics={row['metrics_json']} pdf={row.get('comparison_pdf')}")


if __name__ == "__main__":
    main()
