#!/usr/bin/env python3
"""Compare hardcoded HVA vs hardcoded UCCSD vs Qiskit UCCSD for Hubbard runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

ROOT = Path(__file__).resolve().parents[1]

METHOD_KEYS = ["hva", "hardcoded_uccsd", "qiskit_uccsd"]
METHOD_LABELS = {
    "hva": "Hardcoded HVA",
    "hardcoded_uccsd": "Hardcoded UCCSD",
    "qiskit_uccsd": "Qiskit UCCSD",
}
METHOD_COLORS = {
    "hva": "#1f77b4",
    "hardcoded_uccsd": "#2ca02c",
    "qiskit_uccsd": "#ff7f0e",
}
PAIR_KEYS = [
    ("hva", "hardcoded_uccsd"),
    ("hva", "qiskit_uccsd"),
    ("hardcoded_uccsd", "qiskit_uccsd"),
]
TARGET_METRICS = [
    "fidelity",
    "energy_trotter",
    "n_up_site0_trotter",
    "n_dn_site0_trotter",
    "doublon_trotter",
]


@dataclass
class RunArtifacts:
    L: int
    hardcoded_hva_json: Path
    hardcoded_uccsd_json: Path
    qiskit_json: Path
    metrics_json: Path
    compare_pdf: Path


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def _first_crossing(times: np.ndarray, vals: np.ndarray, thr: float) -> float | None:
    idx = np.where(vals > thr)[0]
    if idx.size == 0:
        return None
    return float(times[int(idx[0])])


def _delta_stats(times: np.ndarray, vals: np.ndarray) -> dict[str, float | None]:
    return {
        "max_abs_delta": float(np.max(vals)),
        "mean_abs_delta": float(np.mean(vals)),
        "final_abs_delta": float(vals[-1]),
        "first_time_abs_delta_gt_1e-4": _first_crossing(times, vals, 1e-4),
        "first_time_abs_delta_gt_1e-3": _first_crossing(times, vals, 1e-3),
    }


def _extract_exact_filtered(payload: dict[str, Any]) -> float | None:
    vqe = payload.get("vqe", {})
    val = vqe.get("exact_filtered_energy")
    if val is not None:
        return float(val)
    gs = payload.get("ground_state", {})
    gs_val = gs.get("exact_energy")
    if gs_val is not None:
        return float(gs_val)
    return None


def _compare_payloads(
    *,
    hva: dict[str, Any],
    hardcoded_uccsd: dict[str, Any],
    qiskit_uccsd: dict[str, Any],
) -> dict[str, Any]:
    rows = {
        "hva": hva["trajectory"],
        "hardcoded_uccsd": hardcoded_uccsd["trajectory"],
        "qiskit_uccsd": qiskit_uccsd["trajectory"],
    }

    base_key = METHOD_KEYS[0]
    base_rows = rows[base_key]
    times = _arr(base_rows, "time")
    for method_key in METHOD_KEYS[1:]:
        cur = rows[method_key]
        if len(cur) != len(base_rows):
            raise ValueError(f"Trajectory length mismatch for {method_key}.")
        if not np.allclose(_arr(cur, "time"), times, atol=1e-12, rtol=0.0):
            raise ValueError(f"Time-grid mismatch for {method_key}.")

    exact_candidates = [
        _extract_exact_filtered(hva),
        _extract_exact_filtered(hardcoded_uccsd),
        _extract_exact_filtered(qiskit_uccsd),
    ]
    exact_filtered_energy = next((float(v) for v in exact_candidates if v is not None), None)

    vqe_energy: dict[str, float] = {}
    for method_key, payload in [("hva", hva), ("hardcoded_uccsd", hardcoded_uccsd), ("qiskit_uccsd", qiskit_uccsd)]:
        e = payload.get("vqe", {}).get("energy")
        vqe_energy[method_key] = float(e) if e is not None else float("nan")

    vqe_abs_error = {}
    vqe_ge_exact = {}
    for method_key in METHOD_KEYS:
        if exact_filtered_energy is None or not np.isfinite(vqe_energy[method_key]):
            vqe_abs_error[method_key] = float("nan")
            vqe_ge_exact[method_key] = False
            continue
        err = abs(vqe_energy[method_key] - exact_filtered_energy)
        vqe_abs_error[method_key] = float(err)
        vqe_ge_exact[method_key] = bool(vqe_energy[method_key] >= exact_filtered_energy - 1e-8)

    pairwise: dict[str, dict[str, dict[str, float | None]]] = {}
    for metric_key in TARGET_METRICS:
        pairwise[metric_key] = {}
        for a_key, b_key in PAIR_KEYS:
            deltas = np.abs(_arr(rows[a_key], metric_key) - _arr(rows[b_key], metric_key))
            pairwise[metric_key][f"{a_key}__vs__{b_key}"] = _delta_stats(times, deltas)

    vs_exact: dict[str, dict[str, dict[str, float | None]]] = {}
    for method_key in METHOD_KEYS:
        cur_rows = rows[method_key]
        fidelity_err = np.abs(1.0 - _arr(cur_rows, "fidelity"))
        e_err = np.abs(_arr(cur_rows, "energy_trotter") - _arr(cur_rows, "energy_exact"))
        nup_err = np.abs(_arr(cur_rows, "n_up_site0_trotter") - _arr(cur_rows, "n_up_site0_exact"))
        ndn_err = np.abs(_arr(cur_rows, "n_dn_site0_trotter") - _arr(cur_rows, "n_dn_site0_exact"))
        dbl_err = np.abs(_arr(cur_rows, "doublon_trotter") - _arr(cur_rows, "doublon_exact"))
        vs_exact[method_key] = {
            "fidelity_to_1": _delta_stats(times, fidelity_err),
            "energy_trotter_vs_exact": _delta_stats(times, e_err),
            "n_up_site0_trotter_vs_exact": _delta_stats(times, nup_err),
            "n_dn_site0_trotter_vs_exact": _delta_stats(times, ndn_err),
            "doublon_trotter_vs_exact": _delta_stats(times, dbl_err),
        }

    vqe_finite = {method_key: bool(np.isfinite(vqe_energy[method_key])) for method_key in METHOD_KEYS}
    exact_filtered_finite = bool(exact_filtered_energy is not None and np.isfinite(exact_filtered_energy))
    sanity = {
        "exact_filtered_finite": exact_filtered_finite,
        "vqe_energy_finite": vqe_finite,
        "vqe_energy_ge_exact_filtered_minus_1e-8": vqe_ge_exact,
    }
    sanity["all_pass"] = bool(
        sanity["exact_filtered_finite"]
        and all(sanity["vqe_energy_finite"].values())
        and all(sanity["vqe_energy_ge_exact_filtered_minus_1e-8"].values())
    )

    return {
        "time_grid": {
            "num_times": int(times.size),
            "t0": float(times[0]),
            "t_final": float(times[-1]),
            "dt": float(times[1] - times[0]) if times.size > 1 else 0.0,
        },
        "exact_filtered_energy": exact_filtered_energy,
        "vqe_energy": vqe_energy,
        "vqe_abs_error_vs_exact_filtered": vqe_abs_error,
        "trajectory_pairwise_deltas": pairwise,
        "trajectory_vs_exact_errors": vs_exact,
        "sanity": sanity,
    }


def _autozoom(ax: Any, *arrays: np.ndarray, pad_frac: float = 0.05) -> None:
    combined = np.concatenate([arr for arr in arrays if arr.size > 0])
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))
    span = hi - lo
    pad = span * pad_frac if span > 0 else 1e-8
    ax.set_ylim(lo - pad, hi + pad)


def _render_text_page(pdf: PdfPages, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.96
    for line in lines:
        ax.text(0.03, y, line, transform=ax.transAxes, va="top", ha="left", family="monospace", fontsize=9)
        y -= 0.03
        if y < 0.03:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(11.0, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.96
    pdf.savefig(fig)
    plt.close(fig)


def _write_comparison_pdf(
    *,
    pdf_path: Path,
    L: int,
    hva: dict[str, Any],
    hardcoded_uccsd: dict[str, Any],
    qiskit_uccsd: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    rows = {
        "hva": hva["trajectory"],
        "hardcoded_uccsd": hardcoded_uccsd["trajectory"],
        "qiskit_uccsd": qiskit_uccsd["trajectory"],
    }
    times = _arr(rows["hva"], "time")
    markevery = max(1, times.size // 25)
    exact_filtered = float(metrics["exact_filtered_energy"])

    def s(method_key: str, metric_key: str) -> np.ndarray:
        return _arr(rows[method_key], metric_key)

    with PdfPages(str(pdf_path)) as pdf:
        info_lines = [
            f"L={L} HVA vs UCCSD vs Qiskit-UCCSD",
            "",
            f"exact_filtered_energy={exact_filtered!r}",
            f"hva_vqe_energy={metrics['vqe_energy']['hva']!r}",
            f"hardcoded_uccsd_vqe_energy={metrics['vqe_energy']['hardcoded_uccsd']!r}",
            f"qiskit_uccsd_vqe_energy={metrics['vqe_energy']['qiskit_uccsd']!r}",
            "",
            "sanity:",
            f"  exact_filtered_finite={metrics['sanity']['exact_filtered_finite']}",
            f"  vqe_energy_finite={metrics['sanity']['vqe_energy_finite']}",
            f"  vqe_energy_ge_exact_filtered_minus_1e-8={metrics['sanity']['vqe_energy_ge_exact_filtered_minus_1e-8']}",
            f"  all_pass={metrics['sanity']['all_pass']}",
        ]
        _render_text_page(pdf, info_lines)

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 8.5))
        labels = ["Exact(filtered)", METHOD_LABELS["hva"], METHOD_LABELS["hardcoded_uccsd"], METHOD_LABELS["qiskit_uccsd"]]
        vals = [
            exact_filtered,
            float(metrics["vqe_energy"]["hva"]),
            float(metrics["vqe_energy"]["hardcoded_uccsd"]),
            float(metrics["vqe_energy"]["qiskit_uccsd"]),
        ]
        colors = ["#111111", METHOD_COLORS["hva"], METHOD_COLORS["hardcoded_uccsd"], METHOD_COLORS["qiskit_uccsd"]]
        ax1.bar(np.arange(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
        ax1.set_xticks(np.arange(len(vals)))
        ax1.set_xticklabels(labels, rotation=18, ha="right")
        ax1.set_ylabel("Energy")
        ax1.set_title("VQE Energy Comparison")
        ax1.grid(axis="y", alpha=0.25)

        err_vals = [
            float(metrics["vqe_abs_error_vs_exact_filtered"]["hva"]),
            float(metrics["vqe_abs_error_vs_exact_filtered"]["hardcoded_uccsd"]),
            float(metrics["vqe_abs_error_vs_exact_filtered"]["qiskit_uccsd"]),
        ]
        ax2.bar(np.arange(3), err_vals, color=[METHOD_COLORS[k] for k in METHOD_KEYS], edgecolor="black", linewidth=0.4)
        ax2.set_xticks(np.arange(3))
        ax2.set_xticklabels([METHOD_LABELS[k] for k in METHOD_KEYS], rotation=18, ha="right")
        ax2.set_ylabel("|E_vqe - E_exact(filtered)|")
        ax2.set_title("VQE Absolute Error")
        ax2.grid(axis="y", alpha=0.25)
        fig1.suptitle(f"L={L}: Ground-State Comparison", fontsize=13)
        fig1.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2, (axf, axe) = plt.subplots(1, 2, figsize=(11.0, 8.5), sharex=True)
        for method_key in METHOD_KEYS:
            axf.plot(
                times,
                s(method_key, "fidelity"),
                label=METHOD_LABELS[method_key],
                color=METHOD_COLORS[method_key],
                marker="o",
                markersize=3,
                markevery=markevery,
            )
            axe.plot(
                times,
                s(method_key, "energy_trotter"),
                label=f"{METHOD_LABELS[method_key]} trotter",
                color=METHOD_COLORS[method_key],
            )
            axe.plot(
                times,
                s(method_key, "energy_exact"),
                label=f"{METHOD_LABELS[method_key]} exact",
                color=METHOD_COLORS[method_key],
                linestyle="--",
                linewidth=1.0,
            )
        axf.set_title("Fidelity(t)")
        axf.set_xlabel("Time")
        axf.grid(alpha=0.25)
        axf.legend(fontsize=8)
        _autozoom(axf, *[s(method_key, "fidelity") for method_key in METHOD_KEYS])

        axe.set_title("Energy(t): trotter + exact")
        axe.set_xlabel("Time")
        axe.grid(alpha=0.25)
        axe.legend(fontsize=7)
        _autozoom(
            axe,
            *[s(method_key, "energy_trotter") for method_key in METHOD_KEYS],
            *[s(method_key, "energy_exact") for method_key in METHOD_KEYS],
        )
        fig2.suptitle(f"L={L}: Dynamics Overview", fontsize=13)
        fig2.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3, (ax_up, ax_dn, ax_db) = plt.subplots(1, 3, figsize=(11.0, 8.5), sharex=True)
        for method_key in METHOD_KEYS:
            ax_up.plot(times, s(method_key, "n_up_site0_trotter"), color=METHOD_COLORS[method_key], label=f"{METHOD_LABELS[method_key]} trotter")
            ax_up.plot(times, s(method_key, "n_up_site0_exact"), color=METHOD_COLORS[method_key], linestyle="--", linewidth=1.0, label=f"{METHOD_LABELS[method_key]} exact")
            ax_dn.plot(times, s(method_key, "n_dn_site0_trotter"), color=METHOD_COLORS[method_key], label=f"{METHOD_LABELS[method_key]} trotter")
            ax_dn.plot(times, s(method_key, "n_dn_site0_exact"), color=METHOD_COLORS[method_key], linestyle="--", linewidth=1.0, label=f"{METHOD_LABELS[method_key]} exact")
            ax_db.plot(times, s(method_key, "doublon_trotter"), color=METHOD_COLORS[method_key], label=f"{METHOD_LABELS[method_key]} trotter")
            ax_db.plot(times, s(method_key, "doublon_exact"), color=METHOD_COLORS[method_key], linestyle="--", linewidth=1.0, label=f"{METHOD_LABELS[method_key]} exact")

        ax_up.set_title("Site-0 n_up")
        ax_up.set_xlabel("Time")
        ax_up.grid(alpha=0.25)
        ax_up.legend(fontsize=6)
        _autozoom(ax_up, *[s(method_key, "n_up_site0_trotter") for method_key in METHOD_KEYS], *[s(method_key, "n_up_site0_exact") for method_key in METHOD_KEYS])

        ax_dn.set_title("Site-0 n_dn")
        ax_dn.set_xlabel("Time")
        ax_dn.grid(alpha=0.25)
        ax_dn.legend(fontsize=6)
        _autozoom(ax_dn, *[s(method_key, "n_dn_site0_trotter") for method_key in METHOD_KEYS], *[s(method_key, "n_dn_site0_exact") for method_key in METHOD_KEYS])

        ax_db.set_title("Doublon")
        ax_db.set_xlabel("Time")
        ax_db.grid(alpha=0.25)
        ax_db.legend(fontsize=6)
        _autozoom(ax_db, *[s(method_key, "doublon_trotter") for method_key in METHOD_KEYS], *[s(method_key, "doublon_exact") for method_key in METHOD_KEYS])

        fig3.suptitle(f"L={L}: Occupations & Doublon (trotter vs exact)", fontsize=13)
        fig3.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig3)
        plt.close(fig3)

        fig4, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]
        for a_key, b_key in PAIR_KEYS:
            pair_label = f"{METHOD_LABELS[a_key]} vs {METHOD_LABELS[b_key]}"
            ax00.plot(times, np.abs(s(a_key, "fidelity") - s(b_key, "fidelity")), label=pair_label)
            ax01.plot(times, np.abs(s(a_key, "energy_trotter") - s(b_key, "energy_trotter")), label=pair_label)
            ax10.plot(times, np.abs(s(a_key, "n_up_site0_trotter") - s(b_key, "n_up_site0_trotter")), label=pair_label)
            ax11.plot(times, np.abs(s(a_key, "doublon_trotter") - s(b_key, "doublon_trotter")), label=pair_label)

        ax00.set_title("Pairwise |Δ fidelity|")
        ax01.set_title("Pairwise |Δ energy_trotter|")
        ax10.set_title("Pairwise |Δ n_up_site0|")
        ax11.set_title("Pairwise |Δ doublon|")
        for ax in [ax00, ax01, ax10, ax11]:
            ax.set_xlabel("Time")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6)
        fig4.suptitle(f"L={L}: Pairwise Deltas", fontsize=13)
        fig4.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig4)
        plt.close(fig4)

        fig5, axes5 = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        bx00, bx01 = axes5[0, 0], axes5[0, 1]
        bx10, bx11 = axes5[1, 0], axes5[1, 1]
        for method_key in METHOD_KEYS:
            bx00.plot(times, np.abs(s(method_key, "energy_trotter") - s(method_key, "energy_exact")), color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
            bx01.plot(times, np.abs(s(method_key, "n_up_site0_trotter") - s(method_key, "n_up_site0_exact")), color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
            bx10.plot(times, np.abs(s(method_key, "n_dn_site0_trotter") - s(method_key, "n_dn_site0_exact")), color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
            bx11.plot(times, np.abs(s(method_key, "doublon_trotter") - s(method_key, "doublon_exact")), color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])

        bx00.set_title("|E_trotter - E_exact|")
        bx01.set_title("|n_up_trotter - n_up_exact|")
        bx10.set_title("|n_dn_trotter - n_dn_exact|")
        bx11.set_title("|doublon_trotter - doublon_exact|")
        for ax in [bx00, bx01, bx10, bx11]:
            ax.set_xlabel("Time")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig5.suptitle(f"L={L}: Per-Method Error vs Exact", fontsize=13)
        fig5.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig5)
        plt.close(fig5)


def _write_bundle_pdf(
    *,
    bundle_path: Path,
    per_l_data: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]],
    summary: dict[str, Any],
    include_per_l_pages: bool,
) -> None:
    lvals = [L for L, _hva, _hc, _qk, _m in per_l_data]
    x = np.arange(len(lvals), dtype=float)

    exact_filtered = np.array([float(metrics["exact_filtered_energy"]) for _L, _hva, _hc, _qk, metrics in per_l_data], dtype=float)
    hva_e = np.array([float(metrics["vqe_energy"]["hva"]) for _L, _hva, _hc, _qk, metrics in per_l_data], dtype=float)
    hc_e = np.array([float(metrics["vqe_energy"]["hardcoded_uccsd"]) for _L, _hva, _hc, _qk, metrics in per_l_data], dtype=float)
    qk_e = np.array([float(metrics["vqe_energy"]["qiskit_uccsd"]) for _L, _hva, _hc, _qk, metrics in per_l_data], dtype=float)

    with PdfPages(str(bundle_path)) as pdf:
        lines = [
            "HVA vs UCCSD vs Qiskit-UCCSD Summary",
            "",
            f"generated_utc: {summary['generated_utc']}",
            f"all_pass: {summary['all_pass']}",
            f"l_values: {summary['l_values']}",
            "",
            "Per-L results:",
        ]
        for row in summary["results"]:
            lines.append(f"  L={row['L']} pass={row['pass']} metrics={row['metrics_json']}")
        _render_text_page(pdf, lines)

        fig1, ax1 = plt.subplots(figsize=(11.0, 8.5))
        ax1.plot(x, exact_filtered, marker="D", linewidth=2.0, color="#111111", label="Exact(filtered)")
        ax1.plot(x, hva_e, marker="o", linewidth=1.8, color=METHOD_COLORS["hva"], label=METHOD_LABELS["hva"])
        ax1.plot(x, hc_e, marker="s", linewidth=1.8, color=METHOD_COLORS["hardcoded_uccsd"], label=METHOD_LABELS["hardcoded_uccsd"])
        ax1.plot(x, qk_e, marker="^", linewidth=1.8, color=METHOD_COLORS["qiskit_uccsd"], label=METHOD_LABELS["qiskit_uccsd"])
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"L={L}" for L in lvals])
        ax1.set_ylabel("Energy")
        ax1.set_title("VQE Energy Comparison Across L")
        ax1.grid(alpha=0.25)
        ax1.legend()
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(11.0, 8.5))
        width = 0.23
        ax2.bar(x - width, np.abs(hva_e - exact_filtered), width=width, color=METHOD_COLORS["hva"], label="|HVA-Exact|")
        ax2.bar(x, np.abs(hc_e - exact_filtered), width=width, color=METHOD_COLORS["hardcoded_uccsd"], label="|HC-UCCSD-Exact|")
        ax2.bar(x + width, np.abs(qk_e - exact_filtered), width=width, color=METHOD_COLORS["qiskit_uccsd"], label="|Qiskit-UCCSD-Exact|")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"L={L}" for L in lvals])
        ax2.set_ylabel("Absolute Error")
        ax2.set_title("VQE Error vs Exact(filtered)")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(fontsize=8)

        ax3.plot(x, np.abs(hva_e - hc_e), marker="o", linewidth=1.8, color="#9467bd", label="|HVA - HC-UCCSD|")
        ax3.plot(x, np.abs(hva_e - qk_e), marker="s", linewidth=1.8, color="#8c564b", label="|HVA - Qiskit-UCCSD|")
        ax3.plot(x, np.abs(hc_e - qk_e), marker="^", linewidth=1.8, color="#17becf", label="|HC-UCCSD - Qiskit-UCCSD|")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"L={L}" for L in lvals])
        ax3.set_ylabel("Absolute Delta")
        ax3.set_title("Cross-Method VQE Delta")
        ax3.grid(alpha=0.25)
        ax3.legend(fontsize=8)
        pdf.savefig(fig2)
        plt.close(fig2)

        if include_per_l_pages:
            for L, hva, hardcoded_uccsd, qiskit_uccsd, metrics in per_l_data:
                _write_comparison_pages_into_pdf(
                    pdf=pdf,
                    L=L,
                    hva=hva,
                    hardcoded_uccsd=hardcoded_uccsd,
                    qiskit_uccsd=qiskit_uccsd,
                    metrics=metrics,
                )


def _write_comparison_pages_into_pdf(
    *,
    pdf: PdfPages,
    L: int,
    hva: dict[str, Any],
    hardcoded_uccsd: dict[str, Any],
    qiskit_uccsd: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    # Render the same set of pages as the standalone report without reopening another PdfPages.
    rows = {
        "hva": hva["trajectory"],
        "hardcoded_uccsd": hardcoded_uccsd["trajectory"],
        "qiskit_uccsd": qiskit_uccsd["trajectory"],
    }
    times = _arr(rows["hva"], "time")
    markevery = max(1, times.size // 25)
    exact_filtered = float(metrics["exact_filtered_energy"])

    def s(method_key: str, metric_key: str) -> np.ndarray:
        return _arr(rows[method_key], metric_key)

    info_lines = [
        f"Bundle L={L} HVA vs UCCSD vs Qiskit-UCCSD",
        "",
        f"exact_filtered_energy={exact_filtered!r}",
        f"hva_vqe_energy={metrics['vqe_energy']['hva']!r}",
        f"hardcoded_uccsd_vqe_energy={metrics['vqe_energy']['hardcoded_uccsd']!r}",
        f"qiskit_uccsd_vqe_energy={metrics['vqe_energy']['qiskit_uccsd']!r}",
        f"all_pass={metrics['sanity']['all_pass']}",
    ]
    _render_text_page(pdf, info_lines)

    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    for method_key in METHOD_KEYS:
        ax.plot(
            times,
            s(method_key, "energy_trotter"),
            label=f"{METHOD_LABELS[method_key]} trotter",
            color=METHOD_COLORS[method_key],
            marker="o",
            markersize=3,
            markevery=markevery,
        )
        ax.plot(
            times,
            s(method_key, "energy_exact"),
            label=f"{METHOD_LABELS[method_key]} exact",
            color=METHOD_COLORS[method_key],
            linestyle="--",
            linewidth=1.0,
        )
    ax.set_title(f"L={L}: Energy(t)")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _autozoom(
        ax,
        *[s(method_key, "energy_trotter") for method_key in METHOD_KEYS],
        *[s(method_key, "energy_exact") for method_key in METHOD_KEYS],
    )
    pdf.savefig(fig)
    plt.close(fig)


def _run_command(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hardcoded HVA/UCCSD and Qiskit UCCSD pipeline outputs.")
    parser.add_argument("--l-values", type=str, default="2,3")
    parser.add_argument("--run-pipelines", action="store_true", default=True)
    parser.add_argument("--no-run-pipelines", dest="run_pipelines", action="store_false")
    parser.add_argument("--with-per-l-pdfs", action="store_true")

    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--dv", type=float, default=0.0)
    parser.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--suzuki-order", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=64)

    parser.add_argument("--hardcoded-vqe-reps", type=int, default=2)
    parser.add_argument("--hardcoded-vqe-restarts", type=int, default=3)
    parser.add_argument("--hardcoded-vqe-seed", type=int, default=7)
    parser.add_argument("--hardcoded-vqe-maxiter", type=int, default=600)

    parser.add_argument("--qiskit-vqe-reps", type=int, default=2)
    parser.add_argument("--qiskit-vqe-restarts", type=int, default=3)
    parser.add_argument("--qiskit-vqe-seed", type=int, default=7)
    parser.add_argument("--qiskit-vqe-maxiter", type=int, default=600)

    parser.add_argument("--qpe-eval-qubits", type=int, default=5)
    parser.add_argument("--qpe-shots", type=int, default=256)
    parser.add_argument("--qpe-seed", type=int, default=11)
    parser.add_argument("--skip-qpe", action="store_true")

    parser.add_argument("--initial-state-source", choices=["exact", "vqe", "hf"], default="vqe")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    l_values = [int(x.strip()) for x in str(args.l_values).split(",") if x.strip()]
    command_log_path = artifacts_dir / "hva_uccsd_qiskit_commands_run.txt"
    command_log: list[str] = []

    run_rows: list[RunArtifacts] = []
    for L in l_values:
        hardcoded_hva_json = artifacts_dir / f"hardcoded_hva_pipeline_L{L}.json"
        hardcoded_uccsd_json = artifacts_dir / f"hardcoded_uccsd_pipeline_L{L}.json"
        qiskit_json = artifacts_dir / f"qiskit_pipeline_L{L}.json"
        metrics_json = artifacts_dir / f"hva_uccsd_qiskit_L{L}_metrics.json"
        compare_pdf = artifacts_dir / f"hva_uccsd_qiskit_L{L}_comparison.pdf"

        if args.run_pipelines:
            common_cmd = [
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
                "--initial-state-source",
                str(args.initial_state_source),
                "--qpe-eval-qubits",
                str(args.qpe_eval_qubits),
                "--qpe-shots",
                str(args.qpe_shots),
                "--qpe-seed",
                str(args.qpe_seed),
                "--skip-pdf",
            ]
            if args.skip_qpe:
                common_cmd.append("--skip-qpe")

            hc_u_cmd = [
                sys.executable,
                "pipelines/hardcoded_hubbard_pipeline.py",
                *common_cmd,
                "--vqe-ansatz",
                "uccsd",
                "--vqe-reps",
                str(args.hardcoded_vqe_reps),
                "--vqe-restarts",
                str(args.hardcoded_vqe_restarts),
                "--vqe-seed",
                str(args.hardcoded_vqe_seed),
                "--vqe-maxiter",
                str(args.hardcoded_vqe_maxiter),
                "--output-json",
                str(hardcoded_uccsd_json),
                "--output-pdf",
                str(artifacts_dir / f"hardcoded_uccsd_pipeline_L{L}.pdf"),
            ]
            command_log.append(" ".join(hc_u_cmd))
            code, out, err = _run_command(hc_u_cmd)
            if code != 0:
                raise RuntimeError(f"Hardcoded UCCSD pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}")

            hc_h_cmd = [
                sys.executable,
                "pipelines/hardcoded_hubbard_pipeline.py",
                *common_cmd,
                "--vqe-ansatz",
                "hva",
                "--vqe-reps",
                str(args.hardcoded_vqe_reps),
                "--vqe-restarts",
                str(args.hardcoded_vqe_restarts),
                "--vqe-seed",
                str(args.hardcoded_vqe_seed),
                "--vqe-maxiter",
                str(args.hardcoded_vqe_maxiter),
                "--output-json",
                str(hardcoded_hva_json),
                "--output-pdf",
                str(artifacts_dir / f"hardcoded_hva_pipeline_L{L}.pdf"),
            ]
            command_log.append(" ".join(hc_h_cmd))
            code, out, err = _run_command(hc_h_cmd)
            if code != 0:
                raise RuntimeError(f"Hardcoded HVA pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}")

            qk_cmd = [
                sys.executable,
                "pipelines/qiskit_hubbard_baseline_pipeline.py",
                *common_cmd,
                "--vqe-reps",
                str(args.qiskit_vqe_reps),
                "--vqe-restarts",
                str(args.qiskit_vqe_restarts),
                "--vqe-seed",
                str(args.qiskit_vqe_seed),
                "--vqe-maxiter",
                str(args.qiskit_vqe_maxiter),
                "--output-json",
                str(qiskit_json),
                "--output-pdf",
                str(artifacts_dir / f"qiskit_pipeline_L{L}.pdf"),
            ]
            command_log.append(" ".join(qk_cmd))
            code, out, err = _run_command(qk_cmd)
            if code != 0:
                raise RuntimeError(f"Qiskit UCCSD pipeline failed for L={L}.\nSTDOUT:\n{out}\nSTDERR:\n{err}")

        run_rows.append(
            RunArtifacts(
                L=L,
                hardcoded_hva_json=hardcoded_hva_json,
                hardcoded_uccsd_json=hardcoded_uccsd_json,
                qiskit_json=qiskit_json,
                metrics_json=metrics_json,
                compare_pdf=compare_pdf,
            )
        )

    if command_log:
        with command_log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(command_log) + "\n")

    per_l_data: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    for row in run_rows:
        for p in [row.hardcoded_hva_json, row.hardcoded_uccsd_json, row.qiskit_json]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required JSON: {p}")

        hva = json.loads(row.hardcoded_hva_json.read_text(encoding="utf-8"))
        hardcoded_uccsd = json.loads(row.hardcoded_uccsd_json.read_text(encoding="utf-8"))
        qiskit_uccsd = json.loads(row.qiskit_json.read_text(encoding="utf-8"))
        metrics = _compare_payloads(hva=hva, hardcoded_uccsd=hardcoded_uccsd, qiskit_uccsd=qiskit_uccsd)

        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "L": int(row.L),
            "hardcoded_hva_json": str(row.hardcoded_hva_json),
            "hardcoded_uccsd_json": str(row.hardcoded_uccsd_json),
            "qiskit_uccsd_json": str(row.qiskit_json),
            "metrics": metrics,
        }
        row.metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if args.with_per_l_pdfs:
            _write_comparison_pdf(
                pdf_path=row.compare_pdf,
                L=row.L,
                hva=hva,
                hardcoded_uccsd=hardcoded_uccsd,
                qiskit_uccsd=qiskit_uccsd,
                metrics=metrics,
            )

        per_l_data.append((row.L, hva, hardcoded_uccsd, qiskit_uccsd, metrics))
        results.append(
            {
                "L": int(row.L),
                "pass": bool(metrics["sanity"]["all_pass"]),
                "metrics_json": str(row.metrics_json),
                "comparison_pdf": str(row.compare_pdf) if args.with_per_l_pdfs else None,
                "exact_filtered_energy": float(metrics["exact_filtered_energy"]),
                "vqe_energy": metrics["vqe_energy"],
            }
        )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Hardcoded HVA vs hardcoded UCCSD vs Qiskit UCCSD comparison.",
        "l_values": l_values,
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
            "hardcoded_vqe_reps": int(args.hardcoded_vqe_reps),
            "hardcoded_vqe_restarts": int(args.hardcoded_vqe_restarts),
            "hardcoded_vqe_seed": int(args.hardcoded_vqe_seed),
            "hardcoded_vqe_maxiter": int(args.hardcoded_vqe_maxiter),
            "qiskit_vqe_reps": int(args.qiskit_vqe_reps),
            "qiskit_vqe_restarts": int(args.qiskit_vqe_restarts),
            "qiskit_vqe_seed": int(args.qiskit_vqe_seed),
            "qiskit_vqe_maxiter": int(args.qiskit_vqe_maxiter),
            "skip_qpe": bool(args.skip_qpe),
            "initial_state_source": str(args.initial_state_source),
        },
        "results": results,
        "all_pass": bool(all(row["pass"] for row in results)),
    }

    summary_json = artifacts_dir / "hva_uccsd_qiskit_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bundle_pdf = artifacts_dir / "hva_uccsd_qiskit_bundle.pdf"
    _write_bundle_pdf(
        bundle_path=bundle_pdf,
        per_l_data=per_l_data,
        summary=summary,
        include_per_l_pages=bool(args.with_per_l_pdfs),
    )

    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote bundle PDF:   {bundle_pdf}")
    if command_log:
        print(f"Wrote command log:  {command_log_path}")
    for row in results:
        print(f"L={row['L']}: pass={row['pass']} metrics={row['metrics_json']} pdf={row['comparison_pdf']}")


if __name__ == "__main__":
    main()
