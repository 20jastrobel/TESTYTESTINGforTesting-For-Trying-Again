#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import spmatrix
except Exception:  # pragma: no cover - SciPy is optional
    spmatrix = None  # type: ignore[assignment]

try:
    from scipy.sparse.linalg import eigsh
except Exception:  # pragma: no cover - SciPy is optional
    eigsh = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydephasing.quantum.ed_hubbard_holstein import (  # noqa: E402
    HHEDBasis,
    build_hh_sector_basis,
    build_hh_sector_hamiltonian_ed,
    hermiticity_residual as ed_hermiticity_residual,
)
from pydephasing.quantum.hubbard_latex_python_pairs import (  # noqa: E402
    build_hubbard_holstein_hamiltonian,
)
from pydephasing.quantum.restrict_paulipoly import (  # noqa: E402
    hermiticity_residual as pp_hermiticity_residual,
    matrix_to_dense as pp_matrix_to_dense,
    restrict_pauli_polynomial_matrix,
)
from pydephasing.quantum.vqe_latex_python_pairs_test import (  # noqa: E402
    HubbardHolsteinLayerwiseAnsatz,
    exact_ground_energy_sector,
    hubbard_holstein_reference_state,
    vqe_minimize,
)


@dataclass(frozen=True)
class HHCase:
    n_sites: int
    n_ph_max: int
    J: float
    U: float
    omega0: float
    g: float
    delta_v: Tuple[float, ...]
    include_zero_point: bool = True


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _parse_bool(raw: str) -> bool:
    v = str(raw).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"cannot parse bool from '{raw}'")


def _default_sector_list(n_sites: int) -> List[Tuple[int, int]]:
    L = int(n_sites)
    half = ((L + 1) // 2, L // 2)
    cand = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        half,
        (L // 2, L // 2),
        (L, 0),
        (0, L),
    ]
    out: List[Tuple[int, int]] = []
    for (nu, nd) in cand:
        if 0 <= int(nu) <= L and 0 <= int(nd) <= L and (int(nu), int(nd)) not in out:
            out.append((int(nu), int(nd)))
    return out


def _parse_sector_list(raw: str, n_sites: int) -> List[Tuple[int, int]]:
    token = str(raw).strip().lower()
    if token in {"", "default"}:
        return _default_sector_list(n_sites)
    if token == "all":
        out: List[Tuple[int, int]] = []
        for nu in range(n_sites + 1):
            for nd in range(n_sites + 1):
                out.append((nu, nd))
        return out

    out: List[Tuple[int, int]] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            raise ValueError(f"invalid sector spec '{p}', expected N_up:N_dn")
        a, b = p.split(":", 1)
        nu, nd = int(a), int(b)
        if nu < 0 or nd < 0 or nu > n_sites or nd > n_sites:
            raise ValueError(f"sector ({nu},{nd}) invalid for n_sites={n_sites}")
        out.append((nu, nd))
    return out


def _parse_scan_ne(raw: str, n_sites: int) -> List[int]:
    token = str(raw).strip().lower()
    if token in {"", "none"}:
        return []
    out: List[int] = []
    for part in token.split(","):
        p = part.strip()
        if not p:
            continue
        if p == "half":
            out.append(int(n_sites))
        else:
            out.append(int(p))
    dedup: List[int] = []
    for ne in out:
        if 0 <= int(ne) <= 2 * int(n_sites) and int(ne) not in dedup:
            dedup.append(int(ne))
    return dedup


def _random_case(rng: np.random.Generator, n_sites: int, n_ph_max: int, drive_scale: float) -> HHCase:
    J = float(rng.uniform(0.2, 1.3))
    U = float(rng.uniform(0.0, 4.0))
    omega0 = float(rng.uniform(0.1, 1.4))
    g = float(rng.uniform(-1.0, 1.0))
    if abs(float(drive_scale)) > 1e-15:
        delta_v = tuple(float(x) for x in rng.uniform(-abs(float(drive_scale)), abs(float(drive_scale)), size=n_sites))
    else:
        delta_v = tuple(0.0 for _ in range(n_sites))
    return HHCase(
        n_sites=int(n_sites),
        n_ph_max=int(n_ph_max),
        J=J,
        U=U,
        omega0=omega0,
        g=g,
        delta_v=delta_v,
        include_zero_point=True,
    )


def _build_hh_pauli_polynomial(case: HHCase, *, indexing: str, pbc: bool):
    v_t = list(case.delta_v) if any(abs(x) > 1e-15 for x in case.delta_v) else None
    v0 = [0.0] * int(case.n_sites) if v_t is not None else None
    return build_hubbard_holstein_hamiltonian(
        dims=int(case.n_sites),
        J=float(case.J),
        U=float(case.U),
        omega0=float(case.omega0),
        g=float(case.g),
        n_ph_max=int(case.n_ph_max),
        boson_encoding="binary",
        v_t=v_t,
        v0=v0,
        t_eval=None,
        repr_mode="JW",
        indexing=str(indexing),
        pbc=bool(pbc),
        include_zero_point=bool(case.include_zero_point),
    )


def _lowest_k_eigs(h_mat: Any, k: int, *, use_sparse_eigsh: bool) -> np.ndarray:
    if int(k) <= 0:
        return np.array([], dtype=float)

    if (
        use_sparse_eigsh
        and eigsh is not None
        and spmatrix is not None
        and isinstance(h_mat, spmatrix)
        and h_mat.shape[0] > int(k)
    ):
        vals = eigsh(h_mat, k=int(k), which="SA", return_eigenvectors=False, tol=1e-10, maxiter=400000)
        vals = np.sort(np.real(vals))
        return np.asarray(vals, dtype=float)

    dense = pp_matrix_to_dense(h_mat)
    vals = np.linalg.eigvalsh(dense)
    return np.asarray(np.real(vals[: max(1, min(int(k), int(vals.size)))]), dtype=float)


def _run_ebqe(
    *,
    case: HHCase,
    h_poly: Any,
    indexing: str,
    pbc: bool,
    num_particles: Tuple[int, int],
    restarts: int,
    maxiter: int,
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    try:
        v_t = list(case.delta_v) if any(abs(x) > 1e-15 for x in case.delta_v) else None
        v0 = [0.0] * int(case.n_sites) if v_t is not None else None

        ansatz = HubbardHolsteinLayerwiseAnsatz(
            dims=int(case.n_sites),
            J=float(case.J),
            U=float(case.U),
            omega0=float(case.omega0),
            g=float(case.g),
            n_ph_max=int(case.n_ph_max),
            boson_encoding="binary",
            v=None,
            v_t=v_t,
            v0=v0,
            t_eval=None,
            reps=int(reps),
            repr_mode="JW",
            indexing=str(indexing),
            pbc=bool(pbc),
            include_zero_point=bool(case.include_zero_point),
        )
        psi_ref = hubbard_holstein_reference_state(
            dims=int(case.n_sites),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            n_ph_max=int(case.n_ph_max),
            boson_encoding="binary",
            indexing=str(indexing),
        )
        res = vqe_minimize(
            h_poly,
            ansatz,
            np.asarray(psi_ref, dtype=complex),
            restarts=int(restarts),
            seed=int(seed),
            maxiter=int(maxiter),
            method="SLSQP",
        )
        return {
            "success": True,
            "energy": float(res.energy),
            "nfev": int(res.nfev),
            "nit": int(res.nit),
            "best_restart": int(res.best_restart),
            "message": str(res.message),
        }
    except Exception as exc:  # pragma: no cover - runtime dependency/runtime-size dependent
        return {
            "success": False,
            "energy": None,
            "error": str(exc),
        }


def _sector_metrics(
    *,
    case: HHCase,
    h_poly: Any,
    basis: HHEDBasis,
    num_particles: Tuple[int, int],
    indexing: str,
    pbc: bool,
    k_lowest: int,
    use_sparse_eigsh: bool,
    e0_tol: float,
    herm_tol: float,
    run_ebqe_flag: bool,
    ebqe_restarts: int,
    ebqe_maxiter: int,
    ebqe_reps: int,
    ebqe_seed: int,
) -> Dict[str, Any]:
    h_ed = build_hh_sector_hamiltonian_ed(
        dims=int(case.n_sites),
        J=float(case.J),
        U=float(case.U),
        omega0=float(case.omega0),
        g=float(case.g),
        n_ph_max=int(case.n_ph_max),
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        indexing=str(indexing),
        boson_encoding="binary",
        pbc=bool(pbc),
        delta_v=list(case.delta_v),
        include_zero_point=bool(case.include_zero_point),
        basis=basis,
        sparse=True,
        return_basis=False,
    )
    h_pp = restrict_pauli_polynomial_matrix(
        h_poly,
        basis_indices=basis.basis_indices,
        sparse=True,
    )

    herm_ed = float(ed_hermiticity_residual(h_ed))
    herm_pp = float(pp_hermiticity_residual(h_pp))
    if herm_ed > float(herm_tol):
        raise AssertionError(f"ED matrix is not Hermitian within tolerance: {herm_ed} > {herm_tol}")
    if herm_pp > float(herm_tol):
        raise AssertionError(f"PP-restricted matrix is not Hermitian within tolerance: {herm_pp} > {herm_tol}")

    k_eval = max(1, int(k_lowest))
    evals_ed = _lowest_k_eigs(h_ed, k_eval, use_sparse_eigsh=bool(use_sparse_eigsh))
    evals_pp = _lowest_k_eigs(h_pp, k_eval, use_sparse_eigsh=bool(use_sparse_eigsh))

    e0_ed = float(evals_ed[0])
    e0_pp = float(evals_pp[0])
    abs_diff_e0 = float(abs(e0_pp - e0_ed))
    if abs_diff_e0 > float(e0_tol):
        raise AssertionError(
            f"E0 mismatch in sector {num_particles}: |E0_PP-E0_ED|={abs_diff_e0} > tol={e0_tol}"
        )

    k_match_abs_max = 0.0
    if int(k_lowest) > 1:
        k_cmp = min(int(len(evals_ed)), int(len(evals_pp)), int(k_lowest))
        if k_cmp > 0:
            k_match_abs_max = float(np.max(np.abs(evals_pp[:k_cmp] - evals_ed[:k_cmp])))
            if k_match_abs_max > float(e0_tol):
                raise AssertionError(
                    f"lowest-{k_cmp} mismatch in sector {num_particles}: {k_match_abs_max} > tol={e0_tol}"
                )

    e_exact_filtered = float(
        exact_ground_energy_sector(
            h_poly,
            num_sites=int(case.n_sites),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            indexing=str(indexing),
        )
    )

    ebqe = (
        _run_ebqe(
            case=case,
            h_poly=h_poly,
            indexing=str(indexing),
            pbc=bool(pbc),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            restarts=int(ebqe_restarts),
            maxiter=int(ebqe_maxiter),
            reps=int(ebqe_reps),
            seed=int(ebqe_seed),
        )
        if run_ebqe_flag
        else {"success": False, "energy": None, "skipped": True}
    )

    return {
        "sector": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        "dim_sector": int(basis.dimension),
        "hermiticity_ed": herm_ed,
        "hermiticity_pp": herm_pp,
        "e0_ed": e0_ed,
        "e0_pp_restricted": e0_pp,
        "abs_diff_pp_vs_ed": abs_diff_e0,
        "e0_exact_filtered_existing": e_exact_filtered,
        "abs_diff_exact_filtered_vs_ed": float(abs(e_exact_filtered - e0_ed)),
        "ebqe": ebqe,
        "lowest_ed": [float(x) for x in evals_ed.tolist()],
        "lowest_pp_restricted": [float(x) for x in evals_pp.tolist()],
        "lowest_abs_diff_max": float(k_match_abs_max),
    }


def _sector_splits_for_total_ne(n_sites: int, n_e: int) -> List[Tuple[int, int]]:
    L = int(n_sites)
    ne_i = int(n_e)
    out: List[Tuple[int, int]] = []
    for n_up in range(L + 1):
        n_dn = ne_i - n_up
        if 0 <= n_dn <= L:
            out.append((int(n_up), int(n_dn)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Hubbard-Holstein sector energies: "
            "ED physical basis vs restricted PauliPolynomial, "
            "plus existing exact-filtered and EBQE comparators."
        )
    )
    parser.add_argument("--n-sites", type=str, default="2,3,4")
    parser.add_argument("--n-ph-max", type=str, default="1,2,3")
    parser.add_argument("--random-cases", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--indexing", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--pbc", type=str, default="false")
    parser.add_argument("--drive-scale", type=float, default=0.0)
    parser.add_argument("--include-zero-point", type=str, default="true")

    parser.add_argument("--sectors", type=str, default="default")
    parser.add_argument("--scan-ne", type=str, default="half")
    parser.add_argument("--k-lowest", type=int, default=1)
    parser.add_argument("--use-sparse-eigsh", action="store_true")

    parser.add_argument("--e0-tol-dense", type=float, default=1e-8)
    parser.add_argument("--e0-tol-sparse", type=float, default=1e-6)
    parser.add_argument("--herm-tol", type=float, default=1e-9)

    parser.add_argument("--skip-ebqe", action="store_true")
    parser.add_argument("--ebqe-restarts", type=int, default=1)
    parser.add_argument("--ebqe-maxiter", type=int, default=40)
    parser.add_argument("--ebqe-reps", type=int, default=1)

    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    pbc = _parse_bool(args.pbc)
    include_zero_point = _parse_bool(args.include_zero_point)
    use_sparse_eigsh = bool(args.use_sparse_eigsh)
    e0_tol = float(args.e0_tol_sparse if use_sparse_eigsh else args.e0_tol_dense)
    herm_tol = float(args.herm_tol)

    n_sites_list = _parse_int_list(args.n_sites)
    n_ph_list = _parse_int_list(args.n_ph_max)

    all_cases: List[HHCase] = []
    for n_sites in n_sites_list:
        for n_ph in n_ph_list:
            if int(args.random_cases) <= 0:
                all_cases.append(
                    HHCase(
                        n_sites=int(n_sites),
                        n_ph_max=int(n_ph),
                        J=1.0,
                        U=2.0,
                        omega0=0.6,
                        g=0.3,
                        delta_v=tuple(0.0 for _ in range(int(n_sites))),
                        include_zero_point=include_zero_point,
                    )
                )
            else:
                for _ in range(int(args.random_cases)):
                    case = _random_case(rng, int(n_sites), int(n_ph), float(args.drive_scale))
                    all_cases.append(
                        HHCase(
                            n_sites=case.n_sites,
                            n_ph_max=case.n_ph_max,
                            J=case.J,
                            U=case.U,
                            omega0=case.omega0,
                            g=case.g,
                            delta_v=case.delta_v,
                            include_zero_point=include_zero_point,
                        )
                    )

    output: Dict[str, Any] = {
        "config": {
            "seed": int(args.seed),
            "indexing": str(args.indexing),
            "pbc": bool(pbc),
            "use_sparse_eigsh": bool(use_sparse_eigsh),
            "k_lowest": int(args.k_lowest),
            "e0_tolerance": float(e0_tol),
            "hermiticity_tolerance": float(herm_tol),
            "skip_ebqe": bool(args.skip_ebqe),
        },
        "cases": [],
    }

    global_max_pp_vs_ed = 0.0
    global_max_exact_filtered_vs_ed = 0.0

    for case_idx, case in enumerate(all_cases):
        print(
            f"[case {case_idx + 1}/{len(all_cases)}] "
            f"L={case.n_sites}, n_ph_max={case.n_ph_max}, "
            f"J={case.J:.4f}, U={case.U:.4f}, omega0={case.omega0:.4f}, g={case.g:.4f}"
        )

        h_poly = _build_hh_pauli_polynomial(case, indexing=str(args.indexing), pbc=bool(pbc))
        sectors = _parse_sector_list(str(args.sectors), int(case.n_sites))
        scan_ne_values = _parse_scan_ne(str(args.scan_ne), int(case.n_sites))

        sector_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        case_entry: Dict[str, Any] = {
            "case_index": int(case_idx),
            "params": asdict(case),
            "sectors": [],
            "fixed_ne_scan": [],
        }

        for sec_idx, (n_up, n_dn) in enumerate(sectors):
            basis = build_hh_sector_basis(
                dims=int(case.n_sites),
                n_ph_max=int(case.n_ph_max),
                num_particles=(int(n_up), int(n_dn)),
                indexing=str(args.indexing),
                boson_encoding="binary",
            )
            ebqe_seed = int(args.seed) + 1009 * case_idx + 67 * sec_idx
            metrics = _sector_metrics(
                case=case,
                h_poly=h_poly,
                basis=basis,
                num_particles=(int(n_up), int(n_dn)),
                indexing=str(args.indexing),
                pbc=bool(pbc),
                k_lowest=int(args.k_lowest),
                use_sparse_eigsh=bool(use_sparse_eigsh),
                e0_tol=float(e0_tol),
                herm_tol=float(herm_tol),
                run_ebqe_flag=not bool(args.skip_ebqe),
                ebqe_restarts=int(args.ebqe_restarts),
                ebqe_maxiter=int(args.ebqe_maxiter),
                ebqe_reps=int(args.ebqe_reps),
                ebqe_seed=ebqe_seed,
            )
            sector_cache[(int(n_up), int(n_dn))] = metrics
            case_entry["sectors"].append(metrics)
            global_max_pp_vs_ed = max(global_max_pp_vs_ed, float(metrics["abs_diff_pp_vs_ed"]))
            global_max_exact_filtered_vs_ed = max(
                global_max_exact_filtered_vs_ed, float(metrics["abs_diff_exact_filtered_vs_ed"])
            )
            print(
                f"  sector (N_up,N_dn)=({n_up},{n_dn}) "
                f"dim={metrics['dim_sector']} "
                f"|E0_PP-E0_ED|={metrics['abs_diff_pp_vs_ed']:.3e} "
                f"|E0_exactFiltered-E0_ED|={metrics['abs_diff_exact_filtered_vs_ed']:.3e}"
            )

        for n_e in scan_ne_values:
            splits = _sector_splits_for_total_ne(int(case.n_sites), int(n_e))
            for split in splits:
                if split in sector_cache:
                    continue
                basis = build_hh_sector_basis(
                    dims=int(case.n_sites),
                    n_ph_max=int(case.n_ph_max),
                    num_particles=(int(split[0]), int(split[1])),
                    indexing=str(args.indexing),
                    boson_encoding="binary",
                )
                metrics = _sector_metrics(
                    case=case,
                    h_poly=h_poly,
                    basis=basis,
                    num_particles=(int(split[0]), int(split[1])),
                    indexing=str(args.indexing),
                    pbc=bool(pbc),
                    k_lowest=int(args.k_lowest),
                    use_sparse_eigsh=bool(use_sparse_eigsh),
                    e0_tol=float(e0_tol),
                    herm_tol=float(herm_tol),
                    run_ebqe_flag=not bool(args.skip_ebqe),
                    ebqe_restarts=int(args.ebqe_restarts),
                    ebqe_maxiter=int(args.ebqe_maxiter),
                    ebqe_reps=int(args.ebqe_reps),
                    ebqe_seed=int(args.seed) + 1009 * case_idx + 5003 + int(split[0]) * 53 + int(split[1]) * 89,
                )
                sector_cache[split] = metrics
                global_max_pp_vs_ed = max(global_max_pp_vs_ed, float(metrics["abs_diff_pp_vs_ed"]))
                global_max_exact_filtered_vs_ed = max(
                    global_max_exact_filtered_vs_ed, float(metrics["abs_diff_exact_filtered_vs_ed"])
                )

            e0_ed_vals = [float(sector_cache[s]["e0_ed"]) for s in splits]
            e0_pp_vals = [float(sector_cache[s]["e0_pp_restricted"]) for s in splits]
            e0_exact_filtered_vals = [float(sector_cache[s]["e0_exact_filtered_existing"]) for s in splits]
            e0_ebqe_vals = [
                float(sector_cache[s]["ebqe"]["energy"])
                for s in splits
                if bool(sector_cache[s]["ebqe"].get("success")) and sector_cache[s]["ebqe"].get("energy") is not None
            ]

            min_ed = float(min(e0_ed_vals))
            min_pp = float(min(e0_pp_vals))
            min_exact_filtered = float(min(e0_exact_filtered_vals))
            min_ebqe = float(min(e0_ebqe_vals)) if e0_ebqe_vals else None
            abs_diff_scan = float(abs(min_pp - min_ed))
            if abs_diff_scan > float(e0_tol):
                raise AssertionError(
                    f"Fixed N_e={n_e} scan mismatch: |min_PP-min_ED|={abs_diff_scan} > tol={e0_tol}"
                )

            scan_row = {
                "n_e": int(n_e),
                "splits": [{"n_up": int(s[0]), "n_dn": int(s[1])} for s in splits],
                "min_e0_ed": min_ed,
                "min_e0_pp_restricted": min_pp,
                "abs_diff_pp_vs_ed": abs_diff_scan,
                "min_e0_exact_filtered_existing": min_exact_filtered,
                "abs_diff_exact_filtered_vs_ed": float(abs(min_exact_filtered - min_ed)),
                "min_e0_ebqe": min_ebqe,
            }
            case_entry["fixed_ne_scan"].append(scan_row)
            print(
                f"  fixed N_e={n_e}: "
                f"|min_PP-min_ED|={scan_row['abs_diff_pp_vs_ed']:.3e}, "
                f"|min_exactFiltered-min_ED|={scan_row['abs_diff_exact_filtered_vs_ed']:.3e}"
            )

        output["cases"].append(case_entry)

    output["summary"] = {
        "num_cases": int(len(all_cases)),
        "global_max_abs_diff_pp_vs_ed": float(global_max_pp_vs_ed),
        "global_max_abs_diff_exact_filtered_vs_ed": float(global_max_exact_filtered_vs_ed),
        "accepted_e0_tolerance": float(e0_tol),
    }

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote benchmark results to {out_path}")

    print("\nBenchmark completed successfully.")
    print(
        "Max |E0_PP_restricted - E0_ED| = "
        f"{float(output['summary']['global_max_abs_diff_pp_vs_ed']):.6e}"
    )
    print(
        "Max |E0_exact_filtered_existing - E0_ED| = "
        f"{float(output['summary']['global_max_abs_diff_exact_filtered_vs_ed']):.6e}"
    )


if __name__ == "__main__":
    main()
