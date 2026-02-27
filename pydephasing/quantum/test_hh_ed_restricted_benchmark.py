import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydephasing.quantum.ed_hubbard_holstein import (
    build_hh_sector_basis,
    build_hh_sector_hamiltonian_ed,
    hermiticity_residual as ed_hermiticity_residual,
    matrix_to_dense as ed_matrix_to_dense,
)
from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from pydephasing.quantum.restrict_paulipoly import (
    hermiticity_residual as pp_hermiticity_residual,
    matrix_to_dense as pp_matrix_to_dense,
    restrict_pauli_polynomial_matrix,
)


def _build_poly(
    *,
    dims: int,
    J: float,
    U: float,
    omega0: float,
    g: float,
    n_ph_max: int,
):
    return build_hubbard_holstein_hamiltonian(
        dims=int(dims),
        J=float(J),
        U=float(U),
        omega0=float(omega0),
        g=float(g),
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        v_t=None,
        v0=None,
        t_eval=None,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )


def test_restricted_pp_matches_direct_ed_sector_matrix():
    dims = 2
    n_ph_max = 2
    num_particles = (1, 1)

    H_poly = _build_poly(dims=dims, J=0.9, U=2.2, omega0=0.7, g=0.4, n_ph_max=n_ph_max)
    basis = build_hh_sector_basis(
        dims=dims,
        n_ph_max=n_ph_max,
        num_particles=num_particles,
        indexing="blocked",
        boson_encoding="binary",
    )
    H_ed = build_hh_sector_hamiltonian_ed(
        dims=dims,
        J=0.9,
        U=2.2,
        omega0=0.7,
        g=0.4,
        n_ph_max=n_ph_max,
        num_particles=num_particles,
        indexing="blocked",
        boson_encoding="binary",
        pbc=False,
        include_zero_point=True,
        basis=basis,
        sparse=True,
    )
    H_pp = restrict_pauli_polynomial_matrix(H_poly, basis_indices=basis.basis_indices, sparse=True)

    assert ed_hermiticity_residual(H_ed) < 1e-10
    assert pp_hermiticity_residual(H_pp) < 1e-10

    M_ed = ed_matrix_to_dense(H_ed)
    M_pp = pp_matrix_to_dense(H_pp)
    assert np.allclose(M_ed, M_pp, atol=1e-10)

    e_ed = np.linalg.eigvalsh(M_ed)
    e_pp = np.linalg.eigvalsh(M_pp)
    assert np.allclose(e_ed, e_pp, atol=1e-10)


def test_fixed_ne_scan_min_matches_between_ed_and_restricted_pp():
    dims = 2
    n_ph_max = 2
    H_poly = _build_poly(dims=dims, J=1.0, U=1.8, omega0=0.5, g=-0.35, n_ph_max=n_ph_max)

    splits = [(0, 2), (1, 1), (2, 0)]
    min_ed = None
    min_pp = None
    for num_particles in splits:
        basis = build_hh_sector_basis(
            dims=dims,
            n_ph_max=n_ph_max,
            num_particles=num_particles,
            indexing="blocked",
            boson_encoding="binary",
        )
        H_ed = build_hh_sector_hamiltonian_ed(
            dims=dims,
            J=1.0,
            U=1.8,
            omega0=0.5,
            g=-0.35,
            n_ph_max=n_ph_max,
            num_particles=num_particles,
            indexing="blocked",
            boson_encoding="binary",
            pbc=False,
            include_zero_point=True,
            basis=basis,
            sparse=True,
        )
        H_pp = restrict_pauli_polynomial_matrix(H_poly, basis_indices=basis.basis_indices, sparse=True)

        e0_ed = float(np.min(np.linalg.eigvalsh(ed_matrix_to_dense(H_ed))))
        e0_pp = float(np.min(np.linalg.eigvalsh(pp_matrix_to_dense(H_pp))))

        min_ed = e0_ed if min_ed is None else min(min_ed, e0_ed)
        min_pp = e0_pp if min_pp is None else min(min_pp, e0_pp)

    assert min_ed is not None and min_pp is not None
    assert abs(float(min_ed) - float(min_pp)) < 1e-10
