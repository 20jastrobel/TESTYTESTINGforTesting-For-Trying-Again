from __future__ import annotations

import numpy as np

from pydephasing.quantum.ed_hubbard_holstein import (
    build_hh_sector_basis,
    build_hh_sector_hamiltonian_ed,
    encode_state_to_qubit_index,
    hermiticity_residual as ed_hermiticity_residual,
    matrix_to_dense as ed_matrix_to_dense,
)
from pydephasing.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_drive,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
    n_sites_from_dims,
)
from pydephasing.quantum.restrict_paulipoly import (
    matrix_to_dense as pp_matrix_to_dense,
    hermiticity_residual as pp_hermiticity_residual,
    restrict_pauli_polynomial_matrix,
)
from pydephasing.quantum.vqe_latex_python_pairs_test import hamiltonian_matrix


def _build_hh_poly(**kwargs):
    return build_hubbard_holstein_hamiltonian(
        **kwargs,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        boson_encoding="binary",
    )


def test_restricted_sector_projection_matches_exact_in_multiple_random_cases():
    rng = np.random.default_rng(2026)
    dims = 2
    n_ph_max = 1

    for _ in range(3):
        J = float(rng.uniform(0.2, 1.4))
        U = float(rng.uniform(-0.25, 1.8))
        omega0 = float(rng.uniform(0.1, 1.1))
        g = float(rng.uniform(-0.4, 0.4))

        H_poly = _build_hh_poly(
            dims=dims,
            J=J,
            U=U,
            omega0=omega0,
            g=g,
            n_ph_max=n_ph_max,
        )
        for num_particles in [(0, 2), (1, 1), (2, 0)]:
            basis = build_hh_sector_basis(
                dims=dims,
                n_ph_max=n_ph_max,
                num_particles=num_particles,
                indexing="blocked",
                boson_encoding="binary",
            )
            H_ed = build_hh_sector_hamiltonian_ed(
                dims=dims,
                J=J,
                U=U,
                omega0=omega0,
                g=g,
                n_ph_max=n_ph_max,
                num_particles=num_particles,
                indexing="blocked",
                boson_encoding="binary",
                pbc=False,
                include_zero_point=True,
                basis=basis,
                sparse=True,
            )
            H_pp = restrict_pauli_polynomial_matrix(
                H_poly,
                basis_indices=basis.basis_indices,
                sparse=True,
            )

            assert ed_hermiticity_residual(H_ed) < 1e-10
            assert pp_hermiticity_residual(H_pp) < 1e-10

            e_ed = np.linalg.eigvalsh(ed_matrix_to_dense(H_ed))
            e_pp = np.linalg.eigvalsh(pp_matrix_to_dense(H_pp))
            assert np.isclose(float(np.min(e_ed)), float(np.min(e_pp)), atol=1e-10)


def test_basis_round_trip_and_lexicographic_indexing():
    basis = build_hh_sector_basis(
        dims=2,
        n_ph_max=2,
        num_particles=(1, 1),
        indexing="blocked",
        boson_encoding="binary",
    )

    assert basis.basis_indices == sorted(basis.basis_indices)
    assert len(basis.basis_indices) == len(set(basis.basis_indices))
    assert all(key in basis.state_to_index for key in basis.state_to_index)

    for i in range(min(24, basis.dimension)):
        st = basis.basis_states[i]
        encoded = encode_state_to_qubit_index(
            fermion_bits=int(st.fermion_bits),
            phonons=st.phonons,
            n_sites=basis.n_sites,
            n_ph_max=basis.n_ph_max,
            qpb=basis.qpb,
        )
        assert basis.basis_indices[i] == encoded
        assert basis.state_to_index[(int(st.fermion_bits), tuple(st.phonons))] == i


def test_nph_zero_case_reduces_to_hubbard_embedding():
    dims = 2
    n_sites = n_sites_from_dims(dims)
    qpb = boson_qubits_per_site(0, "binary")
    nq_total = 2 * n_sites + n_sites * qpb

    H_hh = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=0.9,
        U=1.7,
        omega0=0.5,
        g=0.25,
        n_ph_max=0,
        boson_encoding="binary",
        include_zero_point=False,
        v_t=None,
        v0=None,
        t_eval=None,
        indexing="blocked",
        pbc=False,
    )
    H_hubb = build_hubbard_hamiltonian(
        dims=dims,
        t=0.9,
        U=1.7,
        v=None,
        indexing="blocked",
        nq_override=nq_total,
    )
    assert int(H_hh.return_polynomial()[0].nqubit()) == nq_total
    assert np.allclose(hamiltonian_matrix(H_hh), hamiltonian_matrix(H_hubb), atol=1e-12)


def test_custom_drive_from_time_function_and_static_arrays_match():
    def v_t(t: float):
        return (0.12 + 0.03 * float(t), -0.07 - 0.02 * float(t))

    H_t = build_hubbard_holstein_drive(
        dims=2,
        v_t=v_t,
        v0=[0.1, -0.1],
        t=0.75,
        repr_mode="JW",
        indexing="blocked",
    )
    H_expected = build_hubbard_holstein_drive(
        dims=2,
        v_t=[0.12 + 0.03 * 0.75, -0.07 - 0.02 * 0.75],
        v0=[0.1, -0.1],
        t=None,
        repr_mode="JW",
        indexing="blocked",
    )
    assert np.allclose(hamiltonian_matrix(H_t), hamiltonian_matrix(H_expected), atol=1e-12)


def test_custom_edge_override_is_respected():
    default_edges = [(0, 1), (1, 2)]
    alt_edges = [(0, 2)]

    H_default = build_hubbard_holstein_hamiltonian(
        dims=3,
        J=0.8,
        U=1.0,
        omega0=0.7,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        pbc=False,
        edges=None,
        indexing="blocked",
    )
    H_edges = build_hubbard_holstein_hamiltonian(
        dims=3,
        J=0.8,
        U=1.0,
        omega0=0.7,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        pbc=False,
        edges=default_edges,
        indexing="blocked",
    )
    H_alt = build_hubbard_holstein_hamiltonian(
        dims=3,
        J=0.8,
        U=1.0,
        omega0=0.7,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        pbc=False,
        edges=alt_edges,
        indexing="blocked",
    )

    Md = hamiltonian_matrix(H_default)
    Me = hamiltonian_matrix(H_edges)
    Ma = hamiltonian_matrix(H_alt)
    assert np.allclose(Md, Me, atol=1e-12)
    assert not np.allclose(Md, Ma, atol=1e-12)
