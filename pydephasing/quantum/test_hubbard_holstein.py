import numpy as np
import pytest

from pydephasing.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_holstein_coupling,
    build_holstein_phonon_energy,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_drive,
    build_hubbard_holstein_hamiltonian,
    n_sites_from_dims,
    phonon_qubit_indices_for_site,
)
from pydephasing.quantum.vqe_latex_python_pairs_test import hamiltonian_matrix


def _nq_of(H) -> int:
    terms = H.return_polynomial()
    if not terms:
        return 0
    return int(terms[0].nqubit())


def _assert_hermitian(H, atol: float = 1e-10) -> None:
    M = hamiltonian_matrix(H)
    assert np.allclose(M, M.conj().T, atol=atol)


def test_hubbard_builders_nq_override_backward_compatibility():
    dims = 2
    H_ref = build_hubbard_hamiltonian(dims=dims, t=1.0, U=2.0, v=[0.2, -0.1], pbc=False)
    H_none = build_hubbard_hamiltonian(
        dims=dims,
        t=1.0,
        U=2.0,
        v=[0.2, -0.1],
        pbc=False,
        nq_override=None,
    )
    M_ref = hamiltonian_matrix(H_ref)
    M_none = hamiltonian_matrix(H_none)
    assert np.allclose(M_ref, M_none, atol=1e-12)

    H_big = build_hubbard_hamiltonian(
        dims=dims,
        t=1.0,
        U=2.0,
        v=[0.2, -0.1],
        pbc=False,
        nq_override=6,
    )
    assert _nq_of(H_big) == 6


def test_hubbard_nq_override_rejects_too_small():
    with pytest.raises(Exception):
        build_hubbard_hamiltonian(dims=2, t=1.0, U=2.0, nq_override=3)


def test_hubbard_holstein_qubit_count_and_layout():
    dims = 2
    n_sites = n_sites_from_dims(dims)
    qpb = boson_qubits_per_site(2, "binary")
    nq_expected = 2 * n_sites + n_sites * qpb

    H = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=1.0,
        U=2.0,
        omega0=0.7,
        g=0.3,
        n_ph_max=2,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        t_eval=None,
        pbc=False,
    )

    assert _nq_of(H) == nq_expected

    q0 = phonon_qubit_indices_for_site(0, n_sites=n_sites, qpb=qpb, fermion_qubits=2 * n_sites)
    q1 = phonon_qubit_indices_for_site(1, n_sites=n_sites, qpb=qpb, fermion_qubits=2 * n_sites)
    assert q0 == [4, 5]
    assert q1 == [6, 7]
    assert set(q0).isdisjoint(set(q1))


def test_holstein_component_hermiticity_and_total():
    dims = 2
    n_sites = n_sites_from_dims(dims)
    qpb = boson_qubits_per_site(2, "binary")
    nq_total = 2 * n_sites + n_sites * qpb

    H_ph = build_holstein_phonon_energy(
        dims=dims,
        omega0=0.7,
        n_ph_max=2,
        boson_encoding="binary",
        zero_point=True,
    )
    H_g = build_holstein_coupling(
        dims=dims,
        g=0.3,
        n_ph_max=2,
        boson_encoding="binary",
        indexing="interleaved",
    )
    H_drive = build_hubbard_holstein_drive(
        dims=dims,
        v_t=[0.25, -0.15],
        v0=[0.05, -0.10],
        t=None,
        nq_override=nq_total,
    )
    H_tot = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=1.0,
        U=2.0,
        omega0=0.7,
        g=0.3,
        n_ph_max=2,
        boson_encoding="binary",
        v_t=[0.25, -0.15],
        v0=[0.05, -0.10],
        t_eval=None,
        pbc=False,
    )

    _assert_hermitian(H_ph)
    _assert_hermitian(H_g)
    _assert_hermitian(H_drive)
    _assert_hermitian(H_tot)


def test_hubbard_limit_g0_omega0_0_matches_embedded_hubbard():
    dims = 2
    n_sites = n_sites_from_dims(dims)
    qpb = boson_qubits_per_site(2, "binary")
    nq_total = 2 * n_sites + n_sites * qpb

    H_hh = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=1.0,
        U=2.0,
        omega0=0.0,
        g=0.0,
        n_ph_max=2,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        t_eval=None,
        include_zero_point=False,
        pbc=False,
    )
    H_hubb = build_hubbard_hamiltonian(
        dims=dims,
        t=1.0,
        U=2.0,
        v=None,
        pbc=False,
        nq_override=nq_total,
    )

    M_hh = hamiltonian_matrix(H_hh)
    M_hubb = hamiltonian_matrix(H_hubb)
    assert np.allclose(M_hh, M_hubb, atol=1e-10)


def test_one_site_exact_matrix_spotcheck():
    dims = 1
    U = 1.3
    omega0 = 0.6
    g = 0.2

    H = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=0.0,
        U=U,
        omega0=omega0,
        g=g,
        n_ph_max=1,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        include_zero_point=True,
        pbc=False,
    )

    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    n_f = 0.5 * (I2 - Z)
    n_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    x_b = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    # Matrix order is q_(n-1) ... q_0 = q2 (phonon), q1 (down), q0 (up).
    n_up = np.kron(np.kron(I2, I2), n_f)
    n_dn = np.kron(np.kron(I2, n_f), I2)
    n_b_glob = np.kron(np.kron(n_b, I2), I2)
    x_b_glob = np.kron(np.kron(x_b, I2), I2)

    I8 = np.eye(8, dtype=complex)
    H_ref = (
        U * (n_up @ n_dn)
        + omega0 * (n_b_glob + 0.5 * I8)
        + g * (x_b_glob @ ((n_up + n_dn) - I8))
    )

    M = hamiltonian_matrix(H)
    assert np.allclose(M, H_ref, atol=1e-10)
