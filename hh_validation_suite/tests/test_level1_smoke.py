from __future__ import annotations

import numpy as np
import pytest

from pydephasing.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
    mode_index,
    n_sites_from_dims,
    phonon_qubit_indices_for_site,
)
from pydephasing.quantum.vqe_latex_python_pairs_test import hamiltonian_matrix


def _nq_of(poly):
    terms = poly.return_polynomial()
    if not terms:
        return 0
    return int(terms[0].nqubit())


def test_site_counts_and_indexing_conventions():
    assert n_sites_from_dims(2) == 2
    assert n_sites_from_dims((2, 3)) == 6

    assert mode_index(0, 0, indexing="interleaved") == 0
    assert mode_index(0, 1, indexing="interleaved") == 1
    assert mode_index(3, 0, indexing="interleaved") == 6
    assert mode_index(3, 1, indexing="interleaved") == 7

    assert mode_index(2, 0, indexing="blocked", n_sites=5) == 2
    assert mode_index(2, 1, indexing="blocked", n_sites=5) == 7


def test_hubbard_holstein_qubit_counts_and_layout():
    dims = 2
    n_sites = n_sites_from_dims(dims)
    n_ph_max = 2
    qpb = boson_qubits_per_site(n_ph_max, "binary")
    expected_nq = 2 * n_sites + n_sites * qpb

    H = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=1.0,
        U=2.0,
        omega0=0.75,
        g=0.4,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        t_eval=None,
        pbc=False,
    )
    assert _nq_of(H) == expected_nq

    n_bos = n_sites * qpb
    n_ferm = 2 * n_sites
    assert phonon_qubit_indices_for_site(0, n_sites=n_sites, qpb=qpb, fermion_qubits=n_ferm) == list(
        range(n_ferm, n_ferm + n_bos // n_sites)
    )
    assert phonon_qubit_indices_for_site(1, n_sites=n_sites, qpb=qpb, fermion_qubits=n_ferm) == list(
        range(n_ferm + qpb, n_ferm + 2 * qpb)
    )


def test_hubbard_holstein_matrix_is_hermitian():
    H = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=0.8,
        U=1.9,
        omega0=0.6,
        g=0.15,
        n_ph_max=1,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        t_eval=None,
        pbc=False,
    )
    M = hamiltonian_matrix(H)
    assert np.allclose(M, M.conj().T, atol=1e-12)


def test_hubbard_holstein_limit_matches_embedded_hubbard():
    dims = 3
    n_sites = n_sites_from_dims(dims)
    n_ph_max = 1
    qpb = boson_qubits_per_site(n_ph_max, "binary")
    nq_total = 2 * n_sites + n_sites * qpb

    H_hh = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=1.2,
        U=0.7,
        omega0=0.0,
        g=0.0,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        t_eval=None,
        include_zero_point=False,
        pbc=False,
    )
    H_hubbard = build_hubbard_hamiltonian(
        dims=dims,
        t=1.2,
        U=0.7,
        v=None,
        pbc=False,
        nq_override=nq_total,
    )

    Mh = hamiltonian_matrix(H_hh)
    Mu = hamiltonian_matrix(H_hubbard)
    assert np.allclose(Mh, Mu, atol=1e-10)


@pytest.mark.parametrize("bad_input", [-1, 0, (2, 0)])
def test_invalid_inputs_are_rejected(bad_input):
    with pytest.raises(Exception):
        if isinstance(bad_input, int):
            n_sites_from_dims(bad_input)
        else:
            n_sites_from_dims(bad_input)  # type: ignore[arg-type]
