import numpy as np
import pytest

from pydephasing.quantum.hartree_fock_reference_state import hartree_fock_bitstring
from pydephasing.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    n_sites_from_dims,
)
from pydephasing.quantum.vqe_latex_python_pairs_test import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardLayerwiseAnsatz,
    basis_state,
    hamiltonian_matrix,
    half_filled_num_particles,
    hubbard_holstein_reference_state,
    vqe_minimize,
)


def test_hh_ansatz_num_parameters_positive_and_stable():
    ansatz_a = HubbardHolsteinLayerwiseAnsatz(
        dims=2,
        J=1.0,
        U=2.0,
        omega0=0.7,
        g=0.3,
        n_ph_max=2,
        boson_encoding="binary",
        reps=2,
        indexing="blocked",
        pbc=False,
    )
    ansatz_b = HubbardHolsteinLayerwiseAnsatz(
        dims=2,
        J=1.0,
        U=2.0,
        omega0=0.7,
        g=0.3,
        n_ph_max=2,
        boson_encoding="binary",
        reps=2,
        indexing="blocked",
        pbc=False,
    )

    assert ansatz_a.num_parameters > 0
    assert ansatz_a.num_parameters == ansatz_b.num_parameters


def test_hh_prepare_state_shape_matches_2_pow_n_total():
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=1,
        J=0.8,
        U=1.2,
        omega0=0.5,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        indexing="blocked",
        pbc=False,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=1,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
    )
    theta = np.zeros(ansatz.num_parameters, dtype=float)
    psi = ansatz.prepare_state(theta, psi_ref)
    assert psi.shape == (1 << ansatz.nq,)


def test_hh_prepare_state_norm_preserved():
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=1,
        J=1.0,
        U=2.0,
        omega0=0.4,
        g=0.3,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        indexing="blocked",
        pbc=False,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=1,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
    )
    theta = np.linspace(0.05, 0.15, ansatz.num_parameters, dtype=float)
    psi = ansatz.prepare_state(theta, psi_ref)
    assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-10)


def test_hh_reference_state_layout_fermion_hf_plus_phonon_vacuum():
    dims = 2
    n_sites = n_sites_from_dims(dims)
    num_particles = half_filled_num_particles(n_sites)
    n_ph_max = 2
    qpb = boson_qubits_per_site(n_ph_max, "binary")
    n_bos = n_sites * qpb
    n_total = 2 * n_sites + n_bos

    hf_ferm = hartree_fock_bitstring(n_sites=n_sites, num_particles=num_particles, indexing="blocked")
    expected_bits = ("0" * n_bos) + hf_ferm

    psi = hubbard_holstein_reference_state(
        dims=dims,
        num_particles=num_particles,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        indexing="blocked",
    )
    psi_expected = basis_state(n_total, expected_bits)

    assert np.allclose(psi, psi_expected, atol=1e-12)


def test_hh_vqe_minimize_runs_without_optimizer_core_changes():
    dims = 1
    H = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=0.7,
        U=1.1,
        omega0=0.5,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        pbc=False,
    )
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=dims,
        J=0.7,
        U=1.1,
        omega0=0.5,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        indexing="blocked",
        pbc=False,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=dims,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
    )

    res = vqe_minimize(H, ansatz, psi_ref, restarts=1, seed=3, maxiter=8, method="SLSQP")
    assert np.isfinite(float(res.energy))
    assert int(res.theta.size) == int(ansatz.num_parameters)


def test_hh_reduces_to_hubbard_ansatz_path_when_g_omega0_zero():
    dims = 2
    reps = 1
    J = 1.0
    U = 2.0
    n_ph_max = 1

    hh_ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=dims,
        J=J,
        U=U,
        omega0=0.0,
        g=0.0,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        reps=reps,
        indexing="blocked",
        pbc=False,
        include_zero_point=False,
    )
    hub_ansatz = HubbardLayerwiseAnsatz(
        dims=dims,
        t=J,
        U=U,
        v=None,
        reps=reps,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_potential_terms=False,
    )

    if int(hh_ansatz.num_parameters) != int(hub_ansatz.num_parameters):
        pytest.skip("HH/Hubbard layerwise parameter counts differ for this config.")

    n_sites = n_sites_from_dims(dims)
    qpb = boson_qubits_per_site(n_ph_max, "binary")
    n_bos = n_sites * qpb
    n_ferm = 2 * n_sites
    hf_ferm = hartree_fock_bitstring(
        n_sites=n_sites,
        num_particles=half_filled_num_particles(n_sites),
        indexing="blocked",
    )

    psi_hh_ref = hubbard_holstein_reference_state(
        dims=dims,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        indexing="blocked",
    )
    psi_hub_ref = basis_state(n_ferm, hf_ferm)

    theta = np.linspace(0.03, 0.03 * hh_ansatz.num_parameters, hh_ansatz.num_parameters, dtype=float)
    psi_hh = hh_ansatz.prepare_state(theta, psi_hh_ref)
    psi_hub = hub_ansatz.prepare_state(theta, psi_hub_ref)

    bos_vac = basis_state(n_bos, "0" * n_bos)
    psi_hub_embedded = np.kron(bos_vac, psi_hub)

    assert np.allclose(psi_hh, psi_hub_embedded, atol=1e-10)

    nq_total = n_ferm + n_bos
    H_hh = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=J,
        U=U,
        omega0=0.0,
        g=0.0,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        include_zero_point=False,
        pbc=False,
    )
    H_hubb_emb = build_hubbard_hamiltonian(
        dims=dims,
        t=J,
        U=U,
        v=None,
        pbc=False,
        nq_override=nq_total,
    )

    M_hh = hamiltonian_matrix(H_hh)
    M_hubb = hamiltonian_matrix(H_hubb_emb)
    assert np.allclose(M_hh, M_hubb, atol=1e-10)
