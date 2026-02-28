from __future__ import annotations

import numpy as np

from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from pydephasing.quantum.vqe_latex_python_pairs_test import (
    HubbardHolsteinLayerwiseAnsatz,
    expval_pauli_polynomial,
    hubbard_holstein_reference_state,
    vqe_minimize,
)


def _reference_state(dims: int, n_ph_max: int):
    return hubbard_holstein_reference_state(
        dims=dims,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        indexing="blocked",
    )


def test_hh_layerwise_parameter_count_scales_with_reps():
    common = dict(
        dims=2,
        J=0.9,
        U=2.1,
        omega0=0.6,
        g=0.2,
        n_ph_max=2,
        boson_encoding="binary",
        pbc=False,
    )
    one = HubbardHolsteinLayerwiseAnsatz(reps=1, **common)
    three = HubbardHolsteinLayerwiseAnsatz(reps=3, **common)
    assert three.num_parameters == 3 * one.num_parameters
    assert one.num_parameters > 0


def test_hh_ansatz_prepare_state_is_unit_norm_and_reproducible():
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=1,
        J=0.8,
        U=1.2,
        omega0=0.5,
        g=0.3,
        n_ph_max=1,
        boson_encoding="binary",
        reps=2,
        pbc=False,
    )
    psi_ref = _reference_state(dims=1, n_ph_max=1)
    theta = np.linspace(0.02, 0.24, ansatz.num_parameters, dtype=float)

    psi_a = ansatz.prepare_state(theta, psi_ref)
    psi_b = ansatz.prepare_state(theta, psi_ref)

    assert psi_a.shape == (1 << ansatz.nq,)
    assert np.isclose(np.linalg.norm(psi_a), 1.0, atol=1e-10)
    assert np.allclose(psi_a, psi_b, atol=1e-12)


def test_vqe_minimize_returns_objective_energy_for_output_theta():
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=1,
        J=0.7,
        U=1.1,
        omega0=0.4,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        pbc=False,
    )
    H = build_hubbard_holstein_hamiltonian(
        dims=1,
        J=0.7,
        U=1.1,
        omega0=0.4,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        t_eval=None,
        pbc=False,
    )
    psi_ref = _reference_state(dims=1, n_ph_max=1)

    res = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=2,
        seed=3,
        maxiter=8,
        method="SLSQP",
    )

    psi_v = ansatz.prepare_state(res.theta, psi_ref)
    e_check = expval_pauli_polynomial(psi_v, H)

    assert np.isfinite(float(res.energy))
    assert 0 <= res.best_restart < 2
    assert np.isclose(float(e_check), float(res.energy), atol=1e-8)
    assert np.isclose(np.linalg.norm(psi_v), 1.0, atol=1e-10)


def test_reproducible_seed_produces_reproducible_vqe_energy():
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=1,
        J=0.6,
        U=0.9,
        omega0=0.5,
        g=0.25,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        pbc=False,
    )
    H = build_hubbard_holstein_hamiltonian(
        dims=1,
        J=0.6,
        U=0.9,
        omega0=0.5,
        g=0.25,
        n_ph_max=1,
        boson_encoding="binary",
        v_t=None,
        v0=None,
        t_eval=None,
        pbc=False,
    )
    psi_ref = _reference_state(dims=1, n_ph_max=1)

    r1 = vqe_minimize(H, ansatz, psi_ref, restarts=1, seed=11, maxiter=6, method="SLSQP")
    r2 = vqe_minimize(H, ansatz, psi_ref, restarts=1, seed=11, maxiter=6, method="SLSQP")

    assert np.isclose(r1.energy, r2.energy, atol=1e-10)
    assert r1.theta.shape == r2.theta.shape
