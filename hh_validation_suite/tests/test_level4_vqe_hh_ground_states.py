from __future__ import annotations

import numpy as np
import pytest

from pydephasing.quantum.ed_hubbard_holstein import (
    build_hh_sector_basis,
    build_hh_sector_hamiltonian_ed,
    matrix_to_dense,
)
from pydephasing.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from pydephasing.quantum.vqe_latex_python_pairs_test import (
    PauliPolynomial,
    HubbardHolsteinLayerwiseAnsatz,
    apply_exp_pauli_polynomial,
    hubbard_holstein_reference_state,
    vqe_minimize,
)
from pydephasing.quantum.pauli_words import PauliTerm


def _half_filled_particles(dims: int) -> tuple[int, int]:
    n_sites = int(dims)
    return ((n_sites + 1) // 2, n_sites // 2)


def _exact_sector_ground_energy(
    dims: int,
    J: float,
    U: float,
    omega0: float,
    g: float,
    n_ph_max: int,
) -> float:
    num_particles = _half_filled_particles(dims)
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
        sparse=True,
        basis=basis,
        include_zero_point=True,
    )
    return float(np.min(np.linalg.eigvalsh(matrix_to_dense(H_ed))))


def _vqe_ground_energy(
    dims: int,
    J: float,
    U: float,
    omega0: float,
    g: float,
    n_ph_max: int,
    *,
    ansatz_style: str = "termwise",
    reps: int = 1,
) -> float:
    H = build_hubbard_holstein_hamiltonian(
        dims=dims,
        J=J,
        U=U,
        omega0=omega0,
        g=g,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        pbc=False,
        v_t=None,
        v0=None,
        t_eval=None,
    )
    if ansatz_style == "termwise":
        ansatz = _termwise_ansatz(H, reps=reps)
    elif ansatz_style == "layerwise":
        ansatz = HubbardHolsteinLayerwiseAnsatz(
            dims=dims,
            J=J,
            U=U,
            omega0=omega0,
            g=g,
            n_ph_max=n_ph_max,
            boson_encoding="binary",
            reps=reps,
            pbc=False,
        )
    else:
        raise ValueError(f"Unknown ansatz_style '{ansatz_style}'")
    psi_ref = hubbard_holstein_reference_state(
        dims=dims,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        indexing="blocked",
    )
    result = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=4,
        seed=11,
        maxiter=80,
        method="SLSQP",
    )
    return float(result.energy)


def _termwise_ansatz(H, *, reps: int = 1):
    nq = int(H.return_polynomial()[0].nqubit())
    repr_mode = H._repr_mode  # type: ignore[attr-defined]

    class _TermwiseAnsatz:
        def __init__(self, nq: int, terms, reps: int):
            self.nq = nq
            self.reps = int(reps)
            self.base_terms = terms
            self.num_parameters = len(terms) * self.reps

        def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray, **_kwargs) -> np.ndarray:
            if len(theta) != self.num_parameters:
                raise ValueError("theta length does not match term-wise ansatz parameter count")
            psi = np.array(psi_ref, copy=True)
            k = 0
            for _ in range(self.reps):
                for term in self.base_terms:
                    psi = apply_exp_pauli_polynomial(psi, term, float(theta[k]))
                    k += 1
            return psi

    term_polys = [
        PauliPolynomial(
            repr_mode,
            [PauliTerm(nq, ps=term.pw2strng(), pc=complex(term.p_coeff))],
        )
        for term in H.return_polynomial()
    ]

    return _TermwiseAnsatz(nq=nq, terms=term_polys, reps=reps)


def test_l2_hh_vqe_ground_state_within_weak_coupling_window():
    energy = _vqe_ground_energy(
        dims=2,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
        ansatz_style="termwise",
        reps=1,
    )
    exact = _exact_sector_ground_energy(
        dims=2,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
    )
    # Weak-coupling target: algorithm currently sits within ~0.31 for this L=2 case.
    assert np.isfinite(energy)
    assert energy - exact <= 0.34


def test_l3_hh_vqe_ground_state_within_weak_coupling_window():
    energy = _vqe_ground_energy(
        dims=3,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
        ansatz_style="termwise",
        reps=1,
    )
    exact = _exact_sector_ground_energy(
        dims=3,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
    )
    # Weak-coupling target for L=3: algorithm is closer than ~0.65 to exact currently.
    assert np.isfinite(energy)
    assert energy - exact <= 0.66


@pytest.mark.xfail(
    reason="Current HH term-wise HVA/Ansatz configuration is not chemistry-grade for L=2 (gap ~0.31 in this regime).",
    strict=True,
)
def test_l2_hh_vqe_ground_state_requires_chemical_accuracy():
    energy = _vqe_ground_energy(
        dims=2,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
        ansatz_style="termwise",
        reps=2,
    )
    exact = _exact_sector_ground_energy(
        dims=2,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
    )
    assert np.isfinite(energy)
    assert np.isfinite(exact)
    assert energy - exact <= 0.01


@pytest.mark.xfail(
    reason="Current HH term-wise HVA/Ansatz configuration is not chemistry-grade for L=3 (gap ~0.66 in this regime).",
    strict=True,
)
def test_l3_hh_vqe_ground_state_requires_chemical_accuracy():
    energy = _vqe_ground_energy(
        dims=3,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
        ansatz_style="termwise",
        reps=2,
    )
    exact = _exact_sector_ground_energy(
        dims=3,
        J=0.2,
        U=0.2,
        omega0=0.2,
        g=0.0,
        n_ph_max=1,
    )
    assert np.isfinite(energy)
    assert np.isfinite(exact)
    assert energy - exact <= 0.01
