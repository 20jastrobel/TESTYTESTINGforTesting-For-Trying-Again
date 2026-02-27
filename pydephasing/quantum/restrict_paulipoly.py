from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.sparse import coo_matrix, spmatrix
except Exception:  # pragma: no cover - SciPy is optional
    coo_matrix = None  # type: ignore[assignment]
    spmatrix = None  # type: ignore[assignment]

try:
    from pydephasing.utilities.log import log
except Exception:  # pragma: no cover - local fallback when utilities package is absent
    class _FallbackLog:
        @staticmethod
        def error(msg: str):
            raise RuntimeError(msg)

    log = _FallbackLog()

from pydephasing.quantum.pauli_polynomial_class import PauliPolynomial

MatrixLike = Union[np.ndarray, "spmatrix"]


@dataclass(frozen=True)
class _CompiledPauliTerm:
    coeff: complex
    flip_mask: int
    y_mask: int
    z_mask: int
    y_prefactor: complex


def _compile_pauli_term(pauli_word: str, coeff: complex) -> _CompiledPauliTerm:
    ps = str(pauli_word)
    nq = len(ps)
    flip_mask = 0
    y_mask = 0
    z_mask = 0
    n_y = 0
    for q in range(nq):
        op = ps[nq - 1 - q]  # op acting on qubit q
        if op == "e":
            continue
        if op == "x":
            flip_mask |= (1 << q)
            continue
        if op == "y":
            flip_mask |= (1 << q)
            y_mask |= (1 << q)
            n_y += 1
            continue
        if op == "z":
            z_mask |= (1 << q)
            continue
        log.error(f"invalid Pauli symbol '{op}' in '{pauli_word}'")
    return _CompiledPauliTerm(
        coeff=complex(coeff),
        flip_mask=int(flip_mask),
        y_mask=int(y_mask),
        z_mask=int(z_mask),
        y_prefactor=(1j) ** int(n_y),
    )


def restrict_pauli_polynomial_matrix(
    poly: PauliPolynomial,
    basis_indices: Sequence[int],
    *,
    tol: float = 1e-12,
    sparse: bool = True,
) -> MatrixLike:
    """
    Build P H P on an explicit basis subset without constructing the full 2^n matrix.

    For each Pauli word P and computational basis state |z>:
      P|z> = phase(z,P) |z xor flip_mask(P)>.
    """
    basis = [int(i) for i in basis_indices]
    dim = int(len(basis))
    if dim == 0:
        if sparse and coo_matrix is not None:
            return coo_matrix((0, 0), dtype=complex)
        return np.zeros((0, 0), dtype=complex)

    terms = list(poly.return_polynomial())
    compiled: List[_CompiledPauliTerm] = []
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        compiled.append(_compile_pauli_term(term.pw2strng(), coeff))

    index_of = {idx: i for i, idx in enumerate(basis)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[complex] = []

    for col, z in enumerate(basis):
        z_i = int(z)
        for term in compiled:
            tgt = z_i ^ int(term.flip_mask)
            row = index_of.get(tgt)
            if row is None:
                continue

            parity = (((z_i & int(term.y_mask)).bit_count() + (z_i & int(term.z_mask)).bit_count()) & 1)
            phase = term.y_prefactor if parity == 0 else -term.y_prefactor
            val = complex(term.coeff * phase)
            if abs(val) <= float(tol):
                continue
            rows.append(int(row))
            cols.append(int(col))
            data.append(val)

    if sparse and coo_matrix is not None:
        out = coo_matrix((np.array(data, dtype=complex), (rows, cols)), shape=(dim, dim), dtype=complex).tocsr()
        out.sum_duplicates()
        return out

    out_dense = np.zeros((dim, dim), dtype=complex)
    for r, c, v in zip(rows, cols, data):
        out_dense[int(r), int(c)] += complex(v)
    return out_dense


def matrix_to_dense(h_mat: MatrixLike) -> np.ndarray:
    if spmatrix is not None and isinstance(h_mat, spmatrix):
        return np.asarray(h_mat.toarray(), dtype=complex)
    return np.asarray(h_mat, dtype=complex)


def hermiticity_residual(h_mat: MatrixLike) -> float:
    if spmatrix is not None and isinstance(h_mat, spmatrix):
        diff = (h_mat - h_mat.getH()).tocsr()
        if diff.nnz == 0:
            return 0.0
        return float(np.max(np.abs(diff.data)))
    dense = matrix_to_dense(h_mat)
    return float(np.max(np.abs(dense - dense.conj().T))) if dense.size else 0.0


def lowest_eigenvalues(h_mat: MatrixLike, k: int) -> np.ndarray:
    dense = matrix_to_dense(h_mat)
    if dense.shape[0] == 0:
        return np.array([], dtype=float)
    evals = np.linalg.eigvalsh(dense)
    k_i = max(1, min(int(k), int(evals.size)))
    return np.asarray(np.real(evals[:k_i]), dtype=float)
