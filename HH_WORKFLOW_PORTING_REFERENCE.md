# HH Workflow Porting Reference

This document contains the exact code added/updated in this chat, grouped by file.

## File: pydephasing/quantum/hubbard_latex_python_pairs.py

### Added type aliases
```python
Spin = int  # 0 -> up, 1 -> down
Dims = Union[int, Tuple[int, ...]]  # L or (Lx, Ly, ...)
SitePotential = Optional[Union[float, Sequence[float], Dict[int, float]]]
TimePotential = Optional[
    Union[
        float,
        Sequence[float],
        Dict[int, float],
        Callable[[Optional[float]], Union[float, Sequence[float], Dict[int, float]]],
    ]
]
```

### Added/updated Hubbard embedding + HH builders
```python
def _resolve_hubbard_nq(n_sites: int, nq_override: Optional[int]) -> int:
    nq_min = 2 * int(n_sites)
    if nq_override is None:
        return nq_min
    nq = int(nq_override)
    if nq < nq_min:
        log.error("nq_override must be >= 2*n_sites")
    return nq


def build_hubbard_kinetic(
    dims: Dims,
    t: float,
    *,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    pbc: Union[bool, Sequence[bool]] = True,
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """Hopping term H_t."""
    n_sites = n_sites_from_dims(dims)
    nq = _resolve_hubbard_nq(n_sites=n_sites, nq_override=nq_override)
    if edges is None:
        edges = bravais_nearest_neighbor_edges(dims, pbc=pbc)

    c_dag: Dict[int, PauliPolynomial] = {}
    c: Dict[int, PauliPolynomial] = {}

    def cd(p_mode: int) -> PauliPolynomial:
        if p_mode not in c_dag:
            c_dag[p_mode] = fermion_plus_operator(repr_mode, nq, p_mode)
        return c_dag[p_mode]

    def cm(p_mode: int) -> PauliPolynomial:
        if p_mode not in c:
            c[p_mode] = fermion_minus_operator(repr_mode, nq, p_mode)
        return c[p_mode]

    Ht = PauliPolynomial(repr_mode)

    for (i, j) in edges:
        for spin in (SPIN_UP, SPIN_DN):
            pi = mode_index(i, spin, indexing=indexing, n_sites=n_sites)
            pj = mode_index(j, spin, indexing=indexing, n_sites=n_sites)
            Ht += (-t) * ((cd(pi) * cm(pj)) + (cd(pj) * cm(pi)))

    return Ht


def build_hubbard_onsite(
    dims: Dims,
    U: float,
    *,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """Onsite interaction term H_U."""
    n_sites = n_sites_from_dims(dims)
    nq = _resolve_hubbard_nq(n_sites=n_sites, nq_override=nq_override)

    n_cache: Dict[int, PauliPolynomial] = {}

    def n_op(p_mode: int) -> PauliPolynomial:
        if p_mode not in n_cache:
            n_cache[p_mode] = jw_number_operator(repr_mode, nq, p_mode)
        return n_cache[p_mode]

    HU = PauliPolynomial(repr_mode)

    for i in range(n_sites):
        p_up = mode_index(i, SPIN_UP, indexing=indexing, n_sites=n_sites)
        p_dn = mode_index(i, SPIN_DN, indexing=indexing, n_sites=n_sites)
        HU += U * (n_op(p_up) * n_op(p_dn))

    return HU


def _parse_site_potential(
    v: SitePotential,
    n_sites: int,
) -> List[float]:
    if v is None:
        return [0.0] * n_sites
    if isinstance(v, (int, float, complex)):
        return [float(v)] * n_sites
    if isinstance(v, dict):
        out = [0.0] * n_sites
        for k, val in v.items():
            idx = int(k)
            if idx < 0 or idx >= n_sites:
                log.error("site-potential key out of bounds")
            out[idx] = float(val)
        return out
    if len(v) != n_sites:
        log.error("site potential v must be scalar, dict, or length n_sites")
    return [float(val) for val in v]


def build_hubbard_potential(
    dims: Dims,
    v: SitePotential,
    *,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """Local potential term H_v."""
    n_sites = n_sites_from_dims(dims)
    nq = _resolve_hubbard_nq(n_sites=n_sites, nq_override=nq_override)
    v_list = _parse_site_potential(v, n_sites=n_sites)

    n_cache: Dict[int, PauliPolynomial] = {}

    def n_op(p_mode: int) -> PauliPolynomial:
        if p_mode not in n_cache:
            n_cache[p_mode] = jw_number_operator(repr_mode, nq, p_mode)
        return n_cache[p_mode]

    Hv = PauliPolynomial(repr_mode)
    for i in range(n_sites):
        vi = v_list[i]
        if abs(vi) < 1e-15:
            continue
        for spin in (SPIN_UP, SPIN_DN):
            p_mode = mode_index(i, spin, indexing=indexing, n_sites=n_sites)
            Hv += (-vi) * n_op(p_mode)

    return Hv


def build_hubbard_hamiltonian(
    dims: Dims,
    t: float,
    U: float,
    *,
    v: SitePotential = None,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    pbc: Union[bool, Sequence[bool]] = True,
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """Full Hamiltonian H = H_t + H_U + H_v."""
    Ht = build_hubbard_kinetic(
        dims=dims,
        t=t,
        repr_mode=repr_mode,
        indexing=indexing,
        edges=edges,
        pbc=pbc,
        nq_override=nq_override,
    )
    HU = build_hubbard_onsite(
        dims=dims,
        U=U,
        repr_mode=repr_mode,
        indexing=indexing,
        nq_override=nq_override,
    )
    Hv = build_hubbard_potential(
        dims=dims,
        v=v,
        repr_mode=repr_mode,
        indexing=indexing,
        nq_override=nq_override,
    )
    return Ht + HU + Hv


def boson_qubits_per_site(n_ph_max: int, encoding: str = "binary") -> int:
    """Return qubits-per-site for a local truncated phonon Hilbert space."""
    n_ph_i = int(n_ph_max)
    if n_ph_i < 0:
        log.error("n_ph_max must be >= 0")
    d = n_ph_i + 1
    if encoding == "binary":
        return max(1, int(math.ceil(math.log2(d))))
    log.error("unknown boson encoding")
    return max(1, int(math.ceil(math.log2(d))))


def phonon_qubit_indices_for_site(
    site: int,
    *,
    n_sites: int,
    qpb: int,
    fermion_qubits: int,
) -> List[int]:
    """Global phonon qubits for one site in the packed [fermions][phonons] layout."""
    site_i = int(site)
    n_sites_i = int(n_sites)
    qpb_i = int(qpb)
    fermion_q_i = int(fermion_qubits)

    if n_sites_i <= 0:
        log.error("n_sites must be positive")
    if site_i < 0 or site_i >= n_sites_i:
        log.error("site index out of bounds")
    if qpb_i <= 0:
        log.error("qpb must be positive")
    if fermion_q_i < 0:
        log.error("fermion_qubits must be >= 0")

    base = fermion_q_i + site_i * qpb_i
    return list(range(base, base + qpb_i))


def _normalize_pauli_symbol(sym: str) -> str:
    if not isinstance(sym, str) or len(sym) != 1:
        log.error("Pauli symbol must be a single character")
    trans = {"I": "e", "X": "x", "Y": "y", "Z": "z"}
    if sym in trans:
        return trans[sym]
    out = sym.lower()
    if out not in ("e", "x", "y", "z"):
        log.error(f"invalid Pauli symbol '{sym}'")
    return out


def _identity_pauli_polynomial(repr_mode: str, nq: int) -> PauliPolynomial:
    return PauliPolynomial(repr_mode, [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


def _pauli_monomial_global(
    repr_mode: str,
    nq: int,
    ops: Dict[int, str],
) -> PauliPolynomial:
    nq_i = int(nq)
    chars = ["e"] * nq_i
    for qubit, sym in ops.items():
        q = int(qubit)
        if q < 0 or q >= nq_i:
            log.error("global qubit index out of range for Pauli monomial")
        norm = _normalize_pauli_symbol(sym)
        if norm == "e":
            continue
        chars[nq_i - 1 - q] = norm
    return PauliPolynomial(repr_mode, [PauliTerm(nq_i, ps="".join(chars), pc=1.0)])


@lru_cache(maxsize=None)
def _boson_local_matrices(n_ph_max: int) -> Dict[str, np.ndarray]:
    """Local truncated oscillator matrices on d = n_ph_max + 1 Fock levels."""
    n_ph_i = int(n_ph_max)
    if n_ph_i < 0:
        log.error("n_ph_max must be >= 0")
    d = n_ph_i + 1
    b = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        b[n - 1, n] = np.sqrt(float(n))
    bdag = b.conj().T
    n_op = bdag @ b
    x_op = b + bdag
    return {"b": b, "bdag": bdag, "n": n_op, "x": x_op}


def _embed_to_qubit_space(mat_d: np.ndarray, qpb: int) -> np.ndarray:
    """Zero-pad a dxd local operator into a 2^qpb dimensional qubit subspace."""
    qpb_i = int(qpb)
    if qpb_i <= 0:
        log.error("qpb must be positive")
    d = int(mat_d.shape[0])
    if mat_d.shape != (d, d):
        log.error("mat_d must be square")
    dim = 1 << qpb_i
    if d > dim:
        log.error("local operator dimension exceeds qubit encoding dimension")
    out = np.zeros((dim, dim), dtype=np.complex128)
    out[:d, :d] = mat_d
    return out


@lru_cache(maxsize=None)
def _pauli_basis_mats(qpb: int) -> Tuple[List[str], List[np.ndarray]]:
    """All Pauli basis labels/matrices on qpb local qubits."""
    qpb_i = int(qpb)
    if qpb_i <= 0:
        log.error("qpb must be positive")

    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    oneq = {"I": I, "X": X, "Y": Y, "Z": Z}

    labels: List[str] = []
    mats: List[np.ndarray] = []
    for s in itertools.product("IXYZ", repeat=qpb_i):
        lbl = "".join(s)
        mat = oneq[s[0]]
        for k in range(1, qpb_i):
            mat = np.kron(mat, oneq[s[k]])
        labels.append(lbl)
        mats.append(mat)
    return labels, mats


def _matrix_to_pauli_coeffs(
    mat: np.ndarray,
    qpb: int,
    tol: float,
) -> List[Tuple[str, complex]]:
    """Decompose mat in Pauli basis on qpb qubits."""
    qpb_i = int(qpb)
    labels, mats = _pauli_basis_mats(qpb_i)
    dim = 1 << qpb_i
    out: List[Tuple[str, complex]] = []
    for lbl, P in zip(labels, mats):
        coeff = np.trace(P.conj().T @ mat) / float(dim)
        if abs(coeff) > float(tol):
            out.append((lbl, complex(coeff)))
    return out


@lru_cache(maxsize=None)
def boson_local_operator_pauli_decomp(
    which: str,
    *,
    n_ph_max: int,
    qpb: int,
    tol: float,
) -> List[Tuple[str, complex]]:
    """Cached local Pauli decomposition for boson operators on one site."""
    if which not in ("b", "bdag", "n", "x"):
        log.error("which must be one of: 'b', 'bdag', 'n', 'x'")
    mats = _boson_local_matrices(int(n_ph_max))
    mat_d = mats[which]
    mat_q = _embed_to_qubit_space(mat_d, qpb=int(qpb))
    return _matrix_to_pauli_coeffs(mat_q, qpb=int(qpb), tol=float(tol))


def boson_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    which: str,
    n_ph_max: int,
    encoding: str,
    tol: float = 1e-12,
) -> PauliPolynomial:
    """Bosonic operator embedded on selected global qubits (binary encoding)."""
    if encoding != "binary":
        log.error("only binary boson encoding is implemented")

    qubits_i = [int(q) for q in qubits]
    if len(set(qubits_i)) != len(qubits_i):
        log.error("qubits for boson operator must be unique")
    qpb = len(qubits_i)
    if qpb <= 0:
        log.error("boson operator needs at least one qubit")

    expected_qpb = boson_qubits_per_site(int(n_ph_max), encoding=encoding)
    if qpb != expected_qpb:
        log.error("provided local boson qubit block does not match encoding size")

    terms = boson_local_operator_pauli_decomp(
        which,
        n_ph_max=int(n_ph_max),
        qpb=qpb,
        tol=float(tol),
    )

    H = PauliPolynomial(repr_mode)
    for lbl, coeff in terms:
        ops: Dict[int, str] = {}
        # Local matrix labels are ordered left->right as q_(qpb-1)...q_0.
        # Map to global qubits so the rightmost local char acts on qubits_i[0].
        for local_k, P in enumerate(lbl):
            if P == "I":
                continue
            global_q = qubits_i[qpb - 1 - local_k]
            ops[global_q] = P
        H += complex(coeff) * _pauli_monomial_global(repr_mode, int(nq_total), ops)
    return H


def boson_number_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    n_ph_max: int,
    encoding: str = "binary",
    tol: float = 1e-12,
) -> PauliPolynomial:
    return boson_operator(
        repr_mode,
        nq_total,
        qubits,
        which="n",
        n_ph_max=n_ph_max,
        encoding=encoding,
        tol=tol,
    )


def boson_displacement_operator(
    repr_mode: str,
    nq_total: int,
    qubits: Sequence[int],
    *,
    n_ph_max: int,
    encoding: str = "binary",
    tol: float = 1e-12,
) -> PauliPolynomial:
    return boson_operator(
        repr_mode,
        nq_total,
        qubits,
        which="x",
        n_ph_max=n_ph_max,
        encoding=encoding,
        tol=tol,
    )


def _eval_site_potential_maybe_time_dependent(
    v: TimePotential,
    *,
    t: Optional[float],
    n_sites: int,
) -> List[float]:
    if v is None:
        return [0.0] * int(n_sites)
    if callable(v):
        if t is None:
            log.error("t must be provided when v_t is callable")
        return _parse_site_potential(v(t), n_sites=int(n_sites))
    return _parse_site_potential(v, n_sites=int(n_sites))


def build_holstein_phonon_energy(
    dims: Dims,
    omega0: float,
    *,
    n_ph_max: int,
    boson_encoding: str = "binary",
    repr_mode: str = "JW",
    tol: float = 1e-12,
    zero_point: bool = True,
) -> PauliPolynomial:
    """H_ph = omega0 * sum_i (n_b,i + 1/2)."""
    n_sites = n_sites_from_dims(dims)
    fermion_qubits = 2 * n_sites
    qpb = boson_qubits_per_site(int(n_ph_max), boson_encoding)
    nq_total = fermion_qubits + n_sites * qpb

    Hph = PauliPolynomial(repr_mode)
    I = _identity_pauli_polynomial(repr_mode, nq_total)
    for i in range(n_sites):
        q_i = phonon_qubit_indices_for_site(
            i,
            n_sites=n_sites,
            qpb=qpb,
            fermion_qubits=fermion_qubits,
        )
        n_b = boson_number_operator(
            repr_mode,
            nq_total,
            q_i,
            n_ph_max=n_ph_max,
            encoding=boson_encoding,
            tol=tol,
        )
        Hph += float(omega0) * n_b
        if zero_point:
            Hph += (0.5 * float(omega0)) * I
    return Hph


def build_holstein_coupling(
    dims: Dims,
    g: float,
    *,
    n_ph_max: int,
    boson_encoding: str = "binary",
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    tol: float = 1e-12,
) -> PauliPolynomial:
    """H_g = g * sum_i x_i * (n_i - 1), with n_i = n_{i,up} + n_{i,dn}."""
    n_sites = n_sites_from_dims(dims)
    fermion_qubits = 2 * n_sites
    qpb = boson_qubits_per_site(int(n_ph_max), boson_encoding)
    nq_total = fermion_qubits + n_sites * qpb

    I = _identity_pauli_polynomial(repr_mode, nq_total)
    Hg = PauliPolynomial(repr_mode)

    n_cache: Dict[int, PauliPolynomial] = {}

    def n_op(p_mode: int) -> PauliPolynomial:
        if p_mode not in n_cache:
            n_cache[p_mode] = jw_number_operator(repr_mode, nq_total, p_mode)
        return n_cache[p_mode]

    for i in range(n_sites):
        p_up = mode_index(i, SPIN_UP, indexing=indexing, n_sites=n_sites)
        p_dn = mode_index(i, SPIN_DN, indexing=indexing, n_sites=n_sites)
        n_i = n_op(p_up) + n_op(p_dn)

        q_i = phonon_qubit_indices_for_site(
            i,
            n_sites=n_sites,
            qpb=qpb,
            fermion_qubits=fermion_qubits,
        )
        x_i = boson_displacement_operator(
            repr_mode,
            nq_total,
            q_i,
            n_ph_max=n_ph_max,
            encoding=boson_encoding,
            tol=tol,
        )
        Hg += float(g) * (x_i * (n_i + ((-1.0) * I)))
    return Hg


def build_hubbard_holstein_drive(
    dims: Dims,
    *,
    v_t: TimePotential,
    v0: SitePotential,
    t: Optional[float] = None,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    nq_override: Optional[int] = None,
) -> PauliPolynomial:
    """H_drive = sum_{i,sigma} (v_i(t) - v0_i) n_{i,sigma}."""
    n_sites = n_sites_from_dims(dims)
    v_t_list = _eval_site_potential_maybe_time_dependent(v_t, t=t, n_sites=n_sites)
    v0_list = _parse_site_potential(v0, n_sites=n_sites)
    delta = [float(v_t_list[i]) - float(v0_list[i]) for i in range(n_sites)]

    # build_hubbard_potential uses H_v = -sum_i,sigma v_i * n_i,sigma.
    v_for_existing = [-dv for dv in delta]
    return build_hubbard_potential(
        dims=dims,
        v=v_for_existing,
        repr_mode=repr_mode,
        indexing=indexing,
        nq_override=nq_override,
    )


def build_hubbard_holstein_hamiltonian(
    dims: Dims,
    *,
    J: float,
    U: float,
    omega0: float,
    g: float,
    n_ph_max: int,
    boson_encoding: str = "binary",
    v_t: TimePotential = None,
    v0: SitePotential = None,
    t_eval: Optional[float] = None,
    repr_mode: str = "JW",
    indexing: str = "interleaved",
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    pbc: Union[bool, Sequence[bool]] = True,
    tol: float = 1e-12,
    include_zero_point: bool = True,
) -> PauliPolynomial:
    """Full Hubbard-Holstein Hamiltonian on a shared fermion+phonon register."""
    n_sites = n_sites_from_dims(dims)
    fermion_qubits = 2 * n_sites
    qpb = boson_qubits_per_site(int(n_ph_max), boson_encoding)
    nq_total = fermion_qubits + n_sites * qpb

    H_hubb = build_hubbard_hamiltonian(
        dims=dims,
        t=J,
        U=U,
        v=None,
        repr_mode=repr_mode,
        indexing=indexing,
        edges=edges,
        pbc=pbc,
        nq_override=nq_total,
    )
    H_ph = build_holstein_phonon_energy(
        dims=dims,
        omega0=omega0,
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        repr_mode=repr_mode,
        tol=tol,
        zero_point=include_zero_point,
    )
    H_g = build_holstein_coupling(
        dims=dims,
        g=g,
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        repr_mode=repr_mode,
        indexing=indexing,
        tol=tol,
    )
    H_drive = build_hubbard_holstein_drive(
        dims=dims,
        v_t=v_t,
        v0=v0,
        t=t_eval,
        repr_mode=repr_mode,
        indexing=indexing,
        nq_override=nq_total,
    )
    return H_hubb + H_ph + H_g + H_drive
```

## File: pydephasing/quantum/vqe_latex_python_pairs_test.py

### Added Hubbard-Holstein imports/fallbacks
```python

    fermion_minus_operator = _missing_dep  # type: ignore[assignment]
    fermion_plus_operator = _missing_dep  # type: ignore[assignment]
    PauliTerm = _missing_dep  # type: ignore[assignment]

try:
    # Reuse your canonical Hubbard lattice helpers for consistent indexing/edges.
    from pydephasing.quantum.hubbard_latex_python_pairs import (
        Dims,
        SPIN_DN,
        SPIN_UP,
        Spin,
        boson_qubits_per_site,
        build_holstein_coupling,
        build_holstein_phonon_energy,
        build_hubbard_holstein_drive,
        build_hubbard_kinetic,
        build_hubbard_onsite,
        build_hubbard_potential,
        bravais_nearest_neighbor_edges,
        mode_index,
        n_sites_from_dims,
    )
except Exception:  # pragma: no cover
    Dims = Union[int, Tuple[int, ...]]
    Spin = int
    SPIN_UP = 0
    SPIN_DN = 1

    def n_sites_from_dims(dims: Dims) -> int:
        if isinstance(dims, int):
            return int(dims)
        out = 1
        for L in dims:
            out *= int(L)
        return out

    def bravais_nearest_neighbor_edges(dims: Dims, pbc: Union[bool, Sequence[bool]] = True):
        raise ImportError("bravais_nearest_neighbor_edges unavailable (import Hubbard helpers)")

    def mode_index(site: int, spin: Spin, indexing: str = "interleaved", n_sites: Optional[int] = None) -> int:
        raise ImportError("mode_index unavailable (import Hubbard helpers)")

```

### Added HH typing + helpers + reference state + HH ansatz
```python
SitePotential = Optional[Union[float, Sequence[float], Dict[int, float]]]
TimePotential = Optional[
    Union[
        float,
        Sequence[float],
        Dict[int, float],
        Callable[[Optional[float]], Union[float, Sequence[float], Dict[int, float]]],
    ]
]


def _single_term_polynomials_sorted(
    poly: PauliPolynomial,
    *,
    repr_mode: str,
    coefficient_tolerance: float = 1e-12,
) -> List[PauliPolynomial]:
    terms = list(poly.return_polynomial())
    terms.sort(key=lambda t: (t.pw2strng(), float(np.real(t.p_coeff)), float(np.imag(t.p_coeff))))
    out: List[PauliPolynomial] = []
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(coefficient_tolerance):
            continue
        nq = int(term.nqubit())
        out.append(PauliPolynomial(repr_mode, [PauliTerm(nq, ps=term.pw2strng(), pc=coeff)]))
    return out

@dataclass(frozen=True)
class AnsatzTerm:
    """One parameterized unitary U_k(theta_k) := exp(-i theta_k * H_k)."""

    label: str
    polynomial: PauliPolynomial


class HubbardTermwiseAnsatz:
    """
    Term-wise Hubbard ansatz (HVA-like), aligned with future Trotter time dynamics.

    Per layer, append:
      (A) all hopping terms  H_{<i,j>,sigma}^{(t)}
      (B) all onsite terms   H_i^{(U)}
      (C) all potential terms H_{i,sigma}^{(v)} (optional, only if v_i != 0)

    Full ansatz: reps repetitions of (A)->(B)->(C).
    """

    def __init__(
        self,
        dims: Dims,
        t: float,
        U: float,
        *,
        v: Optional[Union[float, Sequence[float], Dict[int, float]]] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_potential_terms: bool = True,
    ):
        self.dims = dims
        self.n_sites = n_sites_from_dims(dims)
        self.nq = 2 * int(self.n_sites)

        self.t = float(t)
        self.U = float(U)
        self.repr_mode = repr_mode
        self.indexing = indexing

        self.edges = list(edges) if edges is not None else bravais_nearest_neighbor_edges(dims, pbc=pbc)
        self.v_list = _parse_site_potential(v, n_sites=int(self.n_sites))
        self.reps = int(reps)
        if self.reps <= 0:
            log.error("reps must be positive")

        self.include_potential_terms = bool(include_potential_terms)

        self.base_terms: List[AnsatzTerm] = []
        self._build_base_terms()

        self.num_parameters = self.reps * len(self.base_terms)

    def _build_base_terms(self) -> None:
        nq = int(self.nq)
        n_sites = int(self.n_sites)

        for (i, j) in self.edges:
            for spin in (SPIN_UP, SPIN_DN):
                p_i = mode_index(int(i), int(spin), indexing=self.indexing, n_sites=n_sites)
                p_j = mode_index(int(j), int(spin), indexing=self.indexing, n_sites=n_sites)
                poly = hubbard_hop_term(nq, p_i, p_j, t=self.t, repr_mode=self.repr_mode)
                self.base_terms.append(AnsatzTerm(label=f"hop(i={i},j={j},spin={spin})", polynomial=poly))

        for i in range(n_sites):
            p_up = mode_index(i, SPIN_UP, indexing=self.indexing, n_sites=n_sites)
            p_dn = mode_index(i, SPIN_DN, indexing=self.indexing, n_sites=n_sites)
            poly = hubbard_onsite_term(nq, p_up, p_dn, U=self.U, repr_mode=self.repr_mode)
            self.base_terms.append(AnsatzTerm(label=f"onsite(i={i})", polynomial=poly))

        if self.include_potential_terms:
            for i in range(n_sites):
                vi = float(self.v_list[i])
                if abs(vi) < 1e-15:
                    continue
                for spin in (SPIN_UP, SPIN_DN):
                    p_mode = mode_index(i, spin, indexing=self.indexing, n_sites=n_sites)
                    poly = hubbard_potential_term(nq, p_mode, v_i=vi, repr_mode=self.repr_mode)
                    self.base_terms.append(AnsatzTerm(label=f"pot(i={i},spin={spin})", polynomial=poly))

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            log.error("theta has wrong length for this ansatz")
        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for term in self.base_terms:
                psi = apply_exp_pauli_polynomial(
                    psi,
                    term.polynomial,
                    float(theta[k]),
                    ignore_identity=ignore_identity,
                    coefficient_tolerance=coefficient_tolerance,
                    sort_terms=sort_terms,
                )
                k += 1
        return psi


class HubbardLayerwiseAnsatz(HubbardTermwiseAnsatz):
    """
    Layer-wise Hubbard ansatz with shared parameters per physical contribution.

    Per layer, apply:
      1) all hopping terms with shared theta_hop
      2) all onsite terms with shared theta_u
      3) all potential terms with shared theta_v (optional)

    This keeps the term-product structure (future Trotter-friendly) while tying
    parameters at the layer level.

    Parameter count per layer:
      - 2 when potential block is absent
      - 3 when potential block is present
    """

    def _build_base_terms(self) -> None:
        nq = int(self.nq)
        n_sites = int(self.n_sites)

        hop_terms: List[AnsatzTerm] = []
        hop_poly: Optional[PauliPolynomial] = None
        for (i, j) in self.edges:
            for spin in (SPIN_UP, SPIN_DN):
                p_i = mode_index(int(i), int(spin), indexing=self.indexing, n_sites=n_sites)
                p_j = mode_index(int(j), int(spin), indexing=self.indexing, n_sites=n_sites)
                poly = hubbard_hop_term(nq, p_i, p_j, t=self.t, repr_mode=self.repr_mode)
                hop_terms.append(AnsatzTerm(label=f"hop(i={i},j={j},spin={spin})", polynomial=poly))
                hop_poly = poly if hop_poly is None else (hop_poly + poly)

        onsite_terms: List[AnsatzTerm] = []
        onsite_poly: Optional[PauliPolynomial] = None
        for i in range(n_sites):
            p_up = mode_index(i, SPIN_UP, indexing=self.indexing, n_sites=n_sites)
            p_dn = mode_index(i, SPIN_DN, indexing=self.indexing, n_sites=n_sites)
            poly = hubbard_onsite_term(nq, p_up, p_dn, U=self.U, repr_mode=self.repr_mode)
            onsite_terms.append(AnsatzTerm(label=f"onsite(i={i})", polynomial=poly))
            onsite_poly = poly if onsite_poly is None else (onsite_poly + poly)

        potential_terms: List[AnsatzTerm] = []
        potential_poly: Optional[PauliPolynomial] = None
        if self.include_potential_terms:
            for i in range(n_sites):
                vi = float(self.v_list[i])
                if abs(vi) < 1e-15:
                    continue
                for spin in (SPIN_UP, SPIN_DN):
                    p_mode = mode_index(i, spin, indexing=self.indexing, n_sites=n_sites)
                    poly = hubbard_potential_term(nq, p_mode, v_i=vi, repr_mode=self.repr_mode)
                    potential_terms.append(AnsatzTerm(label=f"pot(i={i},spin={spin})", polynomial=poly))
                    potential_poly = poly if potential_poly is None else (potential_poly + poly)

        self.base_terms = []
        self.layer_term_groups: List[Tuple[str, List[AnsatzTerm]]] = []
        if hop_terms and hop_poly is not None:
            self.layer_term_groups.append(("hop_layer", hop_terms))
            self.base_terms.append(AnsatzTerm(label="hop_layer", polynomial=hop_poly))
        if onsite_terms and onsite_poly is not None:
            self.layer_term_groups.append(("onsite_layer", onsite_terms))
            self.base_terms.append(AnsatzTerm(label="onsite_layer", polynomial=onsite_poly))
        if potential_terms and potential_poly is not None:
            self.layer_term_groups.append(("potential_layer", potential_terms))
            self.base_terms.append(AnsatzTerm(label="potential_layer", polynomial=potential_poly))

        if not self.base_terms:
            log.error("HubbardLayerwiseAnsatz produced no layer terms")

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            log.error("theta has wrong length for this ansatz")
        if not hasattr(self, "layer_term_groups"):
            log.error("HubbardLayerwiseAnsatz missing layer term groups")

        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for _label, group_terms in self.layer_term_groups:
                shared_theta = float(theta[k])
                for term in group_terms:
                    psi = apply_exp_pauli_polynomial(
                        psi,
                        term.polynomial,
                        shared_theta,
                        ignore_identity=ignore_identity,
                        coefficient_tolerance=coefficient_tolerance,
                        sort_terms=sort_terms,
                    )
                k += 1
        return psi


def hubbard_holstein_reference_state(
    *,
    dims: Dims,
    num_particles: Optional[Tuple[int, int]] = None,
    n_ph_max: int,
    boson_encoding: str = "binary",
    indexing: str = "blocked",
) -> np.ndarray:
    """
    HH reference state = fermionic HF determinant tensor phonon vacuum.

    Bitstring ordering follows q_(n-1)...q_0:
      full_bitstring = \"0\" * n_bos + hf_fermion_bitstring.
    """
    n_sites = int(n_sites_from_dims(dims))
    n_ferm = 2 * n_sites
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    n_bos = n_sites * qpb
    n_total = n_ferm + n_bos

    if num_particles is None:
        num_particles_i = half_filled_num_particles(n_sites)
    else:
        num_particles_i = (int(num_particles[0]), int(num_particles[1]))

    hf_fermion_bitstring = str(
        hartree_fock_bitstring(
            n_sites=n_sites,
            num_particles=num_particles_i,
            indexing=str(indexing),
        )
    )
    full_bitstring = ("0" * n_bos) + hf_fermion_bitstring
    return basis_state(n_total, full_bitstring)


class HubbardHolsteinLayerwiseAnsatz:
    """
    Layer-wise HH ansatz with shared parameters per physical contribution group.

    Per layer, apply groups in deterministic order:
      1) hopping
      2) onsite-U
      3) fermion potential (optional)
      4) phonon energy
      5) e-ph coupling
      6) HH drive (optional)
    """

    def __init__(
        self,
        dims: Dims,
        J: float,
        U: float,
        omega0: float,
        g: float,
        n_ph_max: int,
        *,
        boson_encoding: str = "binary",
        v: SitePotential = None,
        v_t: TimePotential = None,
        v0: SitePotential = None,
        t_eval: Optional[float] = None,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        edges: Optional[Sequence[Tuple[int, int]]] = None,
        pbc: Union[bool, Sequence[bool]] = True,
        include_zero_point: bool = True,
        coefficient_tolerance: float = 1e-12,
        sort_terms: bool = True,
    ):
        self.dims = dims
        self.n_sites = int(n_sites_from_dims(dims))
        self.n_ferm = 2 * self.n_sites
        self.n_ph_max = int(n_ph_max)
        self.boson_encoding = str(boson_encoding)
        self.qpb = int(boson_qubits_per_site(self.n_ph_max, self.boson_encoding))
        self.n_total = self.n_ferm + self.n_sites * self.qpb
        self.nq = int(self.n_total)

        self.J = float(J)
        self.U = float(U)
        self.omega0 = float(omega0)
        self.g = float(g)
        self.v_list = _parse_site_potential(v, n_sites=self.n_sites)
        self.v_t = v_t
        self.v0 = v0
        self.t_eval = t_eval
        self.include_zero_point = bool(include_zero_point)

        self.repr_mode = str(repr_mode)
        self.indexing = str(indexing)
        self.edges = list(edges) if edges is not None else bravais_nearest_neighbor_edges(dims, pbc=pbc)
        self.reps = int(reps)
        if self.reps <= 0:
            log.error("reps must be positive")

        self.coefficient_tolerance = float(coefficient_tolerance)
        self.sort_terms = bool(sort_terms)

        self.base_terms: List[AnsatzTerm] = []
        self.layer_term_groups: List[Tuple[str, List[AnsatzTerm]]] = []
        self._build_base_terms()
        self.num_parameters = self.reps * len(self.base_terms)

    def _poly_group(
        self,
        label: str,
        poly: PauliPolynomial,
    ) -> None:
        term_polys = _single_term_polynomials_sorted(
            poly,
            repr_mode=self.repr_mode,
            coefficient_tolerance=self.coefficient_tolerance,
        )
        if not term_polys:
            return
        group_terms: List[AnsatzTerm] = []
        group_poly: Optional[PauliPolynomial] = None
        for i, term_poly in enumerate(term_polys):
            group_terms.append(AnsatzTerm(label=f"{label}_term_{i}", polynomial=term_poly))
            group_poly = term_poly if group_poly is None else (group_poly + term_poly)
        assert group_poly is not None
        self.layer_term_groups.append((label, group_terms))
        self.base_terms.append(AnsatzTerm(label=label, polynomial=group_poly))

    def _build_base_terms(self) -> None:
        hop_poly = build_hubbard_kinetic(
            dims=self.dims,
            t=self.J,
            repr_mode=self.repr_mode,
            indexing=self.indexing,
            edges=self.edges,
            pbc=True,
            nq_override=self.n_total,
        )
        self._poly_group("hop_layer", hop_poly)

        onsite_poly = build_hubbard_onsite(
            dims=self.dims,
            U=self.U,
            repr_mode=self.repr_mode,
            indexing=self.indexing,
            nq_override=self.n_total,
        )
        self._poly_group("onsite_layer", onsite_poly)

        potential_poly = build_hubbard_potential(
            dims=self.dims,
            v=self.v_list,
            repr_mode=self.repr_mode,
            indexing=self.indexing,
            nq_override=self.n_total,
        )
        self._poly_group("potential_layer", potential_poly)

        phonon_poly = build_holstein_phonon_energy(
            dims=self.dims,
            omega0=self.omega0,
            n_ph_max=self.n_ph_max,
            boson_encoding=self.boson_encoding,
            repr_mode=self.repr_mode,
            tol=self.coefficient_tolerance,
            zero_point=self.include_zero_point,
        )
        self._poly_group("phonon_layer", phonon_poly)

        eph_poly = build_holstein_coupling(
            dims=self.dims,
            g=self.g,
            n_ph_max=self.n_ph_max,
            boson_encoding=self.boson_encoding,
            repr_mode=self.repr_mode,
            indexing=self.indexing,
            tol=self.coefficient_tolerance,
        )
        self._poly_group("eph_layer", eph_poly)

        if self.v_t is not None or self.v0 is not None:
            drive_poly = build_hubbard_holstein_drive(
                dims=self.dims,
                v_t=self.v_t,
                v0=self.v0,
                t=self.t_eval,
                repr_mode=self.repr_mode,
                indexing=self.indexing,
                nq_override=self.n_total,
            )
            self._poly_group("drive_layer", drive_poly)

        if not self.base_terms:
            log.error("HubbardHolsteinLayerwiseAnsatz produced no layer terms")

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
        if int(theta.size) != int(self.num_parameters):
            log.error("theta has wrong length for this ansatz")
        if not hasattr(self, "layer_term_groups"):
            log.error("HubbardHolsteinLayerwiseAnsatz missing layer term groups")
        if int(psi_ref.size) != (1 << int(self.nq)):
            log.error("psi_ref length must be 2^nq for HubbardHolsteinLayerwiseAnsatz")

        coeff_tol = self.coefficient_tolerance if coefficient_tolerance is None else float(coefficient_tolerance)
        sort_flag = self.sort_terms if sort_terms is None else bool(sort_terms)

        psi = np.array(psi_ref, copy=True)
        k = 0
        for _ in range(self.reps):
            for _label, group_terms in self.layer_term_groups:
                shared_theta = float(theta[k])
                for term in group_terms:
                    psi = apply_exp_pauli_polynomial(
                        psi,
                        term.polynomial,
                        shared_theta,
                        ignore_identity=ignore_identity,
                        coefficient_tolerance=coeff_tol,
                        sort_terms=sort_flag,
                    )
                k += 1
        return psi


class HardcodedUCCSDAnsatz:
    """
    Hardcoded UCCSD-style ansatz built directly from fermionic ladder operators
    mapped through the local JW primitives.

    U(theta) = prod_k exp(-i theta_k G_k),
    where each G_k is Hermitian and corresponds to i(T_k - T_k^dagger)
    for single or double excitations relative to a Hartree-Fock reference sector.
    """

    def __init__(
        self,
        dims: Dims,
        num_particles: Tuple[int, int],
        *,
        reps: int = 1,
        repr_mode: str = "JW",
        indexing: str = "blocked",
        include_singles: bool = True,
        include_doubles: bool = True,
    ):
        self.dims = dims
        self.n_sites = n_sites_from_dims(dims)
        self.nq = 2 * int(self.n_sites)

        self.num_particles = (int(num_particles[0]), int(num_particles[1]))
        self.reps = int(reps)
        if self.reps <= 0:
            log.error("reps must be positive")

        self.repr_mode = repr_mode
```

## File: pipelines/hardcoded_hubbard_pipeline.py

### Updated imports
```python

from pydephasing.quantum.hartree_fock_reference_state import hartree_fock_statevector
from pydephasing.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
```

### Updated VQE namespace + HH ansatz dispatch
```python
def _load_hardcoded_vqe_namespace() -> dict[str, Any]:
    from pydephasing.quantum import vqe_latex_python_pairs_test as vqe_mod

    ns: dict[str, Any] = {name: getattr(vqe_mod, name) for name in dir(vqe_mod)}

    required = [
        "half_filled_num_particles",
        "hartree_fock_bitstring",
        "basis_state",
        "HubbardTermwiseAnsatz",
        "HardcodedUCCSDAnsatz",
        "HubbardLayerwiseAnsatz",
        "HubbardHolsteinLayerwiseAnsatz",
        "HardcodedUCCSDLayerwiseAnsatz",
        "hubbard_holstein_reference_state",
        "exact_ground_energy_sector",
        "vqe_minimize",
    ]
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError(f"Missing required VQE notebook symbols: {missing}")
    return ns


def _run_hardcoded_vqe(
    *,
    num_sites: int,
    ordering: str,
    h_poly: Any,
    ansatz_name: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    reps: int,
    restarts: int,
    seed: int,
    maxiter: int,
) -> tuple[dict[str, Any], np.ndarray]:
    ns = _load_hardcoded_vqe_namespace()
    num_particles = tuple(ns["half_filled_num_particles"](int(num_sites)))
    hf_bits = str(ns["hartree_fock_bitstring"](n_sites=int(num_sites), num_particles=num_particles, indexing=ordering))

    ansatz_name_s = str(ansatz_name).lower()
    if ansatz_name_s == "uccsd":
        nq = 2 * int(num_sites)
        psi_ref = np.asarray(ns["basis_state"](nq, hf_bits), dtype=complex)
        ansatz = ns["HardcodedUCCSDLayerwiseAnsatz"](
            dims=int(num_sites),
            num_particles=num_particles,
            reps=int(reps),
            repr_mode="JW",
            indexing=ordering,
            include_singles=True,
            include_doubles=True,
        )
        method_name = "hardcoded_uccsd_notebook_statevector"
    elif ansatz_name_s == "hva":
        nq = 2 * int(num_sites)
        psi_ref = np.asarray(ns["basis_state"](nq, hf_bits), dtype=complex)
        ansatz = ns["HubbardLayerwiseAnsatz"](
            dims=int(num_sites),
            t=float(t),
            U=float(u),
            v=float(dv),
            reps=int(reps),
            repr_mode="JW",
            indexing=ordering,
            pbc=(str(boundary) == "periodic"),
            include_potential_terms=True,
        )
        method_name = "hardcoded_hva_notebook_statevector"
    elif ansatz_name_s == "hh_hva":
        ansatz = ns["HubbardHolsteinLayerwiseAnsatz"](
            dims=int(num_sites),
            J=float(t),
            U=float(u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            v=float(dv),
            v_t=None,
            v0=None,
            t_eval=None,
            reps=int(reps),
            repr_mode="JW",
            indexing=ordering,
            pbc=(str(boundary) == "periodic"),
            include_zero_point=True,
        )
        psi_ref = np.asarray(
            ns["hubbard_holstein_reference_state"](
                dims=int(num_sites),
                num_particles=num_particles,
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                indexing=ordering,
            ),
            dtype=complex,
        )
        method_name = "hardcoded_hh_hva_notebook_statevector"
    else:
        raise ValueError(f"Unsupported ansatz '{ansatz_name}'. Expected one of: uccsd, hva, hh_hva.")

    expected_nq = int(getattr(ansatz, "nq", int(round(math.log2(psi_ref.size)))))
    if int(psi_ref.size) != (1 << expected_nq):
        raise ValueError("Reference-state size is inconsistent with ansatz qubit register.")
    h_terms = list(h_poly.return_polynomial())
    if h_terms:
        h_nq = int(h_terms[0].nqubit())
        if h_nq != expected_nq:
            raise ValueError(
                f"Hamiltonian/ansatz qubit mismatch: H has {h_nq} qubits, ansatz expects {expected_nq}."
            )

    result = ns["vqe_minimize"](
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(restarts),
        seed=int(seed),
        maxiter=int(maxiter),
        method="SLSQP",
```

### Updated trajectory qubit sizing for HH-compatible runs
```python
def _simulate_trajectory(
    *,
    num_sites: int,
    psi0: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = int(len(ordered_labels_exyz[0])) if ordered_labels_exyz else int(round(math.log2(psi0.size)))
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
    times = np.linspace(0.0, float(t_final), int(num_times))
```

### Added HH CLI arguments
```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardcoded-first Hubbard pipeline runner.")
    parser.add_argument("--L", type=int, required=True, help="Number of lattice sites.")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping coefficient.")
    parser.add_argument("--u", type=float, default=4.0, help="Onsite interaction U.")
    parser.add_argument("--dv", type=float, default=0.0, help="Uniform local potential term v (Hv = -v n).")
    parser.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--suzuki-order", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=64)
    parser.add_argument("--term-order", choices=["native", "sorted"], default="sorted")

    parser.add_argument("--vqe-reps", type=int, default=1)
    parser.add_argument("--vqe-restarts", type=int, default=1)
    parser.add_argument("--vqe-seed", type=int, default=7)
    parser.add_argument("--vqe-maxiter", type=int, default=120)
    parser.add_argument("--vqe-ansatz", choices=["uccsd", "hva", "hh_hva"], default="uccsd")
    parser.add_argument("--omega0", type=float, default=0.0, help="Holstein phonon frequency omega0.")
    parser.add_argument("--g-ep", type=float, default=0.0, help="Holstein electron-phonon coupling g.")
    parser.add_argument("--n-ph-max", type=int, default=1, help="Local phonon cutoff n_ph_max (d=n_ph_max+1).")
    parser.add_argument("--boson-encoding", choices=["binary"], default="binary")

    parser.add_argument("--qpe-eval-qubits", type=int, default=6)
```

### Updated main() HH Hamiltonian selection + HH reference-state handling + metadata
```python
def main() -> None:
    args = parse_args()
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (artifacts_dir / f"hardcoded_pipeline_L{args.L}.json")
    output_pdf = args.output_pdf or (artifacts_dir / f"hardcoded_pipeline_L{args.L}.pdf")

    if str(args.vqe_ansatz).lower() == "hh_hva":
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=int(args.L),
            J=float(args.t),
            U=float(args.u),
            omega0=float(args.omega0),
            g=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            v_t=None,
            v0=None,
            t_eval=None,
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
            include_zero_point=True,
        )
    else:
        h_poly = build_hubbard_hamiltonian(
            dims=int(args.L),
            t=float(args.t),
            U=float(args.u),
            v=float(args.dv),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
        )

    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    if args.term_order == "native":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)

    hmat = _build_hamiltonian_matrix(coeff_map_exyz)
    evals, evecs = np.linalg.eigh(hmat)
    gs_idx = int(np.argmin(evals))
    gs_energy_exact = float(np.real(evals[gs_idx]))
    psi_exact_ground = _normalize_state(np.asarray(evecs[:, gs_idx], dtype=complex).reshape(-1))

    vqe_payload: dict[str, Any]
    try:
        vqe_payload, psi_vqe = _run_hardcoded_vqe(
            num_sites=int(args.L),
            ordering=str(args.ordering),
            h_poly=h_poly,
            ansatz_name=str(args.vqe_ansatz),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            boundary=str(args.boundary),
            omega0=float(args.omega0),
            g_ep=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            reps=int(args.vqe_reps),
            restarts=int(args.vqe_restarts),
            seed=int(args.vqe_seed),
            maxiter=int(args.vqe_maxiter),
        )
    except Exception as exc:
        vqe_payload = {
            "success": False,
            "method": f"hardcoded_{str(args.vqe_ansatz)}_notebook_statevector",
            "ansatz": str(args.vqe_ansatz),
            "parameterization": "layerwise",
            "energy": None,
            "exact_filtered_energy": None,
            "error": str(exc),
        }
        psi_vqe = psi_exact_ground

    num_particles = _half_filled_particles(int(args.L))
    if str(args.vqe_ansatz).lower() == "hh_hva":
        ns = _load_hardcoded_vqe_namespace()
        psi_hf = _normalize_state(
            np.asarray(
                ns["hubbard_holstein_reference_state"](
                    dims=int(args.L),
                    num_particles=num_particles,
                    n_ph_max=int(args.n_ph_max),
                    boson_encoding=str(args.boson_encoding),
                    indexing=str(args.ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi_hf = _normalize_state(
            np.asarray(
                hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
                dtype=complex,
            ).reshape(-1)
        )

    if args.initial_state_source == "vqe" and bool(vqe_payload.get("success", False)):
        psi0 = psi_vqe
    elif args.initial_state_source == "vqe":
        raise RuntimeError("Requested --initial-state-source vqe but hardcoded VQE statevector is unavailable.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
    else:
        psi0 = psi_exact_ground

    if args.skip_qpe:
        qpe_payload = {
            "success": False,
            "method": "qpe_skipped",
            "energy_estimate": None,
            "phase": None,
            "skipped": True,
            "reason": "--skip-qpe enabled",
            "num_evaluation_qubits": int(args.qpe_eval_qubits),
            "shots": int(args.qpe_shots),
        }
    else:
        qpe_payload = _run_qpe_adapter_qiskit(
            coeff_map_exyz=coeff_map_exyz,
            psi_init=psi0,
            eval_qubits=int(args.qpe_eval_qubits),
            shots=int(args.qpe_shots),
            seed=int(args.qpe_seed),
        )

    trajectory, _exact_states = _simulate_trajectory(
        num_sites=int(args.L),
        psi0=psi0,
        hmat=hmat,
        ordered_labels_exyz=ordered_labels_exyz,
        coeff_map_exyz=coeff_map_exyz,
        trotter_steps=int(args.trotter_steps),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        suzuki_order=int(args.suzuki_order),
    )

    sanity = {
        "jw_reference": _reference_sanity(
            num_sites=int(args.L),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            boundary=str(args.boundary),
            ordering=str(args.ordering),
            coeff_map_exyz=coeff_map_exyz,
        )
    }

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded",
        "settings": {
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "dv": float(args.dv),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "term_order": str(args.term_order),
            "vqe_ansatz": str(args.vqe_ansatz),
```

## File: pydephasing/quantum/test_hubbard_holstein.py

### New test module (full file)
```python
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
```

## File: pydephasing/quantum/test_hubbard_holstein_vqe_ansatz.py

### New HH VQE ansatz test module (full file)
```python
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
```
