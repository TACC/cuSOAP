
"""
Pytorch-Based SOAP generator: GPU-first, DScribe-compatible SOAP descriptor implementation using PyTorch + sphericart.

Overview
--------
- Match the *functionality* and *feature ordering* of the DScribe SOAP
  (SOAP( r_cut, n_max, l_max, sigma, rbf, weighting, crossover, average, species, periodic, sparse, dtype )).
- Provide *numerical derivatives* (finite differences) with the same output shape conventions as DScribe.
- Run on **GPU end-to-end** for the heavy parts: spherical harmonics, radial basis evaluation, accumulation, power spectrum.
  (Neighbor list can use torch_cluster on GPU when installed; otherwise a chunked torch.cdist fallback is used.)
--------

Dependencies:
- torch
- sphericart (torch bindings). For CUDA spherical harmonics, build/install sphericart with CUDA support.
Optional:
- torch_cluster (for fast GPU neighbor search: radius / radius_graph)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import time
import torch



# -----------------------------
# Lightweight profiler (per-call segment timings)
# -----------------------------
class Profiler:
    """
    Segment profiler with optional CUDA synchronization for accurate GPU timings.

    Usage:
        prof = Profiler(device)
        with prof.section("neighbor"):
            ...
        times = prof.times
    """
    def __init__(self, device: torch.device):
        self.device = device
        self._cuda = (str(device).startswith("cuda") and torch.cuda.is_available())
        self.times: Dict[str, float] = {}

    def _sync(self):
        if self._cuda:
            torch.cuda.synchronize()

    def section(self, name: str):
        from contextlib import contextmanager
        @contextmanager
        def _ctx():
            self._sync()
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._sync()
                t1 = time.perf_counter()
                self.times[name] = self.times.get(name, 0.0) + (t1 - t0)
        return _ctx()



# ----------------------------
# Helpers: dtype / device / system adapter
# ----------------------------

def _torch_dtype_from_str(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


def _as_torch(x, device, dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _as_long(x, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long)
    return torch.tensor(x, device=device, dtype=torch.long)


def _as_bool(x, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.bool)
    return torch.tensor(x, device=device, dtype=torch.bool)


def system_to_tensors(
    system: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Accepts either:
      - ASE.Atoms-like object (get_positions, get_atomic_numbers, get_cell, get_pbc)
      - dict with keys: positions, atomic_numbers, optional cell, pbc
      - tuple/list: (positions, atomic_numbers, cell?, pbc?)
    Returns: positions (N,3), Z (N,), cell (3,3) or None, pbc (3,) bool or None
    """
    if isinstance(system, (tuple, list)):
        if len(system) < 2:
            raise ValueError("Tuple/list system must be (positions, atomic_numbers, [cell], [pbc]).")
        positions = _as_torch(system[0], device, dtype)
        Z = _as_long(system[1], device)
        cell = _as_torch(system[2], device, dtype) if len(system) >= 3 and system[2] is not None else None
        pbc = _as_bool(system[3], device) if len(system) >= 4 and system[3] is not None else None
        return positions, Z, cell, pbc

    if isinstance(system, dict):
        positions = _as_torch(system["positions"], device, dtype)
        Z = _as_long(system["atomic_numbers"], device)
        cell = _as_torch(system.get("cell"), device, dtype) if system.get("cell") is not None else None
        pbc = _as_bool(system.get("pbc"), device) if system.get("pbc") is not None else None
        return positions, Z, cell, pbc

    # ASE.Atoms-like
    if hasattr(system, "get_positions") and hasattr(system, "get_atomic_numbers"):
        # Pull to CPU numpy, then to torch (I/O only). Computation is torch/GPU.
        positions = _as_torch(system.get_positions(), device, dtype)
        Z = _as_long(system.get_atomic_numbers(), device)
        cell = None
        pbc = None
        if hasattr(system, "get_cell"):
            cell = _as_torch(system.get_cell().array, device, dtype)
        if hasattr(system, "get_pbc"):
            pbc = _as_bool(system.get_pbc(), device)
        return positions, Z, cell, pbc

    raise TypeError("Unsupported system type. Provide ASE.Atoms-like, dict, or (positions, Z, cell, pbc).")


# ----------------------------
# Linear algebra: Löwdin orthonormalization on GPU
# ----------------------------

def _lowdin_invsqrt(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """S^{-1/2} for SPD S using eigen-decomposition, works on GPU."""
    w, V = torch.linalg.eigh(S)
    w = torch.clamp(w, min=eps)
    return V @ torch.diag(w.rsqrt()) @ V.transpose(-1, -2)


# ----------------------------
# Quadrature: Gauss–Legendre in pure torch (GPU capable)
# ----------------------------

def gauss_legendre(n: int, device=None, dtype=torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gauss–Legendre nodes/weights on [-1, 1] using Golub–Welsch algorithm.
    """
    device = device or torch.device("cpu")
    k = torch.arange(1, n, device=device, dtype=dtype)
    beta = k / torch.sqrt(4 * k * k - 1)
    J = torch.zeros((n, n), device=device, dtype=dtype)
    J.diagonal(1).copy_(beta)
    J.diagonal(-1).copy_(beta)
    x, V = torch.linalg.eigh(J)
    w = 2 * (V[0, :] ** 2)
    return x, w


# ----------------------------
# Periodic replication helpers (DScribe-like explicit extension)
# ----------------------------

def _perpendicular_lengths(cell: torch.Tensor) -> torch.Tensor:
    a, b, c = cell[0], cell[1], cell[2]
    v = torch.dot(a, torch.cross(b, c))
    bc = torch.cross(b, c)
    ca = torch.cross(c, a)
    ab = torch.cross(a, b)
    eps = 1e-12
    la = torch.abs(v) / (torch.linalg.norm(bc) + eps)
    lb = torch.abs(v) / (torch.linalg.norm(ca) + eps)
    lc = torch.abs(v) / (torch.linalg.norm(ab) + eps)
    return torch.stack([la, lb, lc], dim=0)


def _pbc_offsets(cell: torch.Tensor, pbc: torch.Tensor, cutoff: float) -> torch.Tensor:
    device = cell.device
    dtype = cell.dtype
    perps = _perpendicular_lengths(cell).to(device=device, dtype=dtype)
    n_rep = torch.zeros((3,), device=device, dtype=torch.long)
    for i in range(3):
        if bool(pbc[i]):
            n_rep[i] = int(torch.ceil(torch.tensor(cutoff, device=device, dtype=dtype) / (perps[i] + 1e-12)).item())
        else:
            n_rep[i] = 0
    ranges = []
    for i in range(3):
        ranges.append(torch.arange(-n_rep[i], n_rep[i] + 1, device=device, dtype=torch.long))
    ox, oy, oz = torch.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
    return torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=-1)


# ----------------------------
# Weighting: DScribe-compatible poly / pow / exp + w0 and threshold->r_cut inference
# ----------------------------

@dataclass
class Weighting:
    """
    DScribe-like weighting.

    Supports:
      - function=None: only w0 is used (w=w0 at r==0 else 1)
      - function="poly": w=c*(1 + 2*(r/r0)^3 - 3*(r/r0)^2)^m for r<=r0 else 0
      - function="pow":  w=c/(d + (r/r0)^m)
      - function="exp":  w=c/(d + exp(-r/r0))

    threshold (not stored) is only used to infer r_cut when r_cut is not provided.
    """
    function: Optional[str] = None
    r0: float = 1.0
    c: float = 1.0
    d: float = 1.0
    m: float = 1.0
    w0: float = 1.0

    def infer_rcut(self, threshold: float = 1e-2) -> float:
        if self.function is None:
            raise ValueError("Cannot infer r_cut when weighting.function is None; provide r_cut explicitly.")
        fn = self.function
        if fn == "poly":
            return float(self.r0)
        elif fn == "pow":
            val = self.c / threshold - self.d
            if val <= 0:
                raise ValueError("Invalid weighting: c/threshold - d must be >0 for pow.")
            return float(self.r0 * (val ** (1.0 / self.m)))
        elif fn == "exp":
            val = self.c / threshold - self.d
            if val <= 0:
                raise ValueError("Invalid weighting: c/threshold - d must be >0 for exp.")
            return float(-self.r0 * math.log(val))
        else:
            raise ValueError(f"Unknown weighting function: {fn}")

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        if self.function is None:
            w = torch.ones_like(r)
            w0 = torch.tensor(self.w0, device=r.device, dtype=r.dtype)
            return torch.where(r < 1e-12, w0, w)

        fn = self.function
        r0 = self.r0
        c = self.c
        d = self.d
        m = self.m

        if fn == "poly":
            rr0 = r / r0
            rr02 = rr0 * rr0
            rr03 = rr02 * rr0
            base = 1.0 + 2.0 * rr03 - 3.0 * rr02
            w = torch.zeros_like(r)
            mask = r <= r0
            w = torch.where(mask, c * torch.clamp(base, min=0.0) ** m, w)
        elif fn == "pow":
            w = c / (d + (r / r0) ** m)
        elif fn == "exp":
            w = c / (d + torch.exp(-r / r0))
        else:
            raise ValueError(f"Unknown weighting function: {fn}")

        w0 = torch.tensor(self.w0, device=r.device, dtype=r.dtype)
        return torch.where(r < 1e-12, w0, w)


# ----------------------------
# Stable scaled modified spherical Bessel i_l(t) * exp(-t) for polynomial backend
# ----------------------------

def _i0e(t: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    tt = torch.clamp(t, min=eps)
    e2 = torch.exp(-2 * tt)
    return (1 - e2) / (2 * tt)


def _i1e(t: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    tt = torch.clamp(t, min=eps)
    e2 = torch.exp(-2 * tt)
    return (1 - e2) / (2 * tt * tt) - (1 + e2) / (2 * tt)


def modified_spherical_bessel_ie(l_max: int, t: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    tt = torch.clamp(t, min=eps)
    out = torch.zeros((l_max + 1,) + tt.shape, device=tt.device, dtype=tt.dtype)
    out[0] = _i0e(tt)
    if l_max == 0:
        return out
    out[1] = _i1e(tt)
    for l in range(1, l_max):
        out[l + 1] = out[l - 1] - (2 * l + 1) / tt * out[l]
    return out


# ----------------------------
# Core SOAP class
# ----------------------------

class SOAP:
    """
    DScribe-like SOAP implemented in PyTorch + sphericart.

    Constructor signature matches the DScribe SOAP:
      SOAP(r_cut=None, n_max=None, l_max=None, sigma=1.0, rbf="gto", weighting=None,
           crossover=True, average="off", species=None, periodic=False, sparse=False, dtype="float64")
    """

    def __init__(
        self,
        r_cut: Optional[float] = None,
        n_max: Optional[int] = None,
        l_max: Optional[int] = None,
        sigma: float = 1.0,
        rbf: str = "gto",
        weighting: Optional[Dict[str, Any]] = None,
        crossover: bool = True,
        average: str = "off",
        species: Optional[Sequence[Union[int, str]]] = None,
        periodic: bool = False,
        sparse: bool = False,
        dtype: str = "float64",
        device: Optional[Union[str, torch.device]] = None,
        quad_n: int = 100,
        max_num_neighbors: Optional[int] = None,
    ):
        if n_max is None or l_max is None:
            raise ValueError("n_max and l_max must be provided.")
        if species is None or len(species) == 0:
            raise ValueError("species must be provided (list of atomic numbers or symbols).")

        self._dtype_str = dtype
        self.dtype = _torch_dtype_from_str(dtype)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve species: allow chemical symbols if ASE is available
        species_Z: List[int] = []
        for s in species:
            if isinstance(s, str):
                try:
                    import ase.data
                    species_Z.append(int(ase.data.atomic_numbers[s]))
                except Exception as e:
                    raise ValueError(f"Cannot resolve species symbol '{s}'. Install ASE or use atomic numbers.") from e
            else:
                species_Z.append(int(s))

        self._atomic_numbers = torch.tensor(sorted(set(species_Z)), device=self.device, dtype=torch.long)
        self._atomic_number_set = set(self._atomic_numbers.tolist())
        self._Z_to_species_index = {int(z): i for i, z in enumerate(self._atomic_numbers.tolist())}
        self.n_species = int(self._atomic_numbers.numel())

        self._n_max = int(n_max)
        self._l_max = int(l_max)
        self._sigma = float(sigma)
        self._eta = 1.0 / (2.0 * self._sigma * self._sigma)
        self._rbf = str(rbf).lower()
        self.crossover = bool(crossover)
        self.average = str(average)
        self.periodic = bool(periodic)
        self.sparse = bool(sparse)
        self.quad_n = int(quad_n)
        self.max_num_neighbors = max_num_neighbors

        if self.average not in ("off", "inner", "outer", "cc"):
            raise ValueError("average must be one of: 'off', 'inner', 'outer', 'cc' (cc = projection coeffs).")

        # Weighting dict matches DScribe: may include threshold for r_cut inference
        self._weighting_dict = dict(weighting) if weighting is not None else None
        if self._weighting_dict is not None:
            thr = float(self._weighting_dict.get("threshold", 1e-2))
            wd = {k: v for k, v in self._weighting_dict.items() if k != "threshold"}
            self._weighting = Weighting(**wd)
        else:
            self._weighting = None
            thr = 1e-2

        if r_cut is None:
            if self._weighting is None:
                raise ValueError("Either r_cut or weighting must be provided.")
            self._r_cut = float(self._weighting.infer_rcut(threshold=thr))
        else:
            self._r_cut = float(r_cut)

        # DScribe cutoff padding for gaussian tails uses threshold=1e-3:
        self._cutoff_padding = float(self._sigma * math.sqrt(-2.0 * math.log(1e-3)))
        self._cutoff = self._r_cut + self._cutoff_padding

        # Precompute basis
        if self._rbf == "gto":
            self._init_gto_basis()
        elif self._rbf == "polynomial":
            self._init_poly_basis()
        else:
            raise ValueError("rbf must be 'gto' or 'polynomial'.")

        # sphericart
        self._init_sphericart()

        # Analytic self-contribution for r=0 (only l=0 survives for a Gaussian at the origin).
        self._init_self_contributions()

        # Precompute indices for fast flattening
        self._triu = torch.triu_indices(self._n_max, self._n_max, offset=0, device=self.device)
        # cache constant tensors on (device,dtype) to avoid per-call .to overhead
        self._const_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
        # scratch buffers for neighbor search / GEMM distance (to reduce allocations)
        self._scratch: Dict[Tuple[str, str, str], torch.Tensor] = {}
        # number of output features (DScribe-compatible ordering)
        self.n_features = int(self.get_number_of_features())

        # feature slice table for optimized power spectrum filling
        self._feat_slices: Optional[List[Tuple[int,int,int,bool,int,int]]] = None
        self._build_feature_slices()


    # ---- sphericart ----

    def _init_sphericart(self):
        try:
            import sphericart.torch as sct
        except Exception as e:
            raise ImportError("sphericart.torch is required. Install/build sphericart with torch bindings.") from e
        self._sct = sct
        self._Y = sct.SphericalHarmonics(self._l_max)

    # ---- basis ----

    def _init_gto_basis(self):
        # DScribe basis generation on CPU uses SciPy gamma/sqrtm; here we do torch float64 on GPU.
        device = self.device
        dtype = torch.float64
        a = torch.linspace(1.0, self._r_cut, self._n_max, device=device, dtype=dtype)
        thr = torch.tensor(1e-3, device=device, dtype=dtype)

        alphas = []
        betas = []
        for l in range(self._l_max + 1):
            alpha_l = -torch.log(thr / (a ** l)) / (a ** 2)
            m = alpha_l[:, None] + alpha_l[None, :]
            gamma_val = torch.exp(torch.lgamma(torch.tensor(l + 1.5, device=device, dtype=dtype)))
            S = 0.5 * gamma_val * (m ** (-(l + 1.5)))
            beta_l = _lowdin_invsqrt(S)
            alphas.append(alpha_l)
            betas.append(beta_l)
        self._alphas = torch.stack(alphas, dim=0)  # (l_max+1, n_max)
        self._betas = torch.stack(betas, dim=0)    # (l_max+1, n_max, n_max)

    def _init_poly_basis(self):
        device = self.device
        dtype = torch.float64
        n = self._n_max
        rc = torch.tensor(self._r_cut, device=device, dtype=dtype)

        S = torch.zeros((n, n), device=device, dtype=dtype)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                S[i - 1, j - 1] = (2.0 * (rc ** (7 + i + j))) / ((5 + i + j) * (6 + i + j) * (7 + i + j))
        betas = _lowdin_invsqrt(S)

        x, w = gauss_legendre(self.quad_n, device=device, dtype=dtype)
        rx = self._r_cut * 0.5 * (x + 1.0)
        wr = self._r_cut * 0.5 * w

        fs = torch.zeros((n, self.quad_n), device=device, dtype=dtype)
        rclip = torch.clamp(rx, 0.0, self._r_cut)
        for k in range(1, n + 1):
            fs[k - 1, :] = (self._r_cut - rclip) ** (k + 2)

        gss = betas @ fs  # (n_max, Q)

        self._rx = rx
        self._wr = wr
        self._gss = gss

    # ---- analytic self contribution (r=0) ----

    def _init_self_contributions(self):
        """Precompute the contribution of a neighbor Gaussian centered at r=0 (i.e., center==atom position).
        For a Gaussian at the origin, the density is spherically symmetric => only l=0 contributes.
        We add this term explicitly to avoid calling sphericart at xyz==0 (undefined angles -> NaNs).
        """
        device = self.device
        dtype = torch.float64  # store in high precision; cast at use-time
        y00 = 1.0 / math.sqrt(4.0 * math.pi)  # real Y_0^0 in the common orthonormal convention

        w0 = float(self._weighting.w0) if self._weighting is not None else 1.0
        eta = torch.tensor(self._eta, device=device, dtype=dtype)

        if self._rbf == "gto":
            # primitive radial coefficients at r=0 for l=0:
            # prim_n = w0 * π^(3/2) * (alpha_n + eta)^(-3/2)
            alpha0 = self._alphas[0].to(device=device, dtype=dtype)  # (n_max,)
            beta0 = self._betas[0].to(device=device, dtype=dtype)    # (n_max,n_max)
            p = alpha0 + eta
            prim = w0 * (math.pi ** 1.5) * torch.pow(p, -1.5)        # (n_max,)
            c0 = beta0 @ prim                                        # (n_max,)
            self._self_l0 = (c0 * y00).to(device=device, dtype=dtype) # (n_max,)
        else:
            # polynomial: coefficients are already orthonormalized (gss). For r=0 and l=0:
            # I_n = ∫ 4π r_x^2 exp(-eta r_x^2) g_n(r_x) dr_x
            rx = self._rx.to(device=device, dtype=dtype)  # (Q,)
            wr = self._wr.to(device=device, dtype=dtype)  # (Q,)
            gss = self._gss.to(device=device, dtype=dtype)  # (n_max,Q)
            common0 = (4.0 * math.pi) * (rx * rx) * torch.exp(-eta * (rx ** 2)) * wr  # (Q,)
            # (Q,) @ (Q,n_max) => (n_max,)
            I0 = common0 @ gss.transpose(0, 1)
            self._self_l0 = (w0 * I0 * y00).to(device=device, dtype=dtype)

    def _add_self_terms(self, coeffs: List[torch.Tensor], center_indices: torch.Tensor, Z: torch.Tensor) -> None:
        """Add precomputed self term to coeffs[0] for centers given as atomic indices."""
        if center_indices is None:
            return
        if coeffs is None or len(coeffs) == 0:
            return
        # l=0 block
        c0 = coeffs[0]  # (C,S,n_max,1)
        if c0.numel() == 0:
            return

        mask = center_indices >= 0
        if not torch.any(mask):
            return

        center_ids = torch.arange(center_indices.shape[0], device=center_indices.device, dtype=torch.long)[mask]
        atom_ids = center_indices[mask]
        # map atomic numbers to species indices
        sp = self._map_Z_to_species(Z[atom_ids])
        valid = sp >= 0
        if not torch.any(valid):
            return

        center_ids = center_ids[valid]
        sp = sp[valid]

        self_vec = self._self_l0.to(device=c0.device, dtype=c0.dtype)  # (n_max,)
        # Add to (center, species, n, m=0)
        c0[center_ids, sp, :, 0] = c0[center_ids, sp, :, 0] + self_vec[None, :]
        coeffs[0] = c0

    # ---- DScribe-like helper methods ----

    def get_number_of_features(self) -> int:
        if self.average == "cc":
            return self.n_species * self._n_max * (self._l_max + 1) ** 2

        if self.crossover:
            ch = self.n_species * self._n_max
            return int((ch * (ch + 1) // 2) * (self._l_max + 1))
        else:
            return int(self.n_species * (self._l_max + 1) * (self._n_max * (self._n_max + 1) // 2))

    def get_location(self, species_pair: Tuple[Union[int, str], Union[int, str]]) -> slice:
        """
        DScribe-like: return slice of feature vector for (Z1, Z2) block.
        Only for average != "cc".
        """
        if self.average == "cc":
            raise ValueError("get_location is not defined for average='cc' (projection coefficients).")
        if len(species_pair) != 2:
            raise ValueError("species_pair must be length-2.")

        def toZ(x):
            if isinstance(x, str):
                import ase.data
                return int(ase.data.atomic_numbers[x])
            return int(x)

        Z1 = toZ(species_pair[0])
        Z2 = toZ(species_pair[1])
        if Z1 not in self._atomic_number_set or Z2 not in self._atomic_number_set:
            raise ValueError("Species not present in descriptor species list.")

        i = self._Z_to_species_index[Z1]
        j = self._Z_to_species_index[Z2]
        if i > j:
            i, j = j, i

        # feature counts per block
        symm = (self._n_max * (self._n_max + 1) // 2) * (self._l_max + 1)
        unsymm = (self._n_max * self._n_max) * (self._l_max + 1)

        if self.crossover:
            # DScribe formula from soap.py get_location
            m_symm = i + int(j > i)
            m_unsymm = j + i * self.n_species - i * (i + 1) / 2 - m_symm
            start = int(m_symm * symm + m_unsymm * unsymm)
            size = symm if i == j else unsymm
            return slice(start, start + size)
        else:
            if i != j:
                raise ValueError("crossover=False: no cross-species blocks exist.")
            start = int(i * symm)
            return slice(start, start + symm)

    # ---- centers handling (DScribe-like) ----

    def prepare_centers(
        self,
        positions: torch.Tensor,
        centers: Optional[Sequence[Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DScribe-like: centers can be atomic indices (int) or cartesian coordinates (len-3).
        Returns:
          centers_xyz: (C,3)
          center_indices: (C,) long, atomic index if center was an index else -1
        """
        if centers is None:
            C = positions.shape[0]
            return positions, torch.arange(C, device=positions.device, dtype=torch.long)

        centers_xyz = []
        center_indices = []
        for c in centers:
            if isinstance(c, (int,)):
                idx = int(c)
                centers_xyz.append(positions[idx])
                center_indices.append(idx)
            else:
                cc = _as_torch(c, device=positions.device, dtype=positions.dtype).reshape(3)
                centers_xyz.append(cc)
                center_indices.append(-1)
        return torch.stack(centers_xyz, dim=0), torch.tensor(center_indices, device=positions.device, dtype=torch.long)

    # ---- neighbor list (GPU) ----

    def _map_Z_to_species(self, Z: torch.Tensor) -> torch.Tensor:
        maxZ = int(torch.max(Z).item()) if Z.numel() else 0
        lut = torch.full((maxZ + 1,), -1, device=Z.device, dtype=torch.long)
        for z, idx in self._Z_to_species_index.items():
            if z <= maxZ:
                lut[z] = idx
        return lut[Z.clamp(min=0, max=maxZ)]

    def _extend_periodic(self, positions: torch.Tensor, Z: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        offsets = _pbc_offsets(cell, pbc, self._cutoff)  # (n_img,3) ints
        shift = offsets.to(dtype=positions.dtype) @ cell  # (n_img,3)
        pos_ext = positions[None, :, :] + shift[:, None, :]
        pos_ext = pos_ext.reshape(-1, 3)
        Z_ext = Z.repeat(offsets.shape[0])
        return pos_ext, Z_ext


    def _get_scratch(self, name: str, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get (and possibly grow) a cached scratch tensor on (device,dtype).

        Used to reduce allocations in hot paths. Returned tensor may be larger than requested;
        callers should slice to requested shape.
        """
        key = (name, str(device), str(dtype))
        t = self._scratch.get(key)
        if t is None:
            t = torch.empty(shape, device=device, dtype=dtype)
            self._scratch[key] = t
            return t
        if len(t.shape) != len(shape) or any(int(t.shape[i]) < int(shape[i]) for i in range(len(shape))):
            new_shape = tuple(max(int(t.shape[i]), int(shape[i])) for i in range(len(shape)))
            t = torch.empty(new_shape, device=device, dtype=dtype)
            self._scratch[key] = t
        return t

    def _radius_edges(
        self,
        centers: torch.Tensor,        # (C,3)
        neigh_pos: torch.Tensor,      # (M,3)
        cutoff: float,
        batch_centers: Optional[torch.Tensor] = None,
        batch_neigh: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return edge indices (center_i, neigh_j) with ||r|| <= cutoff.

        Strategy:
        - If `torch_cluster` is available, use `torch_cluster.radius`.
        - Otherwise use chunked GEMM distance: dist2 = ||c||^2 + ||n||^2 - 2 c·n,
          reusing a cached (chunk,M) scratch buffer.
        """
        try:
            from torch_cluster import radius
            self._last_nl_backend = "torch_cluster"
            row, col = radius(
                x=neigh_pos,
                y=centers,
                r=cutoff,
                batch_x=batch_neigh,
                batch_y=batch_centers,
                max_num_neighbors=self.max_num_neighbors,
            )
            return col, row
        except Exception:
            self._last_nl_backend = "gemm"
            C = int(centers.shape[0])
            M = int(neigh_pos.shape[0])
            if C == 0 or M == 0:
                return (
                    torch.empty((0,), device=centers.device, dtype=torch.long),
                    torch.empty((0,), device=centers.device, dtype=torch.long),
                )

            device = centers.device
            dtype = centers.dtype
            chunk = 1024
            cutoff2 = float(cutoff) * float(cutoff)

            neigh_T = neigh_pos.transpose(0, 1).contiguous()          # (3,M)
            neigh_norm = (neigh_pos * neigh_pos).sum(dim=1)           # (M,)

            center_list: List[torch.Tensor] = []
            neigh_list: List[torch.Tensor] = []

            for s in range(0, C, chunk):
                e = min(s + chunk, C)
                cpos = centers[s:e]                                   # (c,3)
                c = int(cpos.shape[0])

                c_norm = (cpos * cpos).sum(dim=1)                     # (c,)

                buf = self._get_scratch("nl_prod", (c, M), device=device, dtype=dtype)
                prod = buf[:c, :M]
                torch.mm(cpos, neigh_T, out=prod)

                dist2 = prod
                dist2.mul_(-2.0)
                dist2.add_(c_norm[:, None])
                dist2.add_(neigh_norm[None, :])

                mask = dist2 <= cutoff2
                if mask.any():
                    idx = torch.nonzero(mask, as_tuple=False)
                    center_list.append(idx[:, 0] + s)
                    neigh_list.append(idx[:, 1])

            if len(center_list) == 0:
                return (
                    torch.empty((0,), device=device, dtype=torch.long),
                    torch.empty((0,), device=device, dtype=torch.long),
                )

            return torch.cat(center_list, dim=0), torch.cat(neigh_list, dim=0)


    def _build_neighbor_list(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        centers_xyz: torch.Tensor,
        cell: Optional[torch.Tensor],
        pbc: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          center_index: (E,)
          neigh_species: (E,) in [0, n_species)
          rvec: (E,3)
          r: (E,)
        """
        cutoff = self._cutoff

        if self.periodic:
            if cell is None or pbc is None:
                raise ValueError("periodic=True requires cell and pbc.")
            neigh_pos, neigh_Z = self._extend_periodic(positions, Z, cell, pbc)
        else:
            neigh_pos, neigh_Z = positions, Z

        neigh_species_all = self._map_Z_to_species(neigh_Z)
        keep = neigh_species_all >= 0
        neigh_pos = neigh_pos[keep]
        neigh_species_all = neigh_species_all[keep]

        center_index, neigh_index = self._radius_edges(centers_xyz, neigh_pos, cutoff)

        if center_index.numel() == 0:
            return (
                center_index,
                torch.empty((0,), device=positions.device, dtype=torch.long),
                torch.empty((0, 3), device=positions.device, dtype=positions.dtype),
                torch.empty((0,), device=positions.device, dtype=positions.dtype),
            )

        rvec = neigh_pos[neigh_index] - centers_xyz[center_index]
        r = torch.linalg.norm(rvec, dim=-1)
        neigh_species = neigh_species_all[neigh_index]

        # IMPORTANT: sphericart.torch.SphericalHarmonics is undefined at xyz==0 (angles undefined),
        # which can happen for "self" edges when centers coincide with an atom position.
        # We drop r~0 edges here and add the correct analytic self-contribution (l=0 only) later.
        eps_self = 1e-8 if r.dtype == torch.float32 else 1e-12
        mask = r > eps_self
        center_index = center_index[mask]
        neigh_species = neigh_species[mask]
        rvec = rvec[mask]
        r = r[mask]

        return center_index, neigh_species, rvec, r

    # ---- coefficients (GTO/Polynomial) ----

    def _weights(self, r: torch.Tensor) -> torch.Tensor:
        if self._weighting is None:
            return torch.ones_like(r)
        return self._weighting(r)


    def _get_const(self, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """
        Return cached constant tensors on the requested device/dtype.
        """
        key = (str(device), str(dtype))
        cached = self._const_cache.get(key)
        if cached is not None:
            return cached

        out: Dict[str, torch.Tensor] = {}
        out["alphas"] = self._alphas.to(device=device, dtype=dtype)
        out["betas"]  = self._betas.to(device=device, dtype=dtype)
        out["eta"]    = torch.tensor(self._eta, device=device, dtype=dtype)
        # Polynomial backend constants (if initialized)
        if hasattr(self, "_rx"):
            out["rx"] = self._rx.to(device=device, dtype=dtype)
        if hasattr(self, "_ws"):
            out["ws"] = self._ws.to(device=device, dtype=dtype)
        if hasattr(self, "_B"):
            out["B"]  = self._B.to(device=device, dtype=dtype)

        self._const_cache[key] = out
        return out

    def _build_feature_slices(self) -> None:
        """
        Precompute descriptor slice ranges in DScribe ordering:
          for j in species:
            for jd in range(j, jd_limit):
              for l in 0..l_max:
                - if j==jd: take upper-triangle (k<=kd) of (n_max x n_max)
                - else: take full (n_max x n_max)
        Store as list: (j, jd, l, is_diag, start, end)
        """
        if self.average == "cc":
            # CC projection uses a different output layout; no power-spectrum slices needed.
            self._feat_slices = None
            return

        slices: List[Tuple[int,int,int,bool,int,int]] = []
        off = 0
        n = self._n_max
        n_pairs_diag = int(n * (n + 1) // 2)
        for j in range(self.n_species):
            jd_limit = self.n_species if self.crossover else (j + 1)
            for jd in range(j, jd_limit):
                is_diag = (j == jd)
                for l in range(self._l_max + 1):
                    if is_diag:
                        start, end = off, off + n_pairs_diag
                        off = end
                    else:
                        start, end = off, off + n * n
                        off = end
                    slices.append((j, jd, l, is_diag, start, end))
        self._feat_slices = slices
        if off != self.n_features:
            raise RuntimeError(f"Feature slice size mismatch: built={off}, expected={self.n_features}")


    def _coefficients_gto(
        self,
        center_index: torch.Tensor,
        neigh_species: torch.Tensor,
        rvec: torch.Tensor,
        r: torch.Tensor,
        n_centers: int,
        prof: Optional[Profiler] = None,
    ) -> List[torch.Tensor]:
        """
        Returns list over l: c_l (C, S, n_max, 2l+1)
        """
        device = r.device
        dtype = r.dtype

        # sphericart.torch.SphericalHarmonics uses the direction of xyz; xyz==0 is undefined.

        # Build unit vectors and guard any tiny radii (should mostly be removed in _build_neighbor_list).

        eps = 1e-8 if dtype == torch.float32 else 1e-12

        r_safe = torch.clamp(r, min=eps)

        unit = rvec / r_safe[:, None]

        tiny = r < eps

        if torch.any(tiny):

            unit = unit.clone()

            unit[tiny] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

        if prof is not None:
            with prof.section("ylm"):
                Y_all = self._Y.compute(unit)  # (E, (l_max+1)^2)
        else:
            Y_all = self._Y.compute(unit)  # (E, (l_max+1)^2)
        
        if prof is not None:
            with prof.section("weights"):
                w = self._weights(r)
        else:
            w = self._weights(r)

        const = self._get_const(device, dtype)
        alphas = const["alphas"]
        betas = const["betas"]
        eta = const["eta"]

        idx0 = center_index * self.n_species + neigh_species
        n_rows = n_centers * self.n_species

        out = []
        for l in range(self._l_max + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            Y = Y_all[:, start:end]  # (E, 2l+1)

            alpha_l = alphas[l]      # (n_max,)
            beta_l = betas[l]        # (n_max,n_max)

            if prof is not None:
                with prof.section("radial"):
                    p = alpha_l[None, :] + eta  # (1,n_max)
                    exp_term = torch.exp(-(alpha_l[None, :] * eta / p) * (r[:, None] ** 2))  # (E,n_max)
                    r_l = (r ** l)[:, None]  # (E,1)
                    pref = (math.pi ** 1.5) * (eta / p) ** l * (p ** (-1.5))                 # (1,n_max)
                    prim = w[:, None] * (pref * r_l * exp_term)                              # (E,n_max)
            else:
                p = alpha_l[None, :] + eta  # (1,n_max)
                exp_term = torch.exp(-(alpha_l[None, :] * eta / p) * (r[:, None] ** 2))  # (E,n_max)
                r_l = (r ** l)[:, None]  # (E,1)
                pref = (math.pi ** 1.5) * (eta / p) ** l * (p ** (-1.5))                 # (1,n_max)
                prim = w[:, None] * (pref * r_l * exp_term)                              # (E,n_max)

            if prof is not None:
                with prof.section("scatter"):
                    contrib = prim[:, :, None] * Y[:, None, :]                                 # (E,n_max,2l+1)
                    contrib_flat = contrib.reshape(contrib.shape[0], -1)                       # (E, n_max*(2l+1))
                    acc_flat = torch.zeros((n_rows, contrib_flat.shape[1]), device=device, dtype=dtype)
                    acc_flat.index_add_(0, idx0, contrib_flat)
                    acc = acc_flat.view(n_centers, self.n_species, self._n_max, 2 * l + 1)
            else:
                contrib = prim[:, :, None] * Y[:, None, :]                                 # (E,n_max,2l+1)
                contrib_flat = contrib.reshape(contrib.shape[0], -1)                       # (E, n_max*(2l+1))
                acc_flat = torch.zeros((n_rows, contrib_flat.shape[1]), device=device, dtype=dtype)
                acc_flat.index_add_(0, idx0, contrib_flat)
                acc = acc_flat.view(n_centers, self.n_species, self._n_max, 2 * l + 1)

            if prof is not None:
                with prof.section("beta"):
                    c = torch.einsum("ab,csbm->csam", beta_l, acc)
            else:
                c = torch.einsum("ab,csbm->csam", beta_l, acc)
            out.append(c)

        return out

    def _coefficients_polynomial(
        self,
        center_index: torch.Tensor,
        neigh_species: torch.Tensor,
        rvec: torch.Tensor,
        r: torch.Tensor,
        n_centers: int,
        prof: Optional[Profiler] = None,
    ) -> List[torch.Tensor]:
        device = r.device
        dtype = r.dtype
        # sphericart.torch.SphericalHarmonics uses the direction of xyz; xyz==0 is undefined.
        # Build unit vectors and guard any tiny radii (should mostly be removed in _build_neighbor_list).
        eps = 1e-8 if dtype == torch.float32 else 1e-12
        r_safe = torch.clamp(r, min=eps)
        unit = rvec / r_safe[:, None]
        tiny = r < eps
        if torch.any(tiny):
            unit = unit.clone()
            unit[tiny] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        Y_all = self._Y.compute(unit)
        wj = self._weights(r)

        rx = self._rx.to(device=device, dtype=dtype)        # (Q,)
        wr = self._wr.to(device=device, dtype=dtype)        # (Q,)
        gss = self._gss.to(device=device, dtype=dtype)      # (n_max,Q)
        eta = torch.tensor(self._eta, device=device, dtype=dtype)

        rx2 = rx * rx
        rj = r
        t = 2 * eta * (rx[None, :] * rj[:, None])           # (E,Q)
        ie = modified_spherical_bessel_ie(self._l_max, t)   # (l_max+1,E,Q)

        gauss = torch.exp(-eta * (rx[None, :] - rj[:, None]) ** 2)   # (E,Q)
        common = (4.0 * math.pi) * (rx2[None, :] * gauss) * wr[None, :]  # (E,Q)

        idx0 = center_index * self.n_species + neigh_species
        n_rows = n_centers * self.n_species

        out = []
        for l in range(self._l_max + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            Y = Y_all[:, start:end]  # (E,2l+1)

            F = common * ie[l]       # (E,Q)
            I = F @ gss.transpose(0, 1)   # (E,n_max)

            prim = wj[:, None] * I
            contrib = prim[:, :, None] * Y[:, None, :]
            contrib_flat = contrib.reshape(contrib.shape[0], -1)
            acc_flat = torch.zeros((n_rows, contrib_flat.shape[1]), device=device, dtype=dtype)
            acc_flat.index_add_(0, idx0, contrib_flat)
            acc = acc_flat.view(n_centers, self.n_species, self._n_max, 2 * l + 1)
            out.append(acc)

        return out

    # ---- output assembly ----

    def _prefactor_l(self, l: int, device, dtype) -> torch.Tensor:
        # standard SOAP prefactor: sqrt(8*pi^2 / (2l+1))
        return torch.tensor(math.sqrt(8.0 * math.pi * math.pi / (2 * l + 1)), device=device, dtype=dtype)

    def _projection_cc(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        C = coeffs[0].shape[0]
        L = (self._l_max + 1) ** 2
        full = torch.zeros((C, self.n_species, self._n_max, L), device=coeffs[0].device, dtype=coeffs[0].dtype)
        for l, c_l in enumerate(coeffs):
            start = l * l
            end = (l + 1) * (l + 1)
            full[:, :, :, start:end] = c_l 
        return full.reshape(C, -1)

    def _power_spectrum(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        """
        Optimized power spectrum:
        - For each l, compute all species-pair correlations with a single einsum
        - Fill the descriptor using precomputed feature slices (DScribe ordering)

        Ordering matches DScribe loops:
          for j in species:
            for jd in range(j, jd_limit):
              for l in 0..l_max:
                take triangle/full block
        """
        device = coeffs[0].device
        dtype = coeffs[0].dtype

        if self.average == "inner":
            coeffs_use = [c.mean(dim=0, keepdim=True) for c in coeffs]
        else:
            coeffs_use = coeffs

        C_use = int(coeffs_use[0].shape[0])
        desc = torch.empty((C_use, self.n_features), device=device, dtype=dtype)

        triu_i = self._triu[0]
        triu_j = self._triu[1]

        if self._feat_slices is None:
            self._build_feature_slices()

        # compute per-l correlations; fill according to slice table
        for l, c_l in enumerate(coeffs_use):
            pref = self._prefactor_l(l, device, dtype)
            # P[c, j, jd, n, nd] = sum_m c[c,j,n,m] * c[c,jd,nd,m]
            P_all = torch.einsum("cjnm,ckdm->cjknd", c_l, c_l)
            P_all = pref * P_all

            # fill all segments corresponding to this l
            for (j, jd, l2, is_diag, start, end) in self._feat_slices:
                if l2 != l:
                    continue
                if is_diag:
                    Pjj = P_all[:, j, j, :, :]
                    desc[:, start:end] = Pjj[:, triu_i, triu_j]
                else:
                    desc[:, start:end] = P_all[:, j, jd, :, :].reshape(C_use, -1)

        if self.average == "outer":
            desc = desc.mean(dim=0)
        elif self.average == "inner":
            desc = desc.squeeze(0)

        return desc
    def create(
        self,
        system: Union[Any, List[Any]],
        centers: Optional[Union[Sequence[Any], List[Optional[Sequence[Any]]]]] = None,
        n_jobs: int = 1,
        only_physical_cores: bool = False,
        verbose: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        DScribe-like create:
          - system can be a single structure or a list of structures
          - centers can be None, or list (for one system), or list-of-lists (for multiple systems)
        Returns:
          - one system: tensor
          - many systems: list[tensor] (for variable sizes) or stacked tensor when shapes match
        """
        if isinstance(system, list):
            systems = system
            if centers is None:
                centers_list = [None] * len(systems)
            else:
                if not isinstance(centers, list):
                    raise ValueError("For multiple systems, centers must be a list with one entry per system.")
                centers_list = centers
                if len(centers_list) != len(systems):
                    raise ValueError("centers list must match number of systems.")
            outputs = []
            for sys_i, cen_i in zip(systems, centers_list):
                outputs.append(self.create(sys_i, cen_i))
            # Try stack if same shape
            try:
                return torch.stack(outputs, dim=0)
            except Exception:
                return outputs

        # single system
        positions, Z, cell, pbc = system_to_tensors(system, self.device, self.dtype)
        # validate species
        bad = set(torch.unique(Z).tolist()) - self._atomic_number_set
        if bad:
            raise ValueError(f"System contains atomic numbers not in species list: {sorted(bad)}")

        centers_xyz, center_indices = self.prepare_centers(positions, centers)

        center_index, neigh_species, rvec, r = self._build_neighbor_list(
            positions, Z, centers_xyz, cell, pbc
        )

        n_centers = int(centers_xyz.shape[0])

        if self._rbf == "gto":
            coeffs = self._coefficients_gto(center_index, neigh_species, rvec, r, n_centers)
        else:
            coeffs = self._coefficients_polynomial(center_index, neigh_species, rvec, r, n_centers)


        # Add analytic self-contribution for centers that coincide with an atom position.
        # This is required because we dropped r~0 edges in _build_neighbor_list (to avoid NaNs in sphericart).
        self._add_self_terms(coeffs, center_indices, Z)
        if self.average == "cc":
            out = self._projection_cc(coeffs)  # (C, n_cc)
        else:
            out = self._power_spectrum(coeffs) # (C,feat) or (feat,)
        return out.to_sparse_coo() if self.sparse else out

    # ---- Public: derivatives (DScribe-like numerical) ----


    @torch.no_grad()
    def create_with_profile(
        self,
        system: Any,
        centers: Optional[Sequence[Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        """
        returns (descriptor, profile_times).

        profile_times keys (may include):
          - to_tensors, centers, neighbor, ylm, weights, radial, scatter, beta, self_terms, power_spectrum, total
        """
        prof = Profiler(self.device)

        with prof.section("total"):
            with prof.section("to_tensors"):
                positions, Z, cell, pbc = system_to_tensors(system, self.device, self.dtype)

            bad = set(torch.unique(Z).tolist()) - self._atomic_number_set
            if bad:
                raise ValueError(f"System contains atomic numbers not in species list: {sorted(bad)}")

            with prof.section("centers"):
                centers_xyz, center_indices = self.prepare_centers(positions, centers)

            with prof.section("neighbor"):
                center_index, neigh_species, rvec, r = self._build_neighbor_list(
                    positions, Z, centers_xyz, cell, pbc
                )

            n_centers = int(centers_xyz.shape[0])

            if self._rbf == "gto":
                coeffs = self._coefficients_gto(center_index, neigh_species, rvec, r, n_centers, prof=prof)
            else:
                coeffs = self._coefficients_polynomial(center_index, neigh_species, rvec, r, n_centers, prof=prof)

            with prof.section("self_terms"):
                self._add_self_terms(coeffs, center_indices, Z)

            with prof.section("power_spectrum"):
                if self.average == "cc":
                    out = self._projection_cc(coeffs)
                else:
                    out = self._power_spectrum(coeffs)

        extra = {
            "n_atoms": int(positions.shape[0]),
            "n_centers": int(n_centers),
            "E_edges": int(center_index.numel()),
            "neighbors_per_center": (float(center_index.numel()) / float(n_centers)) if n_centers > 0 else 0.0,
            "nl_backend": getattr(self, "_last_nl_backend", "unknown"),
        }
        return (out.to_sparse_coo() if self.sparse else out), prof.times, extra


    def derivatives_numerical(
        self,
        system: Any,
        centers: Optional[Sequence[Any]] = None,
        include: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        method: str = "auto",
        return_descriptor: bool = True,
        attach: bool = False,
        n_jobs: int = 1,
        only_physical_cores: bool = False,
        verbose: bool = False,
        d: float = 1e-3,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Numerical derivatives using central differences.

        DScribe DescriptorLocal convention:
          derivatives shape: (n_centers, n_atoms_included, 3, n_features)

        include/exclude:
          - include: list of atom indices to differentiate w.r.t.
          - exclude: indices to omit
          - if both None: all atoms

        attach:
          - If centers are given as atomic indices and attach=True, the center position moves with that atom during
            finite-difference displacements (DScribe behavior).
        """
        if method not in ("auto", "numerical"):
            raise ValueError("method must be 'auto' or 'numerical'.")
        # auto->numerical in DScribe
        positions, Z, cell, pbc = system_to_tensors(system, self.device, self.dtype)
        n_atoms = int(positions.shape[0])

        # build included set
        if include is None:
            idx = list(range(n_atoms))
        else:
            idx = [int(i) for i in include]
        if exclude is not None:
            ex = set(int(i) for i in exclude)
            idx = [i for i in idx if i not in ex]
        idx_t = torch.tensor(idx, device=self.device, dtype=torch.long)

        # Prepare centers
        centers_xyz, centers_idx = self.prepare_centers(positions, centers)
        n_centers = int(centers_xyz.shape[0])

        # descriptor baseline
        desc0 = self.create(system, centers) if return_descriptor else None

        # output tensor
        n_feat = self.get_number_of_features() if self.average != "cc" else self.get_number_of_features()
        deriv = torch.zeros((n_centers, len(idx), 3, n_feat), device=self.device, dtype=self.dtype)

        # Central difference for each atom and component
        # This is expensive; use only when you truly need derivatives.
        for ai, atom_index in enumerate(idx):
            for comp in range(3):
                shift = torch.zeros((n_atoms, 3), device=self.device, dtype=self.dtype)
                shift[atom_index, comp] = d

                # +d
                pos_p = positions + shift
                sys_p = {"positions": pos_p, "atomic_numbers": Z, "cell": cell, "pbc": pbc} if self.periodic else {"positions": pos_p, "atomic_numbers": Z}
                # centers handling with attach
                if attach and centers is not None:
                    # If a center is attached to this atom (center given as index), move that center too
                    centers_p = []
                    for c in centers:
                        if isinstance(c, (int,)) and int(c) == atom_index:
                            centers_p.append(pos_p[atom_index])
                        else:
                            centers_p.append(c)
                else:
                    centers_p = centers
                f_p = self.create(sys_p, centers_p)

                # -d
                pos_m = positions - shift
                sys_m = {"positions": pos_m, "atomic_numbers": Z, "cell": cell, "pbc": pbc} if self.periodic else {"positions": pos_m, "atomic_numbers": Z}
                if attach and centers is not None:
                    centers_m = []
                    for c in centers:
                        if isinstance(c, (int,)) and int(c) == atom_index:
                            centers_m.append(pos_m[atom_index])
                        else:
                            centers_m.append(c)
                else:
                    centers_m = centers
                f_m = self.create(sys_m, centers_m)

                # Derivative: (f_p - f_m) / (2d)
                # Shapes:
                #   average='off'/'cc': (n_centers, n_feat)
                #   average='inner'/'outer': (n_feat,)  -> DScribe local derivatives not defined for global averaging;
                #   but DScribe still allows average modes; derivative then treated as 1 center? Here we broadcast.
                if f_p.dim() == 1:
                    # global: treat as one "center"
                    df = (f_p - f_m) / (2.0 * d)
                    deriv[:, ai, comp, :] = df[None, :].expand(n_centers, -1)
                else:
                    df = (f_p - f_m) / (2.0 * d)
                    deriv[:, ai, comp, :] = df

        return (deriv, desc0) if return_descriptor else deriv

    def derivatives(
        self,
        system: Any,
        centers: Optional[Sequence[Any]] = None,
        include: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        method: str = "auto",
        return_descriptor: bool = True,
        attach: bool = False,
        n_jobs: int = 1,
        only_physical_cores: bool = False,
        verbose: bool = False,
    ):
        # DScribe auto->numerical for this version
        return self.derivatives_numerical(
            system=system,
            centers=centers,
            include=include,
            exclude=exclude,
            method=method,
            return_descriptor=return_descriptor,
            attach=attach,
            n_jobs=n_jobs,
            only_physical_cores=only_physical_cores,
            verbose=verbose,
        )
