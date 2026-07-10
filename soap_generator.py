
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

try:
    import os as _os

    # Triton's bundled ptxas may not know very new GPU targets (e.g. the GB10's
    # sm_121a); prefer the system CUDA toolkit ptxas when one is installed.
    if "TRITON_PTXAS_PATH" not in _os.environ:
        for _p in ("/usr/local/cuda/bin/ptxas", "/usr/local/cuda-13.0/bin/ptxas"):
            if _os.path.exists(_p):
                _os.environ["TRITON_PTXAS_PATH"] = _p
                break

    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _cc_jacobian_dense_kernel(
        rvec_ptr,      # (E,3) f64
        B_ptr,         # (E,NMAX,LP1) f64 radial values
        Bhat_ptr,      # (E,NMAX,LP1) f64 radial -2*gamma values
        Yval_ptr,      # (E,LSQ) f64 solid harmonics
        Ygrad_ptr,     # (E,3,LSQ) f64 solid-harmonic Cartesian gradients
        emap_ptr,      # (C*n_inc,) i64 edge id for each (center, atom) pair, -1 = no edge
        sp_ptr,        # (E,) i64 neighbor species
        lm_src_ptr,    # (LSQ,) i32 source lm for each output lm (DScribe cc order)
        l_of_lm_ptr,   # (LSQ,) i32 degree l of each output lm
        scale_ptr,     # (LSQ,) f64 DScribe cc scale per output lm
        out_ptr,       # (C,n_inc,3,n_feat) f32, uninitialized (fully written here)
        n_feat,
        NMAX: tl.constexpr,
        LP1: tl.constexpr,
        LSQ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One (center, atom) pair per pid0. Pairs map to at most one edge
        # (single-image, non-periodic radius search), so every output slot is
        # written by exactly one program: plain stores, no atomics, and no
        # separate zero-initialization pass. For pairs beyond the cutoff (or
        # in the wrong species block) zero is stored; otherwise
        #   d c_{s n lm}/d x_{a,d} =
        #     (rvec_d * Bhat_{n,l} * Y_{lm} + B_{n,l} * gradY_{d,lm}) * scale_lm
        # computed in float64 and stored as float32.
        ca = tl.program_id(0).to(tl.int64)
        e = tl.load(emap_ptr + ca)
        active = e >= 0
        e0 = tl.maximum(e, 0)
        s = tl.load(sp_ptr + e0, mask=active, other=0)

        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < 3 * n_feat
        d = offs // n_feat
        rem = offs % n_feat
        sblk = rem // (NMAX * LSQ)
        rem2 = rem % (NMAX * LSQ)
        n = rem2 // LSQ
        lm = rem2 % LSQ

        msel = inb & active & (sblk == s)

        lms = tl.load(lm_src_ptr + lm, mask=inb, other=0)
        l = tl.load(l_of_lm_ptr + lm, mask=inb, other=0)
        sc = tl.load(scale_ptr + lm, mask=inb, other=0.0)

        rv = tl.load(rvec_ptr + e0 * 3 + d, mask=msel, other=0.0)
        Bv = tl.load(B_ptr + (e0 * NMAX + n) * LP1 + l, mask=msel, other=0.0)
        Bh = tl.load(Bhat_ptr + (e0 * NMAX + n) * LP1 + l, mask=msel, other=0.0)
        Yv = tl.load(Yval_ptr + e0 * LSQ + lms, mask=msel, other=0.0)
        Yg = tl.load(Ygrad_ptr + (e0 * 3 + d) * LSQ + lms, mask=msel, other=0.0)

        val = tl.where(msel, (rv * Bh * Yv + Bv * Yg) * sc, 0.0)
        tl.store(out_ptr + ca * 3 * n_feat + offs, val.to(tl.float32), mask=inb)

    @triton.jit
    def _ps_uv_kernel(
        Yval_ptr,      # (E,LSQ) f64 solid harmonics
        Ygrad_ptr,     # (E,3,LSQ) f64 solid-harmonic Cartesian gradients
        C_ptr,         # (C,S*NMAX,LSQ) f64 per-center coefficients c_{kq,lm}
        cidx_ptr,      # (E,) i64 center index of each edge
        U_ptr,         # (E,S*NMAX,LP1) f64 out: U[e,kq,l]   = sum_m Y_lm c_{kq,lm}
        V_ptr,         # (E,3,S*NMAX,LP1) f64 out: V[e,d,kq,l] = sum_m gradY_{d,lm} c_{kq,lm}
        NMAX: tl.constexpr,
        LP1: tl.constexpr,
        LSQ: tl.constexpr,
        S: tl.constexpr,
        M: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One edge per program: contract the edge's solid harmonics (values and
        # gradients) with its center's coefficients over m for every (kq, l).
        e = tl.program_id(0).to(tl.int64)
        c = tl.load(cidx_ptr + e)
        t = tl.arange(0, BLOCK)
        inb = t < S * NMAX * LP1
        kq = t // LP1
        l = t % LP1
        m = tl.arange(0, M)
        mm = inb[:, None] & (m[None, :] < (2 * l + 1)[:, None])
        lm = (l * l)[:, None] + m[None, :]
        Cv = tl.load(C_ptr + (c * S * NMAX + kq)[:, None] * LSQ + lm, mask=mm, other=0.0)
        Yv = tl.load(Yval_ptr + e * LSQ + lm, mask=mm, other=0.0)
        tl.store(U_ptr + e * (S * NMAX * LP1) + t, tl.sum(Cv * Yv, axis=1), mask=inb)
        for d in tl.static_range(3):
            Yg = tl.load(Ygrad_ptr + (e * 3 + d) * LSQ + lm, mask=mm, other=0.0)
            tl.store(V_ptr + (e * 3 + d) * (S * NMAX * LP1) + t, tl.sum(Cv * Yg, axis=1), mask=inb)

    @triton.jit
    def _ps_jacobian_dense_kernel(
        rvec_ptr,      # (E,3) f64
        B_ptr,         # (E,NMAX,LP1) f64 radial values
        Bhat_ptr,      # (E,NMAX,LP1) f64 radial -2*gamma values
        U_ptr,         # (E,S*NMAX,LP1) f64  U[e,kq,l]   = sum_m Y_lm(e) c_{kq,lm}
        V_ptr,         # (E,3,S*NMAX,LP1) f64 V[e,d,kq,l] = sum_m gradY_{d,lm}(e) c_{kq,lm}
        emap_ptr,      # (C*n_inc,) i64 edge id for each (center, atom) pair, -1 = no edge
        sp_ptr,        # (E,) i64 neighbor species
        fj_ptr,        # (n_feat,) i32 first species j of the feature
        fjd_ptr,       # (n_feat,) i32 second species j' of the feature
        fn_ptr,        # (n_feat,) i32 first radial index n of the feature
        fnd_ptr,       # (n_feat,) i32 second radial index n' of the feature
        fl_ptr,        # (n_feat,) i32 degree l of the feature
        fpref_ptr,     # (n_feat,) f64 sqrt(8 pi^2 / (2l+1))
        out_ptr,       # (C,n_inc,3,n_feat) f32, uninitialized (fully written here)
        n_feat,
        n_inc,
        n_blk,         # number of BLOCK-sized feature blocks per (center, atom) pair
        NMAX: tl.constexpr,
        LP1: tl.constexpr,
        S: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Power-spectrum Jacobian, one (center, atom) pair per pid0. With at
        # most one edge per pair (single-image, non-periodic radius search) the
        # scattered coefficient gradient collapses to the edge gradient
        #   g[d,n,lm] = rvec_d * Bhat_{n,l} * Y_lm + B_{n,l} * gradY_{d,lm}
        # in the species block of the neighbor only, so the product rule
        #   d p_{j n, j' n', l} / d x_{a,d} = pref_l * sum_m (
        #         delta_{j,s}  g[d,n,lm]  * c_{j' n' lm}
        #       + delta_{j',s} g[d,n',lm] * c_{j n lm} )
        # collapses, after distributing g over the precomputed per-edge
        # contractions U = sum_m Y c and V = sum_m gradY c, to two fused
        # multiply-adds per term:
        #   delta_{j,s} * (rvec_d Bhat_n U[j'n'] + B_n V[d,j'n'])  + (n <-> n')
        # -- no per-m loop in the kernel at all. Everything is float64 and
        # stored once as float32: no edge-gradient tensor, no scatter buffer,
        # no per-l einsum intermediates, no zero-init memset.
        #
        # 1D pair-major grid: the n_blk feature blocks of one pair are
        # launch-adjacent, so the pair's B/Bhat/U/V rows are read from cache
        # instead of n_blk times from DRAM.
        pid = tl.program_id(0).to(tl.int64)
        ca = pid // n_blk
        offs = (pid % n_blk) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < 3 * n_feat
        e = tl.load(emap_ptr + ca)
        if e < 0:
            tl.store(out_ptr + ca * 3 * n_feat + offs,
                     tl.zeros((BLOCK,), dtype=tl.float32), mask=inb)
            return
        s = tl.load(sp_ptr + e)

        d = offs // n_feat
        f = offs % n_feat
        j = tl.load(fj_ptr + f, mask=inb, other=0)
        jd = tl.load(fjd_ptr + f, mask=inb, other=0)
        n = tl.load(fn_ptr + f, mask=inb, other=0)
        nd = tl.load(fnd_ptr + f, mask=inb, other=0)
        l = tl.load(fl_ptr + f, mask=inb, other=0)
        pref = tl.load(fpref_ptr + f, mask=inb, other=0.0)

        t1 = inb & (j == s)    # dc lives in species block j
        t2 = inb & (jd == s)   # dc lives in species block j'

        rv = tl.load(rvec_ptr + e * 3 + d, mask=inb, other=0.0)
        idx1 = j * NMAX + n
        idx2 = jd * NMAX + nd
        # masked loads return 0, which zeroes the whole term: no tl.where
        Bv_n = tl.load(B_ptr + (e * NMAX + n) * LP1 + l, mask=t1, other=0.0)
        Bh_n = tl.load(Bhat_ptr + (e * NMAX + n) * LP1 + l, mask=t1, other=0.0)
        U2 = tl.load(U_ptr + (e * S * NMAX + idx2) * LP1 + l, mask=t1, other=0.0)
        V2 = tl.load(V_ptr + ((e * 3 + d) * S * NMAX + idx2) * LP1 + l, mask=t1, other=0.0)
        Bv_nd = tl.load(B_ptr + (e * NMAX + nd) * LP1 + l, mask=t2, other=0.0)
        Bh_nd = tl.load(Bhat_ptr + (e * NMAX + nd) * LP1 + l, mask=t2, other=0.0)
        U1 = tl.load(U_ptr + (e * S * NMAX + idx1) * LP1 + l, mask=t2, other=0.0)
        V1 = tl.load(V_ptr + ((e * 3 + d) * S * NMAX + idx1) * LP1 + l, mask=t2, other=0.0)

        val = pref * ((rv * Bh_n * U2 + Bv_n * V2) + (rv * Bh_nd * U1 + Bv_nd * V1))
        tl.store(out_ptr + ca * 3 * n_feat + offs, val.to(tl.float32), mask=inb)


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
    # i_1(t) e^{-t} = e^{-t} (cosh t / t - sinh t / t^2)
    eps = 1e-12
    tt = torch.clamp(t, min=eps)
    e2 = torch.exp(-2 * tt)
    return (1 + e2) / (2 * tt) - (1 - e2) / (2 * tt * tt)


def modified_spherical_bessel_ie(l_max: int, t: torch.Tensor) -> torch.Tensor:
    # Upward recurrence i_{l+1} = i_{l-1} - (2l+1)/t i_l, out-of-place so the
    # autograd derivative path can differentiate through it. i_l(t) is positive
    # and strictly decreasing in l, so clamping each level into [0, i_{l-1}]
    # suppresses the small-t cancellation noise of the upward recurrence
    # without touching well-conditioned values.
    eps = 1e-12
    tt = torch.clamp(t, min=eps)
    outs = [_i0e(tt)]
    if l_max == 0:
        return torch.stack(outs, dim=0)
    outs.append(torch.clamp(_i1e(tt), min=0.0))
    for l in range(1, l_max):
        nxt = outs[l - 1] - (2 * l + 1) / tt * outs[l]
        nxt = torch.minimum(torch.clamp(nxt, min=0.0), outs[l])
        outs.append(nxt)
    return torch.stack(outs, dim=0)


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
        dtype: str = "float32",
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

        # Warm up CUDA kernels (JIT compilation, cuBLAS handles, sphericart
        # setup) on a tiny dummy system so the first real create()/derivatives()
        # call runs at steady-state speed.
        self._warmup()

    def _warmup(self) -> None:
        if not (str(self.device).startswith("cuda") and torch.cuda.is_available()):
            return
        try:
            n = self.n_species
            pos = torch.zeros((n, 3), device=self.device, dtype=self.dtype)
            pos[:, 2] = torch.arange(n, device=self.device, dtype=self.dtype) * 0.95
            dummy: Dict[str, Any] = {"positions": pos, "atomic_numbers": self._atomic_numbers}
            if self.periodic:
                box = 2.0 * self._cutoff + 1.0
                dummy["cell"] = torch.eye(3, device=self.device, dtype=self.dtype) * box
                dummy["pbc"] = torch.tensor([True, True, True], device=self.device, dtype=torch.bool)
            self.create(dummy)
            if (
                self._rbf in ("gto", "polynomial")
                and not self.periodic
                and self._weighting is None
                and self.average in ("off", "outer", "cc", "inner")
            ):
                self.derivatives_analytical_ps(dummy, return_descriptor=False)
            # Pre-warm the CUDA caching allocator: allocate and free one large
            # block so the first real call's big output tensors are served by
            # splitting this cached block instead of a fresh cudaMalloc
            # (~30 ms/GB on GB10). 2560 MB covers the (C,A,3,n_feat) float32
            # power-spectrum Jacobian plus its U/V workspaces for the 300-atom
            # test case. Override with SOAP_CUDA_POOL_MB (0 disables).
            import os
            pool_mb = int(os.environ.get("SOAP_CUDA_POOL_MB", "2560"))
            if pool_mb > 0:
                buf = torch.empty(pool_mb * 1024 * 1024, device=self.device, dtype=torch.uint8)
                del buf
            torch.cuda.synchronize()
        except Exception:
            # Warmup is best-effort; never fail construction because of it.
            pass


    # ---- sphericart ----

    def _init_sphericart(self):
        try:
            import sphericart.torch as sct
        except Exception as e:
            raise ImportError("sphericart.torch is required. Install/build sphericart with torch bindings.") from e
        self._sct = sct
        self._Y = sct.SphericalHarmonics(self._l_max)
        # Real *solid* harmonics r^l Y_lm and their exact Cartesian gradients
        # (used by the closed-form analytical derivative). By construction
        # SolidHarmonics(xyz) == |xyz|^l * SphericalHarmonics(xyz/|xyz|), so it is
        # consistent with the forward pass to machine precision.
        self._Ysolid = sct.SolidHarmonics(self._l_max)

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
        """
        Polynomial radial basis g_n(r) = sum_k beta_nk (r_cut - r)^{k+2},
        Loewdin-orthonormalized, evaluated on the 100-point Gauss-Legendre
        grid — a faithful replica of DScribe's SOAP.get_basis_poly.

        The overlap matrix S (entries ~ r_cut^{7+i+j}) is severely
        ill-conditioned, so the numerically dominated directions of S^{-1/2}
        depend on the algorithm used. DScribe computes
        scipy.linalg.sqrtm(numpy.linalg.inv(S)) in float64 on the CPU; to
        reproduce DScribe's descriptor we must use the exact same computation
        (the torch eigh-based _lowdin_invsqrt disagrees at the percent level
        for n_max >~ 5). Falls back to _lowdin_invsqrt if SciPy is missing.
        """
        device = self.device
        dtype = torch.float64
        n = self._n_max
        rc = self._r_cut

        import numpy as np
        S_np = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                S_np[i - 1, j - 1] = (2 * rc ** (7 + i + j)) / ((5 + i + j) * (6 + i + j) * (7 + i + j))
        try:
            from scipy.linalg import sqrtm
            betas_np = sqrtm(np.linalg.inv(S_np))
            if betas_np.dtype == np.complex128:
                raise ValueError(
                    "Could not calculate normalization factors for the radial "
                    "basis in the domain of real numbers. Lowering the number of "
                    "radial basis functions (n_max) or increasing the radial "
                    "cutoff (r_cut) is advised."
                )
            betas = torch.from_numpy(np.ascontiguousarray(betas_np)).to(device=device, dtype=dtype)
        except ImportError:
            betas = _lowdin_invsqrt(torch.from_numpy(S_np).to(device=device, dtype=dtype))

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

        # r->0 limit of the l=1 radial factor R_1(r) = c_n1(r)/(r Y_1m(r^)),
        # i.e. the coefficient of the *solid* harmonic r Y_1m in the self edge.
        # The dropped r~0 edge has zero l=1 *value* (Y_solid,1(0)=0) but a nonzero
        # l=1 *gradient* R_1(0)*grad Y_solid,1(0); the fixed-center analytical
        # derivative restores it via _add_self_gradient_terms.
        if self._l_max >= 1:
            if self._rbf == "gto":
                alpha1 = self._alphas[1].to(device=device, dtype=dtype)  # (n_max,)
                beta1 = self._betas[1].to(device=device, dtype=dtype)    # (n_max,n_max)
                p1 = alpha1 + eta
                # prim_k/r at r=0: pi^{3/2} (eta/p)^1 p^{-3/2}  (exp term -> 1)
                pref1 = (math.pi ** 1.5) * (eta / p1) * torch.pow(p1, -1.5)  # (n_max,)
                self._self_l1 = (w0 * (beta1 @ pref1)).to(device=device, dtype=dtype)
            else:
                # I_n1(r)/r at r=0: ie_1(2 eta rx r)/r -> 2 eta rx / 3 (with the
                # gaussian exp(-eta(rx-r)^2) -> exp(-eta rx^2), as in ie scaling)
                rx = self._rx.to(device=device, dtype=dtype)
                wr = self._wr.to(device=device, dtype=dtype)
                gss = self._gss.to(device=device, dtype=dtype)
                common1 = (4.0 * math.pi) * (rx * rx) * torch.exp(-eta * (rx ** 2)) * (2.0 * eta * rx / 3.0) * wr
                self._self_l1 = (w0 * (common1 @ gss.transpose(0, 1))).to(device=device, dtype=dtype)
        else:
            self._self_l1 = None

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

    def _add_self_gradient_terms(
        self,
        coeffs: List[torch.Tensor],
        positions: torch.Tensor,
        centers_xyz: torch.Tensor,
        center_indices: torch.Tensor,
        Z: torch.Tensor,
    ) -> None:
        """
        Restore the gradient of the r~0 self edge in an autograd forward pass.

        _build_neighbor_list drops the edge between a center and a coincident
        atom and _add_self_terms re-adds its value as a position-independent
        l=0 constant, so autograd sees zero gradient for it. Under the DScribe
        fixed-center convention (attach=False) the coincident atom is a real
        neighbor: as it moves off the fixed center, the l=1 coefficients change
        at first order (d/dx [R_l(r) Y_solid,lm(rvec)] at rvec=0 equals
        R_l(0) * grad Y_solid,lm(0), which is nonzero only for l=1; the l=0
        Gaussian term is even in rvec, so its gradient vanishes).

        This adds the value-zero, gradient-exact term
            (R_1(0) * Y_solid,1m(d)) - detach(same),  d = pos[atom] - center
        to the l=1 block. Y_solid (sphericart SolidHarmonics) is polynomial in
        d, hence exactly differentiable at d=0. When index centers are instead
        attached to their atom in the graph (attach=True), d carries zero net
        gradient and the term is a no-op, as it should be.
        """
        if self._l_max < 1 or self._self_l1 is None:
            return
        if coeffs is None or len(coeffs) < 2:
            return
        if not (positions.requires_grad or centers_xyz.requires_grad):
            return
        mask = center_indices >= 0
        if not torch.any(mask):
            return

        center_ids = torch.arange(center_indices.shape[0], device=center_indices.device, dtype=torch.long)[mask]
        atom_ids = center_indices[mask]
        sp = self._map_Z_to_species(Z[atom_ids])
        valid = sp >= 0
        if not torch.any(valid):
            return
        center_ids = center_ids[valid]
        atom_ids = atom_ids[valid]
        sp = sp[valid]

        d = positions[atom_ids] - centers_xyz[center_ids]          # (K,3), ~0
        Y1 = self._Ysolid.compute(d)[:, 1:4]                       # (K,3) = r*Y_1m, m=-1,0,1
        R10 = self._self_l1.to(device=d.device, dtype=d.dtype)     # (n_max,)
        term = R10[None, :, None] * Y1[:, None, :]                 # (K,n_max,3)
        term = term - term.detach()                                # value 0, gradient exact
        coeffs[1] = coeffs[1].index_put((center_ids, sp), term, accumulate=True)

    # ---- DScribe-like helper methods ----

    def get_number_of_features(self) -> int:
        if self.average == "cc":
            return self.n_species * self._n_max * (self._l_max + 1) ** 2
        return self._n_power_features()

    def _n_power_features(self) -> int:
        """Feature count of the power spectrum p_nn'l (independent of self.average)."""
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
        # Same threshold as the r~0 edge drop in _build_neighbor_list: an explicit
        # position that coincides with an atom must be treated like an index center,
        # otherwise that atom's (l=0) contribution to c_nlm is silently lost.
        eps_self = 1e-8 if positions.dtype == torch.float32 else 1e-12
        pos_d = positions.detach()
        for c in centers:
            if isinstance(c, (int,)):
                idx = int(c)
                centers_xyz.append(positions[idx])
                center_indices.append(idx)
            else:
                cc = _as_torch(c, device=positions.device, dtype=positions.dtype).reshape(3)
                centers_xyz.append(cc)
                idx = -1
                if pos_d.shape[0] > 0:
                    d2 = ((pos_d - cc.detach()[None, :]) ** 2).sum(dim=1)
                    j = int(torch.argmin(d2).item())
                    if float(d2[j].item()) <= eps_self * eps_self:
                        idx = j
                center_indices.append(idx)
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

        Note: this routine only produces (integer) edge indices, so it never needs
        gradients. We detach the inputs here so the in-place / out=-argument GEMM
        optimizations below are compatible with an autograd-enabled forward pass
        (the differentiable `rvec` is rebuilt from these indices by the caller).
        """
        centers = centers.detach()
        neigh_pos = neigh_pos.detach()
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
        # GTO basis constants (only when rbf='gto')
        if hasattr(self, "_alphas"):
            out["alphas"] = self._alphas.to(device=device, dtype=dtype)
            out["betas"]  = self._betas.to(device=device, dtype=dtype)
        out["eta"] = torch.tensor(self._eta, device=device, dtype=dtype)
        # Polynomial backend constants (only when rbf='polynomial')
        if hasattr(self, "_rx"):
            out["rx"] = self._rx.to(device=device, dtype=dtype)
            out["wr"] = self._wr.to(device=device, dtype=dtype)
            out["gss"] = self._gss.to(device=device, dtype=dtype)

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
        # Always built in the power-spectrum layout: even for average='cc' the
        # derivative routines evaluate p_nn'l (equations.pdf), which needs these slices.
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
        if off != self._n_power_features():
            raise RuntimeError(f"Feature slice size mismatch: built={off}, expected={self._n_power_features()}")

    def _ps_feat_tables(self, device: torch.device):
        """
        Per-feature lookup tables for the Triton power-spectrum Jacobian
        kernel: feature f -> (j, j', n, n', l, pref_l) in the exact DScribe
        ordering of _build_feature_slices (diagonal species blocks walk the
        row-major upper triangle of (n_max x n_max), matching triu_indices).
        Cached on the instance per device.
        """
        cache = getattr(self, "_ps_feat_cache", None)
        if cache is not None and cache[0] == str(device):
            return cache[1]
        if self._feat_slices is None:
            self._build_feature_slices()
        nmax = self._n_max
        fj: List[int] = []
        fjd: List[int] = []
        fn: List[int] = []
        fnd: List[int] = []
        fl: List[int] = []
        fpref: List[float] = []
        for (j, jd, l, is_diag, start, end) in self._feat_slices:
            pref = math.sqrt(8.0 * math.pi * math.pi / (2 * l + 1))
            if is_diag:
                pairs = [(a, b) for a in range(nmax) for b in range(a, nmax)]
            else:
                pairs = [(a, b) for a in range(nmax) for b in range(nmax)]
            for (a, b) in pairs:
                fj.append(j)
                fjd.append(jd)
                fn.append(a)
                fnd.append(b)
                fl.append(l)
                fpref.append(pref)
        if len(fj) != self._n_power_features():
            raise RuntimeError("Feature table size mismatch in _ps_feat_tables.")
        tables = (
            torch.tensor(fj, device=device, dtype=torch.int32),
            torch.tensor(fjd, device=device, dtype=torch.int32),
            torch.tensor(fn, device=device, dtype=torch.int32),
            torch.tensor(fnd, device=device, dtype=torch.int32),
            torch.tensor(fl, device=device, dtype=torch.int32),
            torch.tensor(fpref, device=device, dtype=torch.float64),
        )
        self._ps_feat_cache = (str(device), tables)
        return tables


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

        I, _ = self._poly_I_J(r, want_J=False)              # (E,n_max,l_max+1)

        idx0 = center_index * self.n_species + neigh_species
        n_rows = n_centers * self.n_species

        out = []
        for l in range(self._l_max + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            Y = Y_all[:, start:end]  # (E,2l+1)

            prim = wj[:, None] * I[:, :, l]
            contrib = prim[:, :, None] * Y[:, None, :]
            contrib_flat = contrib.reshape(contrib.shape[0], -1)
            acc_flat = torch.zeros((n_rows, contrib_flat.shape[1]), device=device, dtype=dtype)
            acc_flat.index_add_(0, idx0, contrib_flat)
            acc = acc_flat.view(n_centers, self.n_species, self._n_max, 2 * l + 1)
            out.append(acc)

        return out

    def _poly_I_J(self, r: torch.Tensor, want_J: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Radial density-projection integrals of the polynomial basis on the
        DScribe Gauss-Legendre grid, for all n and l at once
        (ie_l(t) = i_l(t) e^{-t}, t = 2 eta rx r):

            I[e,n,l] = 4 pi int rx^2 g_n(rx) e^{-eta (rx-r_e)^2} ie_l(t) drx
            J[e,n,l] = 4 pi int rx^3 g_n(rx) e^{-eta (rx-r_e)^2} ie_{l+1}(t) drx

        The Bessel recurrence runs directly on P_l = common * ie_l (the
        positive l-independent common factor commutes with the linear
        recurrence and with the [0, P_{l-1}] stability clamp), and each P_l
        slab is read by a single GEMM against the combined weights
        [gss | gss*rx], which keeps the f64 memory traffic minimal (GB10 f64
        ALU/exp is ~1/64 rate, so this path is bandwidth/launch bound).
        Fully differentiable — the autograd fallback path goes through here
        via _coefficients_polynomial.
        """
        device = r.device
        dtype = r.dtype
        nmax = self._n_max
        Lp1 = self._l_max + 1
        E = int(r.numel())
        rx = self._rx.to(device=device, dtype=dtype)        # (Q,)
        wr = self._wr.to(device=device, dtype=dtype)        # (Q,)
        gssT = self._gss.to(device=device, dtype=dtype).transpose(0, 1)  # (Q,n)
        eta = self._eta
        Q = int(rx.numel())

        if want_J:
            W = torch.cat([gssT, gssT * rx[:, None]], dim=1).contiguous()  # (Q,2n)
        else:
            W = gssT.contiguous()

        I = torch.empty((E, nmax, Lp1), device=device, dtype=dtype)
        J = torch.empty((E, nmax, Lp1), device=device, dtype=dtype) if want_J else None
        L_top = Lp1 if want_J else Lp1 - 1   # highest ie level needed

        rxw = rx * rx * wr                                   # (Q,)
        chunk = max(1, (1 << 23) // max(Q, 1))
        for e0 in range(0, E, chunk):
            e1 = min(e0 + chunk, E)
            re_ = r[e0:e1]
            tt = torch.clamp((2.0 * eta) * rx[None, :] * re_[:, None], min=1e-12)  # (e,Q)
            e2 = torch.exp(-2.0 * tt)
            oOt = 1.0 / tt
            common = (4.0 * math.pi) * rxw[None, :] * torch.exp(
                -eta * (rx[None, :] - re_[:, None]) ** 2
            )                                                                       # (e,Q)
            P_pp: Optional[torch.Tensor] = None
            P_p: Optional[torch.Tensor] = None
            for l in range(L_top + 1):
                if l == 0:
                    P = common * ((1.0 - e2) * (0.5 * oOt))                         # common*ie_0
                elif l == 1:
                    ie1 = torch.clamp(
                        (1.0 + e2) * (0.5 * oOt) - (1.0 - e2) * (0.5 * oOt * oOt), min=0.0
                    )
                    P = common * ie1
                else:
                    P = torch.minimum(
                        torch.clamp(P_pp - (2 * l - 1) * oOt * P_p, min=0.0), P_p
                    )
                out = P @ W                                                          # (e, n or 2n)
                if l < Lp1:
                    I[e0:e1, :, l] = out[:, :nmax]
                if want_J and l >= 1:
                    J[e0:e1, :, l - 1] = out[:, nmax:]
                P_pp, P_p = P_p, P

        return I, J

    def _poly_B_Bhat_r0(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact r->0 limits of the polynomial radial factors B_nl = I_nl(r)/r^l
        and Bhat_nl = (1/r) d/dr B_nl, from i_l(t) ~ t^l/(2l+1)!!:

            B0_nl    = 4 pi int rx^2 g_n e^{-eta rx^2} (2 eta rx)^l / (2l+1)!! drx
            Bhat0_nl = 2 eta * 4 pi int rx^2 g_n e^{-eta rx^2} (2 eta rx)^l
                       * [ 2 eta rx^2 / (2l+3)!! - 1 / (2l+1)!! ] drx

        Used for the r~0 self edges kept in the closed-form derivative edge
        list. Cached on the instance (float64).
        """
        cached = getattr(self, "_poly_r0_cache", None)
        if cached is not None:
            return cached
        device = self.device
        dtype = torch.float64
        rx = self._rx.to(device=device, dtype=dtype)
        wr = self._wr.to(device=device, dtype=dtype)
        gssT = self._gss.to(device=device, dtype=dtype).transpose(0, 1)  # (Q,n)
        eta = self._eta
        nmax = self._n_max
        Lp1 = self._l_max + 1
        common0 = (4.0 * math.pi) * rx * rx * wr * torch.exp(-eta * rx * rx)  # (Q,)
        B0 = torch.empty((nmax, Lp1), device=device, dtype=dtype)
        Bhat0 = torch.empty((nmax, Lp1), device=device, dtype=dtype)
        fl = torch.ones_like(rx)   # (2 eta rx)^l
        df = 1.0                   # (2l+1)!!
        for l in range(Lp1):
            df_next = df * (2 * l + 3)
            B0[:, l] = (common0 * fl / df) @ gssT
            Bhat0[:, l] = (2.0 * eta) * (
                (common0 * fl * (2.0 * eta) * rx * rx / df_next) @ gssT - B0[:, l]
            )
            fl = fl * (2.0 * eta) * rx
            df = df_next
        self._poly_r0_cache = (B0, Bhat0)
        return B0, Bhat0

    def _poly_B_Bhat(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Polynomial-basis radial factors of the solid-harmonic factorization
        used by the closed-form derivative (weighting=None):

            c_nlm(edge)      = B[e,n,l] * Y_solid,lm(rvec)
            d B / d rvec_d   = rvec_d * Bhat[e,n,l]

        With the density-projection integral (ie_l(t) = i_l(t) e^{-t}):

            I_nl(r) = 4 pi int rx^2 g_n(rx) e^{-eta (rx-r)^2} ie_l(2 eta rx r) drx
            J_nl(r) = 4 pi int rx^3 g_n(rx) e^{-eta (rx-r)^2} ie_{l+1}(2 eta rx r) drx

        the identity i_l'(t) = i_{l+1}(t) + (l/t) i_l(t) collapses the quotient
        rule to

            B_nl    = I_nl / r^l
            Bhat_nl = 2 eta (J_nl - r I_nl) / r^{l+1}.

        Both limits are finite at r=0; edges with r < 1e-4 (the kept r~0 self
        edges) are overridden with the exact limits from _poly_B_Bhat_r0.
        """
        device = r.device
        dtype = r.dtype
        Lp1 = self._l_max + 1
        eta = self._eta

        tiny = r < 1e-4
        # placeholder radius for the tiny edges (their rows are overwritten
        # below with the exact limits; this just keeps the quadrature finite)
        rs = torch.where(tiny, torch.ones_like(r), r)

        I, J = self._poly_I_J(rs, want_J=True)               # (E,n,Lp1) each

        # r^l per edge and l: cumprod of [1, r, r, ...] along l
        rl = torch.cumprod(
            torch.cat([torch.ones_like(rs)[:, None], rs[:, None].expand(-1, Lp1 - 1)], dim=1)
            if Lp1 > 1 else torch.ones_like(rs)[:, None],
            dim=1,
        )                                                     # (E,Lp1)
        B = I / rl[:, None, :]
        Bhat = (2.0 * eta) * (J - rs[:, None, None] * I) / (rl * rs[:, None])[:, None, :]

        if bool(tiny.any()):
            B0, Bhat0 = self._poly_B_Bhat_r0()
            B[tiny] = B0.to(device=device, dtype=dtype)
            Bhat[tiny] = Bhat0.to(device=device, dtype=dtype)
        return B, Bhat

    # ---- output assembly ----

    def _prefactor_l(self, l: int, device, dtype) -> torch.Tensor:
        # standard SOAP prefactor: sqrt(8*pi^2 / (2l+1))
        return torch.tensor(math.sqrt(8.0 * math.pi * math.pi / (2 * l + 1)), device=device, dtype=dtype)

    def _to_dscribe_cc(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map a tensor whose last axis is our (l,m) ordering onto DScribe's
        stored-coefficient (cnnd) convention for average='cc':

          - l=0 unchanged;
          - the l=1 block is stored in (z, x, y) order instead of the
            real-harmonic m=(-1,0,1) = (y, z, x) order;
          - l>=2 blocks carry an extra pi^{-3/2} (DScribe's power-spectrum
            prefactor includes the compensating pi^3 for l>=2 only).

        Differentiable, so it applies equally to values and Jacobians.
        """
        L = (self._l_max + 1) ** 2
        perm = torch.arange(L, device=t.device)
        if self._l_max >= 1:
            perm[1:4] = torch.tensor([2, 3, 1], device=t.device)
        scale = torch.ones(L, device=t.device, dtype=t.dtype)
        scale[4:] = math.pi ** -1.5
        return t.index_select(-1, perm) * scale

    def _projection_cc(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        C = coeffs[0].shape[0]
        L = (self._l_max + 1) ** 2
        full = torch.zeros((C, self.n_species, self._n_max, L), device=coeffs[0].device, dtype=coeffs[0].dtype)
        for l, c_l in enumerate(coeffs):
            start = l * l
            end = (l + 1) * (l + 1)
            full[:, :, :, start:end] = c_l
        return self._to_dscribe_cc(full).reshape(C, -1)

    def _power_spectrum(self, coeffs: List[torch.Tensor], average: Optional[str] = None) -> torch.Tensor:
        """
        Optimized power spectrum:
        - For each l, compute all species-pair correlations with a single einsum
        - Fill the descriptor using precomputed feature slices (DScribe ordering)

        `average` overrides self.average (used by the derivative routines to
        evaluate the outer-averaged p_nn'l even for a 'cc' descriptor).

        Ordering matches DScribe loops:
          for j in species:
            for jd in range(j, jd_limit):
              for l in 0..l_max:
                take triangle/full block
        """
        device = coeffs[0].device
        dtype = coeffs[0].dtype
        avg = self.average if average is None else average

        # Center-wise 'inner': row c is the descriptor inner-averaged over
        # center c alone (the coefficient average over a single center is the
        # identity), matching DScribe create(centers=[c], average='inner')
        # stacked over centers. Every mode therefore keeps one row per center.
        coeffs_use = coeffs

        C_use = int(coeffs_use[0].shape[0])
        desc = torch.empty((C_use, self._n_power_features()), device=device, dtype=dtype)

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

        # Center-wise output: one row per center, (C_use, n_feat).
        # 'outer' and 'inner' both keep the per-center spectra (row c is
        # exactly the spectrum DScribe returns for create(centers=[c]) with
        # the same average mode).
        return desc

    def generate(
        self,
        system: Union[Any, List[Any]],
        centers: Optional[Union[Sequence[Any], List[Optional[Sequence[Any]]]]] = None,
        n_jobs: int = 1,
        only_physical_cores: bool = False,
        verbose: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Center-wise SOAP generation: one descriptor row per center, shape
        (n_centers, n_soap_features). E.g. generate(system=molecule,
        centers=[0, 1]) returns a (2, n_features) tensor.

          - system can be a single structure or a list of structures
          - centers can be None (all atoms), or list (for one system), or
            list-of-lists (for multiple systems)
        Returns:
          - one system: (n_centers, n_features) tensor
          - many systems: stacked (n_systems, n_centers, n_features) tensor
            when shapes match, else list of (n_centers_i, n_features) tensors

        average='inner' is center-wise as well: row c is the inner-averaged
        descriptor of center c alone, so the output is (n_centers, n_features)
        just like the other modes.
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
                outputs.append(self.generate(sys_i, cen_i))
            # Try stack if same shape
            try:
                return torch.stack(outputs, dim=0)
            except Exception:
                return outputs

        # single system
        out = self._evaluate(system, centers)
        return out.to_sparse_coo() if self.sparse else out

    def create(
        self,
        system: Union[Any, List[Any]],
        centers: Optional[Union[Sequence[Any], List[Optional[Sequence[Any]]]]] = None,
        n_jobs: int = 1,
        only_physical_cores: bool = False,
        verbose: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        DScribe-like create; delegates to generate() and shares its center-wise
        output convention: (n_centers, n_features) per system.
        """
        return self.generate(
            system,
            centers,
            n_jobs=n_jobs,
            only_physical_cores=only_physical_cores,
            verbose=verbose,
        )

    def _evaluate(
        self,
        system: Any,
        centers: Optional[Sequence[Any]] = None,
        average: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Single-system forward pass with an explicit averaging mode.

        `average=None` uses self.average (the derivative routines always
        differentiate the descriptor itself, including the projection
        coefficients for average='cc'). Always returns a dense, center-wise
        tensor: (n_centers, n_features), for every average mode including
        'inner'.
        """
        avg = self.average if average is None else average
        # Internal compute in float64: the Löwdin-orthonormalized GTO basis is
        # severely ill-conditioned at large n_max (beta entries ~1e6 with
        # cancellation), so a float32 forward loses ~4 significant digits.
        # Computing in float64 and casting the final descriptor to self.dtype
        # costs nothing at these sizes and matches DScribe to ~1e-6.
        positions, Z, cell, pbc = system_to_tensors(system, self.device, torch.float64)
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
        # In an autograd forward pass, also restore the (l=1) gradient of the
        # dropped self edge; zero-valued at the evaluation point, so create()
        # output is unaffected.
        if positions.requires_grad or centers_xyz.requires_grad:
            self._add_self_gradient_terms(coeffs, positions, centers_xyz, center_indices, Z)
        if avg == "cc":
            return self._projection_cc(coeffs).to(self.dtype)  # (C, n_cc)
        return self._power_spectrum(coeffs, avg).to(self.dtype)  # (C, feat)

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
        d: float = 1e-4,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Numerical derivatives using central differences (step `d`, DScribe uses
        1e-4). The displaced evaluations run internally in float64 -- central
        differences in float32 lose most significant digits to cancellation --
        and the result is cast back to the descriptor dtype.

        DScribe DescriptorLocal convention:
          derivatives shape: (n_centers, n_atoms_included, 3, n_features)

        Every mode differentiates the descriptor itself. For average='cc' the
        descriptor is the projection coefficients, so the result is the exact
        analog of DScribe's cc numerical derivatives:

            d c_nlm / d x_j

        with the cc feature layout (species, n, lm) on the feature axis.

        include/exclude:
          - include: list of atom indices to differentiate w.r.t.
          - exclude: indices to omit
          - if both None: all atoms

        attach:
          - DScribe convention: every center is resolved once against the
            undisplaced geometry and stays fixed in space during the
            finite-difference displacements. With attach=True, a center given
            as an atomic index moves with that atom instead.
        """
        if method not in ("auto", "numerical"):
            raise ValueError("method must be 'auto' or 'numerical'.")
        # auto->numerical in DScribe

        # descriptor baseline in the native dtype
        desc0 = self.create(system, centers) if return_descriptor else None

        # Run the finite differences in float64 (see docstring). self.dtype is
        # switched only around the displaced forward passes (not thread-safe,
        # like the rest of the scratch/caching machinery).
        orig_dtype = self.dtype
        self.dtype = torch.float64
        try:
            deriv = self._derivatives_numerical_fd(system, centers, include, exclude, attach, d)
        finally:
            self.dtype = orig_dtype
        deriv = deriv.to(orig_dtype)

        return (deriv, desc0) if return_descriptor else deriv

    def _derivatives_numerical_fd(
        self,
        system: Any,
        centers: Optional[Sequence[Any]],
        include: Optional[Sequence[int]],
        exclude: Optional[Sequence[int]],
        attach: bool,
        d: float,
    ) -> torch.Tensor:
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

        # Prepare centers: resolve every center to a fixed point in space, once,
        # from the undisplaced geometry (DScribe attach=False behavior).
        centers_xyz, centers_idx = self.prepare_centers(positions, centers)
        n_centers = int(centers_xyz.shape[0])
        base_centers = [centers_xyz[k].detach().clone() for k in range(n_centers)]
        # atom each center re-attaches to when attach=True (-1: never moves)
        if attach:
            if centers is None:
                attached = list(range(n_centers))
            else:
                attached = [int(c) if isinstance(c, (int,)) else -1 for c in centers]
        else:
            attached = [-1] * n_centers

        # derivative target: always the descriptor itself; for 'cc' that is
        # the projection coefficients c_nlm, so this yields d c_nlm/d x_j
        target_avg = self.average

        # output tensor; every mode (including 'inner') is center-wise, one
        # Jacobian row per center
        n_rows = n_centers
        n_feat = self.get_number_of_features()
        deriv = torch.zeros((n_rows, len(idx), 3, n_feat), device=self.device, dtype=self.dtype)

        # Central difference for each atom and component
        # This is expensive; use only when you truly need derivatives.
        for ai, atom_index in enumerate(idx):
            for comp in range(3):
                shift = torch.zeros((n_atoms, 3), device=self.device, dtype=self.dtype)
                shift[atom_index, comp] = d

                # +d
                pos_p = positions + shift
                sys_p = {"positions": pos_p, "atomic_numbers": Z, "cell": cell, "pbc": pbc} if self.periodic else {"positions": pos_p, "atomic_numbers": Z}
                centers_p = [pos_p[a] if a == atom_index else b for a, b in zip(attached, base_centers)]
                f_p = self._evaluate(sys_p, centers_p, target_avg)

                # -d
                pos_m = positions - shift
                sys_m = {"positions": pos_m, "atomic_numbers": Z, "cell": cell, "pbc": pbc} if self.periodic else {"positions": pos_m, "atomic_numbers": Z}
                centers_m = [pos_m[a] if a == atom_index else b for a, b in zip(attached, base_centers)]
                f_m = self._evaluate(sys_m, centers_m, target_avg)

                # Derivative: (f_p - f_m) / (2d)
                # f_* is center-wise: (n_centers, n_feat) for every mode
                df = (f_p - f_m) / (2.0 * d)
                deriv[:, ai, comp, :] = df

        return deriv

    # ---- Public: derivatives (analytical) ----

    def derivatives_analytical(
        self,
        system: Any,
        centers: Optional[Sequence[Any]] = None,
        include: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        return_descriptor: bool = True,
        attach: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Exact analytical derivatives of the descriptor w.r.t. atomic coordinates,
        obtained by reverse-mode automatic differentiation of the (fully
        differentiable) forward pass.

        Every step of the forward pass -- sphericart real spherical harmonics, the
        GTO/polynomial radial factors, the Loewdin orthonormalization, the density
        projection c_{nlm}, the analytic self-term and the power spectrum -- is built
        from differentiable torch ops, so AD returns the *exact* gradient, not a
        finite-difference approximation. Every mode differentiates the
        descriptor itself; for average='cc' that is the projection
        coefficients, so the result is the analytical d c_{nlm}/d x_j.

        Output follows the DScribe convention:
            (n_centers, n_atoms_included, 3, n_features)

        Notes
        -----
        * Center convention matches DScribe (attach=False, the default): a center
          given as an atom index is resolved to a FIXED point in space from the
          input geometry and does NOT follow its atom during differentiation --
          the same convention as DScribe's numerical derivatives. The coincident
          atom is still a real neighbor of the fixed center, and the gradient of
          its r~0 self edge (nonzero only in the l=1 block) is restored by
          _add_self_gradient_terms. With attach=True, index centers stay tied to
          their atom in the autograd graph (the "center moves with the atom"
          term is then included automatically).
        * The neighbor search is a cutoff test that yields only integer indices, so it
          is non-differentiable by nature; the descriptor is exactly differentiable
          everywhere except on the measure-zero cutoff shell, exactly as in DScribe.
        """
        positions, Z, cell, pbc = system_to_tensors(system, self.device, self.dtype)
        n_atoms = int(positions.shape[0])

        bad = set(torch.unique(Z).tolist()) - self._atomic_number_set
        if bad:
            raise ValueError(f"System contains atomic numbers not in species list: {sorted(bad)}")

        # included atoms (atoms we differentiate w.r.t.)
        if include is None:
            idx = list(range(n_atoms))
        else:
            idx = [int(i) for i in include]
        if exclude is not None:
            ex = set(int(i) for i in exclude)
            idx = [i for i in idx if i not in ex]
        idx_t = torch.tensor(idx, device=self.device, dtype=torch.long)

        # Differentiable leaf for the positions, fed through the normal forward pass.
        pos = positions.detach().clone().requires_grad_(True)
        sys_t: Dict[str, Any] = {"positions": pos, "atomic_numbers": Z}
        if self.periodic:
            sys_t["cell"] = cell
            sys_t["pbc"] = pbc

        # Resolve centers ONCE from the undisplaced geometry. DScribe fixed-center
        # convention (attach=False): an index center is a fixed point in space, so
        # it must NOT be re-resolved from the differentiable positions (that would
        # tie it to its atom in the graph = attach=True behaviour). prepare_centers
        # re-detects the coincidence inside _evaluate, so the l=0 self-term (value)
        # and the l=1 self-edge gradient are still applied.
        centers_xyz0, _ = self.prepare_centers(positions, centers)
        n_centers = int(centers_xyz0.shape[0])
        if attach:
            eval_centers = centers  # index centers stay tied to their atom
        else:
            eval_centers = [centers_xyz0[i].detach() for i in range(n_centers)]

        # derivative target: always the descriptor itself; for 'cc' that is
        # the projection coefficients c_nlm, so this yields d c_nlm/d x_j
        desc = self._evaluate(sys_t, eval_centers, None)  # center-wise (R, n_feat)
        desc2 = desc
        R = int(desc2.shape[0])                   # one row per center, every mode
        n_feat = int(desc2.shape[1])
        P = R * n_feat

        # Full Jacobian d desc2 / d pos.  Try one vectorized (batched) reverse pass;
        # fall back to a per-output loop if batched AD is unsupported by a custom op.
        try:
            eye = torch.eye(P, device=self.device, dtype=self.dtype).reshape(P, R, n_feat)
            jac = torch.autograd.grad(
                desc2, pos, grad_outputs=eye, is_grads_batched=True, retain_graph=True
            )[0]
            jac = jac.reshape(R, n_feat, n_atoms, 3)
        except Exception:
            jac = torch.zeros((R, n_feat, n_atoms, 3), device=self.device, dtype=self.dtype)
            flat = desc2.reshape(-1)
            jflat = jac.view(P, n_atoms, 3)
            for p in range(P):
                g = torch.autograd.grad(flat[p], pos, retain_graph=True)[0]
                jflat[p] = g

        # (R, n_feat, n_atoms, 3) -> select included atoms -> (R, n_inc, 3, n_feat).
        deriv = jac[:, :, idx_t, :].permute(0, 2, 3, 1).contiguous()

        if return_descriptor:
            return deriv, desc.detach()
        return deriv

    def derivatives_analytical_closed_form(
        self,
        system: Any,
        centers: Optional[Sequence[Any]] = None,
        include: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        return_descriptor: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Explicit closed-form analytical derivative of the projection coefficients
        c_{nlm} -- a direct implementation of the formula derived in SOAP.pdf:

            d c_{nlm}/d x_j = sum_i sum_k (2π)^{3/2} σ^3 β^l_{nk}/(1+2σ^2 α_kl)^{l+3/2}
                              · d/dx_j [ r_i^l Y_lm(r̂_i) e^{-α_kl r_i^2/(1+2σ^2 α_kl)} ]

        The bracket is split exactly as
            d/dx [ r^l Y_lm(r̂) e^{-γ r^2} ]
                = e^{-γ r^2} · d/dx[ r^l Y_lm(r̂) ]            (solid-harmonic gradient)
                  + ( -2 γ x ) · r^l Y_lm(r̂) · e^{-γ r^2}     (Gaussian-width term)
        with γ = α_kl/(1+2σ^2 α_kl).  The solid-harmonic gradient d/dx[r^l Y_lm] is
        obtained directly and exactly from sphericart (SolidHarmonics), so we use the
        real-harmonic gradient that is consistent with the forward pass instead of the
        complex-harmonic recurrence written in the PDF (the two are equivalent; the
        real form is what the code actually evaluates).

        The position-independent self-term (the δ_{l0} contribution of the central
        Gaussian at r=0) drops out of the derivative.

        Restricted to rbf='gto', average='cc', non-periodic, weighting=None -- the
        setting of the SOAP.pdf derivation and of the force-prediction descriptor.
        For anything else use derivatives_analytical (autograd), which is general.

        Output: (n_centers, n_atoms_included, 3, n_features), DScribe convention.
        """
        if self._rbf != "gto":
            raise NotImplementedError("closed-form derivative needs rbf='gto'; use derivatives_analytical.")
        if self.average != "cc":
            raise NotImplementedError("closed-form derivative needs average='cc'; use derivatives_analytical.")
        if self.periodic:
            raise NotImplementedError("closed-form derivative is non-periodic; use derivatives_analytical.")
        if self._weighting is not None:
            raise NotImplementedError("closed-form derivative assumes weighting=None; use derivatives_analytical.")

        positions, Z, cell, pbc = system_to_tensors(system, self.device, self.dtype)
        device, dtype = positions.device, positions.dtype
        n_atoms = int(positions.shape[0])

        bad = set(torch.unique(Z).tolist()) - self._atomic_number_set
        if bad:
            raise ValueError(f"System contains atomic numbers not in species list: {sorted(bad)}")

        # included atoms and their position on output axis 1
        if include is None:
            idx = list(range(n_atoms))
        else:
            idx = [int(i) for i in include]
        if exclude is not None:
            ex = set(int(i) for i in exclude)
            idx = [i for i in idx if i not in ex]
        n_inc = len(idx)
        inv = torch.full((n_atoms,), -1, device=device, dtype=torch.long)  # atom -> out row (or -1)
        for out_pos, a in enumerate(idx):
            inv[a] = out_pos

        centers_xyz, center_indices = self.prepare_centers(positions, centers)
        n_centers = int(centers_xyz.shape[0])

        S = self.n_species
        nmax = self._n_max
        Lp1 = self._l_max + 1
        L = Lp1 * Lp1
        n_feat = S * nmax * L

        deriv = torch.zeros((n_centers, n_inc, 3, n_feat), device=device, dtype=dtype)
        desc0 = self.create(system, centers).detach() if return_descriptor else None

        # ---- neighbor list that *keeps* the neighbor atom index (non-periodic) ----
        eps_self = 1e-8 if dtype == torch.float32 else 1e-12
        center_idx, neigh_idx = self._radius_edges(centers_xyz, positions, self._cutoff)
        if center_idx.numel() > 0:
            rvec = positions[neigh_idx] - centers_xyz[center_idx]
            r = torch.linalg.norm(rvec, dim=-1)
            keep = r > eps_self
            center_idx, neigh_idx, rvec, r = center_idx[keep], neigh_idx[keep], rvec[keep], r[keep]
        E = int(center_idx.numel())

        if E == 0:
            return (deriv, desc0) if return_descriptor else deriv

        neigh_sp = self._map_Z_to_species(Z[neigh_idx])  # (E,) species column for each edge

        const = self._get_const(device, dtype)
        alphas = const["alphas"]   # (Lp1, nmax)
        betas = const["betas"]     # (Lp1, nmax, nmax)
        eta = const["eta"]
        r2 = r * r

        # B[e,n,l]   = sum_k beta^l_{nk} * pref_kl * e^{-gamma_kl r^2}              (radial, no r^l)
        # Bhat[e,n,l]= sum_k beta^l_{nk} * (-2 gamma_kl) * pref_kl * e^{-gamma_kl r^2}
        B = torch.zeros((E, nmax, Lp1), device=device, dtype=dtype)
        Bhat = torch.zeros((E, nmax, Lp1), device=device, dtype=dtype)
        for l in range(Lp1):
            alpha_l = alphas[l]                 # (nmax,)
            beta_l = betas[l]                   # (nmax,nmax)
            p = alpha_l + eta                   # (nmax,)
            gamma = alpha_l * eta / p           # (nmax,)  == alpha/(1+2 sigma^2 alpha)
            pref = (math.pi ** 1.5) * (eta / p) ** l * (p ** (-1.5))   # (nmax,)
            Q = pref[None, :] * torch.exp(-gamma[None, :] * r2[:, None])  # (E,nmax)
            B[:, :, l] = Q @ beta_l.transpose(0, 1)
            Bhat[:, :, l] = (-2.0 * gamma[None, :] * Q) @ beta_l.transpose(0, 1)

        # expand l -> (l,m) columns
        l_of_lm = torch.tensor(
            [l for l in range(Lp1) for _ in range(2 * l + 1)], device=device, dtype=torch.long
        )  # (L,)
        B_lm = B[:, :, l_of_lm]       # (E, nmax, L)
        Bhat_lm = Bhat[:, :, l_of_lm]  # (E, nmax, L)

        # real solid harmonics r^l Y_lm and their exact Cartesian gradients
        Yval, Ygrad = self._Ysolid.compute_with_gradients(rvec)  # (E,L), (E,3,L)

        # edge gradient  g[e,d,n,lm] = d/dr_d ( B_lm * Yval )
        #   = (r_d * Bhat_lm) * Yval     (Gaussian-width term)
        #   +  B_lm * dYval/dr_d         (solid-harmonic term)
        term_gauss = rvec[:, :, None, None] * (Bhat_lm[:, None, :, :] * Yval[:, None, None, :])
        term_solid = B_lm[:, None, :, :] * Ygrad[:, :, None, :]
        g = term_gauss + term_solid  # (E, 3, nmax, L)

        # ---- scatter edge gradients to (center, atom, component, species, n, lm) ----
        # d r_vec/d R_neigh = +I ;  d r_vec/d R_center_atom = -I
        deriv_flat = torch.zeros((n_centers * n_inc * S, 3, nmax, L), device=device, dtype=dtype)

        op_n = inv[neigh_idx]                       # out row of the neighbor atom (or -1)
        valid_n = op_n >= 0
        t_n = (center_idx * n_inc + op_n) * S + neigh_sp
        if torch.any(valid_n):
            deriv_flat.index_add_(0, t_n[valid_n], g[valid_n])

        a_c = center_indices[center_idx]            # atom index of each edge's center (or -1)
        op_c = torch.where(a_c >= 0, inv[a_c.clamp(min=0)], torch.full_like(a_c, -1))
        valid_c = op_c >= 0
        t_c = (center_idx * n_inc + op_c) * S + neigh_sp
        if torch.any(valid_c):
            deriv_flat.index_add_(0, t_c[valid_c], -g[valid_c])

        # (n_centers, n_inc, S, 3, nmax, L) -> (n_centers, n_inc, 3, S, nmax, L) -> features
        deriv = (
            self._to_dscribe_cc(deriv_flat.reshape(n_centers, n_inc, S, 3, nmax, L))
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(n_centers, n_inc, 3, n_feat)
            .contiguous()
        )

        return (deriv, desc0) if return_descriptor else deriv

    def derivatives_analytical_ps(
        self,
        system: Any,
        centers: Optional[Sequence[Any]] = None,
        include: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        return_descriptor: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Closed-form analytical derivative, fully vectorized on the GPU. For
        average='cc' it is the Jacobian of the projection coefficients
        themselves,

            d c_nlm / d x_a

        in the _projection_cc feature layout; for the power-spectrum modes it
        is the per-center power spectrum p_nn'l (equations.pdf):

            d p_nn'l / d x_a = pref_l * sum_m ( dc_nlm/dx_a c_n'lm + c_nlm dc_n'lm/dx_a )

        In both cases the edge gradient of c_nlm is evaluated exactly via
        sphericart SolidHarmonics (values + Cartesian gradients):

            d/dx [ B_l(r) * Y_solid,lm(rvec) ]
                = (-2 gamma x) B_l Y_solid,lm      (Gaussian-width term)
                + B_l * grad Y_solid,lm            (solid-harmonic term)

        DScribe fixed-center convention (attach=False): centers are fixed points
        in space, so only the +dc/dR_neighbor term is scattered (no
        center-motion term). r~0 self edges are KEPT in the edge list: solid
        harmonics are polynomials in rvec, exact at 0, which reproduces both the
        l=0 self value and the l=1-only self-edge gradient automatically.

        average='inner' is center-wise: row c is the Jacobian of the descriptor
        inner-averaged over center c alone (the coefficient average over a
        single center is the identity, so it coincides with the per-center
        power-spectrum Jacobian). This matches DScribe
        derivatives(centers=[c], average='inner') stacked over centers, and
        keeps one Jacobian row per center like the other modes.

        All heavy math runs in float64 (the Löwdin GTO basis is ill-conditioned)
        and the result is cast to self.dtype. This replaces the batched-autograd
        Jacobian (O(n_centers*n_features) reverse passes and an
        (n_out x n_out) identity) with a handful of einsums: for the 30-atom /
        n_max=10 / l_max=5 test case this is ~10 ms instead of ~190 s, using a
        few tens of MB instead of gigabytes.

        Supports rbf='gto' (closed-form radial factors) and rbf='polynomial'
        (radial factors by Gauss-Legendre quadrature, see _poly_B_Bhat); both
        share the same solid-harmonic factorization c_nlm = B_nl(r) *
        Y_solid,lm(rvec) and everything downstream of B/Bhat. Restricted to
        non-periodic, weighting=None; the general autograd path
        (derivatives_analytical) covers everything else.

        Output: (n_centers, n_atoms_included, 3, n_features), DScribe convention.
        """
        if self._rbf not in ("gto", "polynomial"):
            raise NotImplementedError("closed-form ps derivative needs rbf='gto' or 'polynomial'; use derivatives_analytical.")
        if self.periodic:
            raise NotImplementedError("closed-form ps derivative is non-periodic; use derivatives_analytical.")
        if self._weighting is not None:
            raise NotImplementedError("closed-form ps derivative assumes weighting=None; use derivatives_analytical.")

        device = self.device
        dtype = torch.float64
        out_dtype = self.dtype

        positions, Z, cell, pbc = system_to_tensors(system, device, dtype)
        n_atoms = int(positions.shape[0])

        bad = set(torch.unique(Z).tolist()) - self._atomic_number_set
        if bad:
            raise ValueError(f"System contains atomic numbers not in species list: {sorted(bad)}")

        # included atoms and their row on output axis 1
        if include is None:
            idx = list(range(n_atoms))
        else:
            idx = [int(i) for i in include]
        if exclude is not None:
            ex = set(int(i) for i in exclude)
            idx = [i for i in idx if i not in ex]
        n_inc = len(idx)
        inv = torch.full((n_atoms,), -1, device=device, dtype=torch.long)
        if n_inc > 0:
            inv[torch.tensor(idx, device=device, dtype=torch.long)] = torch.arange(n_inc, device=device, dtype=torch.long)

        centers_xyz, center_indices = self.prepare_centers(positions, centers)
        n_centers = int(centers_xyz.shape[0])

        S = self.n_species
        nmax = self._n_max
        Lp1 = self._l_max + 1
        L = Lp1 * Lp1
        # cc: Jacobian of the projection coefficients themselves,
        # (S*nmax*(l_max+1)^2 features); otherwise power-spectrum features.
        n_feat = self.get_number_of_features()

        # Every mode (including center-wise 'inner') has one Jacobian row per
        # center.
        n_rows = n_centers

        # ---- edge list, KEEPING r~0 self edges (solid harmonics exact at 0) ----
        center_idx, neigh_idx = self._radius_edges(centers_xyz, positions, self._cutoff)
        E = int(center_idx.numel())

        if E == 0:
            deriv = torch.zeros((n_rows, n_inc, 3, n_feat), device=device, dtype=out_dtype)
            if return_descriptor:
                return deriv, self.create(system, centers)
            return deriv

        rvec = positions[neigh_idx] - centers_xyz[center_idx]   # (E,3)
        r2 = (rvec * rvec).sum(dim=1)                           # (E,)
        neigh_sp = self._map_Z_to_species(Z[neigh_idx])         # (E,)

        const = self._get_const(device, dtype)
        eta = const["eta"]

        if self._rbf == "gto":
            alphas = const["alphas"]   # (Lp1, nmax)
            betas = const["betas"]     # (Lp1, nmax, nmax)

            # radial factors, all l at once:
            #   B[e,n,l]    = sum_k beta^l_{nk} pref_kl e^{-gamma_kl r^2}
            #   Bhat[e,n,l] = sum_k beta^l_{nk} (-2 gamma_kl) pref_kl e^{-gamma_kl r^2}
            p = alphas + eta                     # (Lp1,nmax)
            gamma = alphas * eta / p             # (Lp1,nmax)
            lvec = torch.arange(Lp1, device=device, dtype=dtype)
            pref = (math.pi ** 1.5) * (eta / p) ** lvec[:, None] * p ** (-1.5)     # (Lp1,nmax)
            Q = pref[None, :, :] * torch.exp(-gamma[None, :, :] * r2[:, None, None])  # (E,Lp1,nmax)
            B = torch.einsum("elk,lnk->enl", Q, betas)                              # (E,nmax,Lp1)
            Bhat = torch.einsum("elk,lnk->enl", (-2.0 * gamma[None, :, :]) * Q, betas)
        else:
            # polynomial radial factors by quadrature (exact r~0 self-edge limits)
            B, Bhat = self._poly_B_Bhat(torch.sqrt(r2))

        l_of_lm = torch.tensor(
            [l for l in range(Lp1) for _ in range(2 * l + 1)], device=device, dtype=torch.long
        )

        # real solid harmonics r^l Y_lm and exact Cartesian gradients
        Yval, Ygrad = self._Ysolid.compute_with_gradients(rvec)   # (E,L), (E,3,L)

        # forward coefficients c[c,s,n,lm] from the SAME edges: the r=0 self
        # edge contributes exactly the analytic l=0 self term (Y_solid,00 = y00).
        # The cc Jacobian does not involve them, so skip when only derivatives
        # are requested for 'cc'.
        need_coeffs = return_descriptor or self.average != "cc"
        if need_coeffs:
            cE = B[:, :, l_of_lm] * Yval[:, None, :]              # (E,nmax,L)
            Cacc = torch.zeros((n_centers * S, nmax * L), device=device, dtype=dtype)
            Cacc.index_add_(0, center_idx * S + neigh_sp, cE.reshape(E, -1))
            Cc = Cacc.view(n_centers, S, nmax, L)

            # Center-wise for every mode: the product rule below acts on each
            # center's own coefficients ('inner' over a single center is the
            # identity, so no averaging is applied).
            Cp = Cc

        op = inv[neigh_idx]
        use_triton = (
            _HAS_TRITON
            and self.average == "cc"
            and device.type == "cuda"
            and out_dtype == torch.float32
            and n_inc > 0
        )

        if use_triton:
            # Fused Triton fast path for the cc coefficient Jacobian: each
            # (center, atom) pair maps to at most one edge, so the edge gradient
            #   g = rvec (x) (Bhat * Y) + B (x) gradY        (all float64)
            # is computed on the fly and stored (already in the DScribe cc
            # convention: l=1 (z,x,y) permutation + pi^{-3/2} for l>=2)
            # straight into the final float32 Jacobian layout — no (E,3,n,L)
            # edge-gradient tensor, no float64 scatter buffer, no permute /
            # convention copies, and no zero-init memset (the kernel writes
            # every output element exactly once). Peak memory drops from ~8 GB
            # to little more than the output tensor itself for the 300-atom /
            # n_max=10 / l_max=5 case.
            deriv = torch.empty((n_rows, n_inc, 3, n_feat), device=device, dtype=out_dtype)
            valid = op >= 0
            emap = torch.full((n_rows * n_inc,), -1, device=device, dtype=torch.long)
            emap[center_idx[valid] * n_inc + op[valid]] = torch.arange(E, device=device)[valid]
            perm = torch.arange(L, device=device)
            if self._l_max >= 1:
                perm[1:4] = torch.tensor([2, 3, 1], device=device)
            scale = torch.ones(L, device=device, dtype=dtype)
            scale[4:] = math.pi ** -1.5
            BLOCK = 1024
            grid = (n_rows * n_inc, triton.cdiv(3 * n_feat, BLOCK))
            _cc_jacobian_dense_kernel[grid](
                rvec.contiguous(),
                B.contiguous(),
                Bhat.contiguous(),
                Yval.contiguous(),
                Ygrad.contiguous(),
                emap,
                neigh_sp.contiguous(),
                perm.to(torch.int32),
                l_of_lm.to(torch.int32),
                scale,
                deriv,
                n_feat,
                NMAX=nmax,
                LP1=Lp1,
                LSQ=L,
                BLOCK=BLOCK,
            )
            if return_descriptor:
                coeffs = [Cc[..., l * l:(l + 1) * (l + 1)] for l in range(Lp1)]
                desc = self._projection_cc(coeffs).to(out_dtype)
                return deriv, desc
            return deriv

        use_triton_ps = (
            _HAS_TRITON
            and self.average != "cc"
            and device.type == "cuda"
            and out_dtype == torch.float32
            and n_inc > 0
        )

        if use_triton_ps:
            # Fused Triton fast path for the power-spectrum Jacobian
            # ('off'/'inner'/'outer', all center-wise): the product rule
            #   d p_{jn,j'n',l}/d x_{a,d} = pref_l sum_m ( dc_{jnlm} c_{j'n'lm}
            #                                            + c_{jnlm} dc_{j'n'lm} )
            # is evaluated per (center, atom) pair directly from the edge
            # quantities and the per-center coefficients Cc, in float64, and
            # stored once into the final float32 Jacobian. This removes the
            # (E,3,n,L) edge-gradient tensor, the (C*A*S,3,n,L) float64
            # scatter buffer, the per-l (C,A,3,S,n,S,n) einsum intermediates
            # and the float64 output copy of the fallback below (~10 GB peak
            # for the 300-atom / n_max=10 / l_max=5 case) — peak memory drops
            # to little more than the float32 output itself.
            deriv = torch.empty((n_rows, n_inc, 3, n_feat), device=device, dtype=out_dtype)
            valid = op >= 0
            emap = torch.full((n_rows * n_inc,), -1, device=device, dtype=torch.long)
            emap[center_idx[valid] * n_inc + op[valid]] = torch.arange(E, device=device)[valid]
            fj, fjd, fn, fnd, fl, fpref = self._ps_feat_tables(device)
            # Per-edge contractions of the coefficient side of the product
            # rule with the solid harmonics and their gradients: the main
            # kernel then needs no per-m loop (two FMAs per term instead of a
            # (2l+1)-point dot product per term).
            U = torch.empty((E, S * nmax, Lp1), device=device, dtype=dtype)
            V = torch.empty((E, 3, S * nmax, Lp1), device=device, dtype=dtype)
            _ps_uv_kernel[(E,)](
                Yval.contiguous(),
                Ygrad.contiguous(),
                Cc.contiguous(),
                center_idx.contiguous(),
                U,
                V,
                NMAX=nmax,
                LP1=Lp1,
                LSQ=L,
                S=S,
                M=triton.next_power_of_2(2 * self._l_max + 1),
                BLOCK=triton.next_power_of_2(S * nmax * Lp1),
            )
            BLOCK = 256
            n_blk = triton.cdiv(3 * n_feat, BLOCK)
            grid = (n_rows * n_inc * n_blk,)
            _ps_jacobian_dense_kernel[grid](
                rvec.contiguous(),
                B.contiguous(),
                Bhat.contiguous(),
                U,
                V,
                emap,
                neigh_sp.contiguous(),
                fj, fjd, fn, fnd, fl, fpref,
                deriv,
                n_feat,
                n_inc,
                n_blk,
                NMAX=nmax,
                LP1=Lp1,
                S=S,
                BLOCK=BLOCK,
            )
            if return_descriptor:
                coeffs = [Cc[..., l * l:(l + 1) * (l + 1)] for l in range(Lp1)]
                desc = self._power_spectrum(coeffs, self.average).to(out_dtype)
                return deriv, desc
            return deriv

        deriv = torch.zeros((n_rows, n_inc, 3, n_feat), device=device, dtype=dtype)
        B_lm = B[:, :, l_of_lm]        # (E,nmax,L)
        Bhat_lm = Bhat[:, :, l_of_lm]  # (E,nmax,L)

        # edge gradient wrt the NEIGHBOR position (fixed centers: no center term)
        g = (
            rvec[:, :, None, None] * (Bhat_lm[:, None, :, :] * Yval[:, None, None, :])
            + B_lm[:, None, :, :] * Ygrad[:, :, None, :]
        )                                                          # (E,3,nmax,L)

        # scatter to D[c, a_out, s, 3, n, lm]
        if n_inc > 0:
            valid = op >= 0
            Dacc = torch.zeros((n_centers * n_inc * S, 3, nmax, L), device=device, dtype=dtype)
            if bool(valid.any()):
                t = (center_idx * n_inc + op.clamp(min=0)) * S + neigh_sp
                Dacc.index_add_(0, t[valid], g[valid])
            D = Dacc.view(n_centers, n_inc, S, 3, nmax, L)

        if self.average == "cc":
            # d c_{nlm}/d x_a directly: the scattered edge gradients ARE the
            # coefficient Jacobian; no power-spectrum product rule. Map to the
            # DScribe cc convention and the _projection_cc feature layout
            # (species, n, lm), flattened.
            if n_inc > 0:
                deriv = (
                    self._to_dscribe_cc(D)
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(n_rows, n_inc, 3, n_feat)
                    .contiguous()
                )
        else:
            # assemble d p / d x per l and fill DScribe feature ordering
            if self._feat_slices is None:
                self._build_feature_slices()
            triu_i = self._triu[0]
            triu_j = self._triu[1]

            if n_inc > 0:
                for l in range(Lp1):
                    s0, s1 = l * l, (l + 1) * (l + 1)
                    Cl = Cp[..., s0:s1]                                # (C,S,n,M)
                    Dl = D[..., s0:s1]                                 # (C,A,S,3,n,M)
                    pref_l = math.sqrt(8.0 * math.pi * math.pi / (2 * l + 1))
                    # X[c,a,d,j,n,k,q] = sum_m Dl[c,a,j,d,n,m] Cl[c,k,q,m]
                    X = torch.einsum("cajdnm,ckqm->cadjnkq", Dl, Cl)
                    PD = pref_l * (X + X.permute(0, 1, 2, 5, 6, 3, 4))
                    for (j, jd, l2, is_diag, start, end) in self._feat_slices:
                        if l2 != l:
                            continue
                        blk = PD[:, :, :, j, :, jd, :]                 # (C,A,3,n,n)
                        if is_diag:
                            deriv[..., start:end] = blk[..., triu_i, triu_j]
                        else:
                            deriv[..., start:end] = blk.reshape(n_rows, n_inc, 3, -1)

        deriv = deriv.to(out_dtype)

        if return_descriptor:
            # descriptor from the same coefficients: projection coefficients
            # for 'cc', per-center power spectrum otherwise
            coeffs = [Cc[..., l * l:(l + 1) * (l + 1)] for l in range(Lp1)]
            if self.average == "cc":
                desc = self._projection_cc(coeffs).to(out_dtype)
            else:
                desc = self._power_spectrum(coeffs, self.average).to(out_dtype)
            return deriv, desc
        return deriv

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
        """
        Dispatch to the requested derivative backend (DScribe-like `method`):

          - "numerical"  : central finite differences (derivatives_numerical).
          - "analytical" : exact autograd derivatives (derivatives_analytical),
                           general over rbf / average / periodic.
          - "auto"       : analytical when supported (here: always, since the autograd
                           path is general), else numerical -- mirroring DScribe, which
                           prefers analytical for SOAP.

        Every mode differentiates the descriptor itself. For average='cc'
        (whose create() returns the projection coefficients c_nlm) the output
        is the analytical coefficient Jacobian
            d c_nlm / d x_j
        with the cc feature layout, matching DScribe's cc numerical
        derivatives. 'inner' is center-wise (row c differentiates the
        descriptor inner-averaged over center c alone), so every mode returns
        (n_centers, n_atoms_included, 3, n_features).
        """
        if method == "numerical":
            return self.derivatives_numerical(
                system=system,
                centers=centers,
                include=include,
                exclude=exclude,
                method="numerical",
                return_descriptor=return_descriptor,
                attach=attach,
                n_jobs=n_jobs,
                only_physical_cores=only_physical_cores,
                verbose=verbose,
            )
        if method in ("auto", "analytical"):
            # Fast closed-form path: identical result to the autograd path under
            # the DScribe fixed-center convention, but ~4 orders of magnitude
            # faster (see derivatives_analytical_ps docstring).
            if (
                not attach
                and self._rbf in ("gto", "polynomial")
                and not self.periodic
                and self._weighting is None
                and self.average in ("off", "outer", "cc", "inner")
            ):
                return self.derivatives_analytical_ps(
                    system=system,
                    centers=centers,
                    include=include,
                    exclude=exclude,
                    return_descriptor=return_descriptor,
                )
            return self.derivatives_analytical(
                system=system,
                centers=centers,
                include=include,
                exclude=exclude,
                return_descriptor=return_descriptor,
                attach=attach,
            )
        raise ValueError("method must be one of: 'auto', 'analytical', 'numerical'.")
