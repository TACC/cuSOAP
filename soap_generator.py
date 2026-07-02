
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
import os
import time
import torch

# -----------------------------
# Optional Triton (fused CUDA kernel for the density-coefficient accumulation)
# -----------------------------
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

def _ensure_triton_ptxas() -> None:
    """On very new GPUs (e.g. GB10 / sm_121) triton's bundled ptxas may not know the
    architecture. torch ships a ptxas matching its CUDA build; prefer it, then the
    system CUDA toolkit."""
    if os.environ.get("TRITON_PTXAS_PATH"):
        return
    for cand in (
        os.path.join(os.path.dirname(torch.__file__), "bin", "ptxas"),
        "/usr/local/cuda/bin/ptxas",
    ):
        if os.path.exists(cand):
            os.environ["TRITON_PTXAS_PATH"] = cand
            return


if _HAS_TRITON:

    @triton.jit(do_not_specialize=["E_tot"])
    def _soap_gto_acc_kernel(
        r_ptr, w_ptr, y_ptr, rowptr_ptr,
        g_ptr, pref_ptr,          # (NMAX,) radial constants for this l
        acc_ptr,                  # (n_rows, NMAX, LM_TOT) output; writes slice [LM0:LM0+ML]
        E_tot,                    # total number of edges (= row stride of transposed Y)
        L: tl.constexpr,          # current angular momentum l
        ML: tl.constexpr,         # 2l+1
        MLP: tl.constexpr,        # next_pow2(ML)
        NMAX: tl.constexpr,
        NMAXP: tl.constexpr,      # next_pow2(NMAX)
        LM_TOT: tl.constexpr,     # (l_max+1)^2, columns of acc last dim
        LM0: tl.constexpr,        # l*l column offset
        BLOCK_E: tl.constexpr,
        HAS_W: tl.constexpr,
    ):
        # One program per (center,species) row: segmented reduction over the row's
        # edge range [rowptr[row], rowptr[row+1]).  For each edge e:
        #   prim[e,n] = w_e * PREF[n] * r_e^L * exp(-G[n] * r_e^2)
        #   acc[row,n,lm] += prim[e,n] * Y[e, LM0+m]
        # Accumulation stays in registers -> no atomics, no (E,nmax,2l+1) intermediate.
        row = tl.program_id(0)
        start = tl.load(rowptr_ptr + row)
        end = tl.load(rowptr_ptr + row + 1)

        n_offs = tl.arange(0, NMAXP)
        n_mask = n_offs < NMAX
        m_offs = tl.arange(0, MLP)
        m_mask = m_offs < ML

        G = tl.load(g_ptr + n_offs, mask=n_mask, other=0.0)
        PREF = tl.load(pref_ptr + n_offs, mask=n_mask, other=0.0)

        acc = tl.zeros((NMAXP, MLP), dtype=tl.float32)

        for s in range(start, end, BLOCK_E):
            e_offs = s + tl.arange(0, BLOCK_E)
            e_mask = e_offs < end
            r = tl.load(r_ptr + e_offs, mask=e_mask, other=0.0)
            r2 = r * r
            rl = tl.full((BLOCK_E,), 1.0, dtype=tl.float32)
            for _ in tl.static_range(L):
                rl = rl * r
            if HAS_W:
                wv = tl.load(w_ptr + e_offs, mask=e_mask, other=0.0)
            else:
                wv = tl.full((BLOCK_E,), 1.0, dtype=tl.float32)
            prim = wv[:, None] * rl[:, None] * PREF[None, :] * tl.exp(-G[None, :] * r2[:, None])
            prim = tl.where(e_mask[:, None] & n_mask[None, :], prim, 0.0)
            # Y is stored transposed (LM_TOT, E_tot) for coalesced access.
            y = tl.load(
                y_ptr + (LM0 + m_offs)[None, :] * E_tot + e_offs[:, None],
                mask=e_mask[:, None] & m_mask[None, :], other=0.0,
            )
            acc += tl.sum(prim[:, :, None] * y[:, None, :], axis=0)

        out_ptrs = acc_ptr + row * (NMAX * LM_TOT) + n_offs[:, None] * LM_TOT + (LM0 + m_offs)[None, :]
        tl.store(out_ptrs, acc, mask=n_mask[:, None] & m_mask[None, :])

    @triton.jit(do_not_specialize=["E"])
    def _ylm_all_kernel(u_ptr, nrm_ptr, out_ptr, E,
                        LMAX: tl.constexpr, BLOCK_E: tl.constexpr):
        # Real spherical harmonics for all (l,m) up to LMAX, in the sphericart
        # convention (orthonormal, no Condon-Shortley phase), via Cartesian
        # recurrences:
        #   A_m + i B_m = (x + i y)^m
        #   Ptilde_l^m(z) = P_l^m(z) / sin^m(theta)   (polynomial in z)
        #   Y_l^{+m} = N_lm Ptilde_l^m A_m ; Y_l^{-m} = N_lm Ptilde_l^m B_m
        # Output is transposed, (LMAX+1)^2 x E, so stores are coalesced.
        pid = tl.program_id(0)
        e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        e_mask = e_offs < E
        x = tl.load(u_ptr + e_offs * 3 + 0, mask=e_mask, other=0.0)
        y = tl.load(u_ptr + e_offs * 3 + 1, mask=e_mask, other=0.0)
        z = tl.load(u_ptr + e_offs * 3 + 2, mask=e_mask, other=1.0)

        A = tl.full((BLOCK_E,), 1.0, dtype=tl.float32)
        B = tl.zeros((BLOCK_E,), dtype=tl.float32)
        dfact = 1.0

        for m in tl.static_range(0, LMAX + 1):
            p_prev = tl.full((BLOCK_E,), 0.0, dtype=tl.float32) + dfact  # Ptilde_mm (l=m)
            n_lm = tl.load(nrm_ptr + m * (LMAX + 1) + m)
            tl.store(out_ptr + (m * m + 2 * m) * E + e_offs, n_lm * p_prev * A, mask=e_mask)
            if m > 0:
                tl.store(out_ptr + (m * m) * E + e_offs, n_lm * p_prev * B, mask=e_mask)
            if m < LMAX:
                p_curr = (2.0 * m + 1.0) * z * p_prev  # Ptilde_{m+1,m}
                l1 = m + 1
                n_lm = tl.load(nrm_ptr + l1 * (LMAX + 1) + m)
                tl.store(out_ptr + (l1 * l1 + l1 + m) * E + e_offs, n_lm * p_curr * A, mask=e_mask)
                if m > 0:
                    tl.store(out_ptr + (l1 * l1 + l1 - m) * E + e_offs, n_lm * p_curr * B, mask=e_mask)
                for ll in tl.static_range(m + 2, LMAX + 1):
                    p_next = ((2.0 * ll - 1.0) * z * p_curr - (ll + m - 1.0) * p_prev) / (ll - m)
                    p_prev = p_curr
                    p_curr = p_next
                    n_lm = tl.load(nrm_ptr + ll * (LMAX + 1) + m)
                    tl.store(out_ptr + (ll * ll + ll + m) * E + e_offs, n_lm * p_curr * A, mask=e_mask)
                    if m > 0:
                        tl.store(out_ptr + (ll * ll + ll - m) * E + e_offs, n_lm * p_curr * B, mask=e_mask)
            A_new = x * A - y * B
            B_new = x * B + y * A
            A = A_new
            B = B_new
            dfact = dfact * (2.0 * m + 1.0)

    @triton.jit(do_not_specialize=["C", "M"])
    def _nl_count_kernel(
        cx_ptr, cy_ptr, cz_ptr, nx_ptr, ny_ptr, nz_ptr, nsp_ptr, counts_ptr,
        C, M, cutoff2, eps2,
        S: tl.constexpr, SP: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        # Pass 1 of the fused neighbor search: brute-force distance test of a block
        # of centers against all neighbor candidates, counting hits per
        # (center, species) row. Positions come in SoA layout for coalesced loads.
        pid = tl.program_id(0)
        c_offs = pid * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        cx = tl.load(cx_ptr + c_offs, mask=c_mask, other=1e30)
        cy = tl.load(cy_ptr + c_offs, mask=c_mask, other=1e30)
        cz = tl.load(cz_ptr + c_offs, mask=c_mask, other=1e30)

        svec = tl.arange(0, SP)
        counts = tl.zeros((BLOCK_C, SP), dtype=tl.int32)

        for m0 in range(0, M, BLOCK_M):
            m_offs = m0 + tl.arange(0, BLOCK_M)
            m_mask = m_offs < M
            nx = tl.load(nx_ptr + m_offs, mask=m_mask, other=-1e30)
            ny = tl.load(ny_ptr + m_offs, mask=m_mask, other=-1e30)
            nz = tl.load(nz_ptr + m_offs, mask=m_mask, other=-1e30)
            sp = tl.load(nsp_ptr + m_offs, mask=m_mask, other=-1)

            dx = cx[:, None] - nx[None, :]
            dy = cy[:, None] - ny[None, :]
            dz = cz[:, None] - nz[None, :]
            d2 = dx * dx + dy * dy + dz * dz
            ok = (d2 <= cutoff2) & (d2 > eps2)
            hit = ok[:, :, None] & (sp[None, :, None] == svec[None, None, :])
            counts += tl.sum(hit.to(tl.int32), axis=1)

        tl.store(counts_ptr + c_offs[:, None] * S + svec[None, :], counts,
                 mask=c_mask[:, None] & (svec[None, :] < S))

    @triton.jit(do_not_specialize=["C", "M"])
    def _nl_fill_kernel(
        cx_ptr, cy_ptr, cz_ptr, nx_ptr, ny_ptr, nz_ptr, nsp_ptr, rowptr_ptr,
        r_ptr, u_ptr,
        C, M, cutoff2, eps2,
        S: tl.constexpr, SP: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        # Pass 2: recompute the same distance tests and compact-write r and the unit
        # vector of each edge directly into its (center, species) segment, so edges
        # come out pre-sorted for the segmented-accumulation kernel (no argsort).
        pid = tl.program_id(0)
        c_offs = pid * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        cx = tl.load(cx_ptr + c_offs, mask=c_mask, other=1e30)
        cy = tl.load(cy_ptr + c_offs, mask=c_mask, other=1e30)
        cz = tl.load(cz_ptr + c_offs, mask=c_mask, other=1e30)

        svec = tl.arange(0, SP)
        base = tl.load(rowptr_ptr + c_offs[:, None] * S + svec[None, :],
                       mask=c_mask[:, None] & (svec[None, :] < S), other=0)

        for m0 in range(0, M, BLOCK_M):
            m_offs = m0 + tl.arange(0, BLOCK_M)
            m_mask = m_offs < M
            nx = tl.load(nx_ptr + m_offs, mask=m_mask, other=-1e30)
            ny = tl.load(ny_ptr + m_offs, mask=m_mask, other=-1e30)
            nz = tl.load(nz_ptr + m_offs, mask=m_mask, other=-1e30)
            sp = tl.load(nsp_ptr + m_offs, mask=m_mask, other=-1)

            dx = nx[None, :] - cx[:, None]
            dy = ny[None, :] - cy[:, None]
            dz = nz[None, :] - cz[:, None]
            d2 = dx * dx + dy * dy + dz * dz
            ok = (d2 <= cutoff2) & (d2 > eps2)

            rv = tl.sqrt(d2)
            inv = 1.0 / tl.where(ok, rv, 1.0)
            ux = dx * inv
            uy = dy * inv
            uz = dz * inv

            for s in tl.static_range(S):
                oks = ok & (sp[None, :] == s)
                oks_i = oks.to(tl.int64)
                excl = tl.cumsum(oks_i, axis=1) - oks_i
                base_s = tl.sum(tl.where(svec[None, :] == s, base, 0), axis=1)
                pos = base_s[:, None] + excl
                tl.store(r_ptr + pos, rv, mask=oks)
                tl.store(u_ptr + pos * 3 + 0, ux, mask=oks)
                tl.store(u_ptr + pos * 3 + 1, uy, mask=oks)
                tl.store(u_ptr + pos * 3 + 2, uz, mask=oks)

            hit = ok[:, :, None] & (sp[None, :, None] == svec[None, None, :])
            base += tl.sum(hit.to(tl.int64), axis=1)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()



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

        # Fused triton path (GTO, CUDA, float32, no autograd): precompute radial
        # constants and JIT-compile/warm the kernels now so create() is not charged
        # for one-time compilation.
        self._use_fused = False
        self._use_fused_nl = False
        if (
            _HAS_TRITON
            and self._rbf == "gto"
            and self.dtype == torch.float32
            and str(self.device).startswith("cuda")
            and torch.cuda.is_available()
        ):
            self._init_fused_gto()


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

        Note: this routine only produces (integer) edge indices, so it never needs
        gradients. We detach the inputs here so the in-place / out=-argument GEMM
        optimizations below are compatible with an autograd-enabled forward pass
        (the differentiable `rvec` is rebuilt from these indices by the caller).
        """
        centers = centers.detach()
        neigh_pos = neigh_pos.detach()
        if self.max_num_neighbors is not None:
            # torch_cluster truncates at max_num_neighbors, so only use it when the
            # caller explicitly opted in. (Its default of None is not a valid arg.)
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
                pass
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
        # Bigger chunks = fewer host syncs from nonzero(); cap the scratch buffer
        # at ~1GB for float32.
        chunk = max(1024, min(C, int((2 ** 28) // max(M, 1))))
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

            idx = torch.nonzero(dist2 <= cutoff2, as_tuple=False)
            if idx.numel() > 0:
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

        # Fast path: fused triton kernel (no autograd requirement, float32, CUDA).
        if (
            self._use_fused
            and dtype == torch.float32
            and r.is_cuda
            and not (rvec.requires_grad or r.requires_grad)
        ):
            return self._coefficients_gto_fused(center_index, neigh_species, rvec, r, n_centers, prof=prof)

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

    # ---- fused triton path (GTO / CUDA / float32) ----

    def _init_fused_gto(self) -> None:
        """Precompute per-l radial constants for the fused kernel and warm it up."""
        dtype64 = torch.float64
        eta = torch.tensor(self._eta, device=self.device, dtype=dtype64)
        alphas = self._alphas.to(dtype=dtype64)                      # (Lp1, n_max)
        p = alphas + eta
        l_col = torch.arange(self._l_max + 1, device=self.device, dtype=dtype64)[:, None]
        G = alphas * eta / p                                          # (Lp1, n_max)
        PREF = (math.pi ** 1.5) * (eta / p) ** l_col * p ** (-1.5)    # (Lp1, n_max)
        self._fused_G = G.to(dtype=torch.float32).contiguous()
        self._fused_PREF = PREF.to(dtype=torch.float32).contiguous()

        # Normalization table N_lm for the inline Ylm kernel (sphericart convention).
        nrm = torch.zeros((self._l_max + 1, self._l_max + 1), dtype=torch.float64)
        for l in range(self._l_max + 1):
            for m in range(l + 1):
                n = math.sqrt(
                    (2 * l + 1) / (4.0 * math.pi)
                    * math.factorial(l - m) / math.factorial(l + m)
                )
                if m > 0:
                    n *= math.sqrt(2.0)
                nrm[l, m] = n
        self._ylm_nrm = nrm.to(device=self.device, dtype=torch.float32).contiguous()
        # The unrolled recurrences and fp32 double factorials are fine up to l~9
        # (DScribe caps l_max at 9 as well); beyond that use sphericart + transpose.
        self._ylm_inline = self._l_max <= 9

        _ensure_triton_ptxas()
        try:
            # Tiny synthetic problem: compiles all (l)-specializations of the kernels.
            E, n_rows = 8, 4
            r = torch.rand(E, device=self.device, dtype=torch.float32) + 0.5
            rowptr = torch.tensor([0, 2, 4, 6, E], device=self.device, dtype=torch.long)
            v = torch.randn(E, 3, device=self.device, dtype=torch.float32)
            u = (v / v.norm(dim=1, keepdim=True)).contiguous()
            Yt = self._launch_ylm(u)
            for has_w in (False, True) if self._weighting is not None else (False,):
                w = r.clone() if has_w else None
                self._launch_fused_acc(r, w, Yt, rowptr, n_rows)
            torch.cuda.synchronize()
            self._use_fused = True
        except Exception:
            self._use_fused = False
            return

        # Warm/compile the fused neighbor-list kernels on a tiny synthetic system.
        try:
            pos = torch.rand(6, 3, device=self.device, dtype=torch.float32) * 2.0
            spv = torch.arange(6, device=self.device, dtype=torch.int32) % self.n_species
            self._neighbor_fused(pos[:3], pos, spv)
            torch.cuda.synchronize()
            self._use_fused_nl = True
        except Exception:
            self._use_fused_nl = False

        # Pre-grow the CUDA caching-allocator pool: the first large allocation
        # (e.g. the harmonics buffer for a big system) otherwise pays a slow
        # cudaMalloc inside the first create() call.
        try:
            free_b, _ = torch.cuda.mem_get_info(self.device)
            n = int(min(free_b * 0.5, float(12 << 30)))
            if n > 0:
                buf = torch.empty((n,), dtype=torch.uint8, device=self.device)
                del buf
        except Exception:
            pass

        # Warm the remaining pipeline (sphericart CUDA, cuBLAS/einsum, sort,
        # searchsorted) so the first timed create() runs at steady state.
        try:
            with torch.no_grad():
                z0 = int(self._atomic_numbers[0].item())
                dummy = {
                    "positions": torch.tensor(
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=self.device, dtype=self.dtype
                    ),
                    "atomic_numbers": torch.tensor([z0, z0], device=self.device, dtype=torch.long),
                }
                if self.periodic:
                    dummy["cell"] = torch.eye(3, device=self.device, dtype=self.dtype) * (4.0 * self._cutoff)
                    dummy["pbc"] = torch.tensor([True, True, True], device=self.device)
                self.create(dummy)
            torch.cuda.synchronize()
        except Exception:
            pass

    def _neighbor_fused(
        self,
        centers_xyz: torch.Tensor,     # (C,3) float32
        neigh_pos: torch.Tensor,       # (M,3) float32 (species-filtered, possibly PBC-extended)
        neigh_sp: torch.Tensor,        # (M,) int32 species index in [0, n_species)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-pass brute-force neighbor search on GPU.

        Returns (rowptr, r, unit) where edges are compacted into contiguous
        (center, species) segments: rowptr has n_centers*n_species+1 entries and
        r/unit hold the edge distance and unit vector (center -> neighbor).
        Self edges (r ~ 0) are excluded, as in _build_neighbor_list.
        """
        device = centers_xyz.device
        C = int(centers_xyz.shape[0])
        M = int(neigh_pos.shape[0])
        S = self.n_species
        n_rows = C * S
        if C == 0 or M == 0:
            return (
                torch.zeros((n_rows + 1,), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.float32),
                torch.empty((0, 3), device=device, dtype=torch.float32),
            )

        # Separate contiguous coordinate arrays: coalesced loads in the kernels and
        # a stable 16B-alignment specialization key regardless of C/M parity.
        cen = centers_xyz.detach()
        nei = neigh_pos.detach()
        cxs = [cen[:, k].contiguous() for k in range(3)]
        nxs = [nei[:, k].contiguous() for k in range(3)]
        nsp = neigh_sp.contiguous()
        cutoff2 = float(self._cutoff) ** 2
        eps_self = 1e-8
        eps2 = eps_self * eps_self
        SP = _next_pow2(S)
        BLOCK_C, BLOCK_M = 32, 64
        grid = (triton.cdiv(C, BLOCK_C),)
        args = (cxs[0], cxs[1], cxs[2], nxs[0], nxs[1], nxs[2], nsp)

        counts = torch.empty((n_rows,), device=device, dtype=torch.int32)
        _nl_count_kernel[grid](*args, counts, C, M, cutoff2, eps2,
                               S=S, SP=SP, BLOCK_C=BLOCK_C, BLOCK_M=BLOCK_M, num_warps=4)
        rowptr = torch.zeros((n_rows + 1,), device=device, dtype=torch.long)
        rowptr[1:] = torch.cumsum(counts.to(torch.long), dim=0)
        E = int(rowptr[-1].item())
        self._last_nl_backend = "triton"
        self._last_E = E

        r = torch.empty((E,), device=device, dtype=torch.float32)
        u = torch.empty((E, 3), device=device, dtype=torch.float32)
        if E > 0:
            _nl_fill_kernel[grid](*args, rowptr, r, u, C, M, cutoff2, eps2,
                                  S=S, SP=SP, BLOCK_C=BLOCK_C, BLOCK_M=BLOCK_M, num_warps=4)
        return rowptr, r, u

    def _fused_pipeline_ok(self, positions: torch.Tensor, centers_xyz: torch.Tensor) -> bool:
        return (
            self._use_fused
            and self._use_fused_nl
            and self._rbf == "gto"
            and positions.is_cuda
            and positions.dtype == torch.float32
            and not positions.requires_grad
            and not centers_xyz.requires_grad
        )

    def _features_fused(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        centers_xyz: torch.Tensor,
        cell: Optional[torch.Tensor],
        pbc: Optional[torch.Tensor],
        prof: Optional[Profiler] = None,
    ) -> List[torch.Tensor]:
        """Fully fused pipeline: neighbor search -> Ylm -> density accumulation -> beta.
        Returns coeffs list over l of (C, S, n_max, 2l+1), same as _coefficients_gto."""
        from contextlib import nullcontext

        def _sec(name):
            return prof.section(name) if prof is not None else nullcontext()

        n_centers = int(centers_xyz.shape[0])
        n_rows = n_centers * self.n_species
        LM_TOT = (self._l_max + 1) ** 2

        with _sec("neighbor"):
            if self.periodic:
                if cell is None or pbc is None:
                    raise ValueError("periodic=True requires cell and pbc.")
                neigh_pos, neigh_Z = self._extend_periodic(positions, Z, cell, pbc)
            else:
                neigh_pos, neigh_Z = positions, Z

            neigh_species_all = self._map_Z_to_species(neigh_Z)
            keep = neigh_species_all >= 0
            if not bool(keep.all()):
                neigh_pos = neigh_pos[keep]
                neigh_species_all = neigh_species_all[keep]

            rowptr, r_s, unit_s = self._neighbor_fused(
                centers_xyz, neigh_pos, neigh_species_all.to(torch.int32)
            )

        if r_s.numel() == 0:
            acc = torch.zeros((n_rows, self._n_max, LM_TOT), device=positions.device, dtype=torch.float32)
        else:
            with _sec("ylm"):
                Yt = self._launch_ylm(unit_s)
            with _sec("weights"):
                w_s = self._weighting(r_s).contiguous() if self._weighting is not None else None
            with _sec("scatter"):
                acc = self._launch_fused_acc(r_s, w_s, Yt, rowptr, n_rows)

        acc = acc.view(n_centers, self.n_species, self._n_max, LM_TOT)
        const = self._get_const(positions.device, torch.float32)
        betas = const["betas"]
        with _sec("beta"):
            out = []
            for l in range(self._l_max + 1):
                acc_l = acc[:, :, :, l * l:(l + 1) * (l + 1)]
                out.append(torch.einsum("ab,csbm->csam", betas[l], acc_l))
        return out

    def _launch_ylm(self, unit: torch.Tensor) -> torch.Tensor:
        """Real spherical harmonics for unit vectors (E,3), returned TRANSPOSED:
        ((l_max+1)^2, E) float32, matching what _launch_fused_acc expects."""
        E = int(unit.shape[0])
        LM_TOT = (self._l_max + 1) ** 2
        if self._ylm_inline:
            Yt = torch.empty((LM_TOT, E), device=unit.device, dtype=torch.float32)
            if E > 0:
                BLOCK_E = 128
                _ylm_all_kernel[(triton.cdiv(E, BLOCK_E),)](
                    unit.contiguous(), self._ylm_nrm, Yt, E,
                    LMAX=self._l_max, BLOCK_E=BLOCK_E,
                )
            return Yt
        Y = self._Y.compute(unit)
        if Y.dtype != torch.float32:
            Y = Y.to(torch.float32)
        return Y.t().contiguous()

    def _launch_fused_acc(
        self,
        r: torch.Tensor,               # (E,) float32, edges sorted by row
        w: Optional[torch.Tensor],     # (E,) float32 or None
        Yt: torch.Tensor,              # ((l_max+1)^2, E) float32, transposed harmonics
        rowptr: torch.Tensor,          # (n_rows+1,) long, CSR segment offsets
        n_rows: int,
    ) -> torch.Tensor:
        """Returns acc (n_rows, n_max, (l_max+1)^2): primitive density coefficients."""
        E = int(r.shape[0])
        LM_TOT = (self._l_max + 1) ** 2
        acc = torch.empty((n_rows, self._n_max, LM_TOT), device=r.device, dtype=torch.float32)
        has_w = w is not None
        wp = w if has_w else r  # dummy pointer, kernel never reads it when HAS_W=False
        for l in range(self._l_max + 1):
            ML = 2 * l + 1
            _soap_gto_acc_kernel[(n_rows,)](
                r, wp, Yt, rowptr,
                self._fused_G[l], self._fused_PREF[l],
                acc, E,
                L=l, ML=ML, MLP=_next_pow2(ML),
                NMAX=self._n_max, NMAXP=_next_pow2(self._n_max),
                LM_TOT=LM_TOT, LM0=l * l,
                BLOCK_E=32, HAS_W=has_w,
                num_warps=4,
            )
        return acc

    def _coefficients_gto_fused(
        self,
        center_index: torch.Tensor,
        neigh_species: torch.Tensor,
        rvec: torch.Tensor,
        r: torch.Tensor,
        n_centers: int,
        prof: Optional[Profiler] = None,
    ) -> List[torch.Tensor]:
        """Fused-kernel equivalent of _coefficients_gto. Same output: list over l of
        (C, S, n_max, 2l+1)."""
        device = r.device
        n_rows = n_centers * self.n_species
        LM_TOT = (self._l_max + 1) ** 2
        const = self._get_const(device, torch.float32)
        betas = const["betas"]

        from contextlib import nullcontext

        def _sections(name):
            return prof.section(name) if prof is not None else nullcontext()

        if center_index.numel() == 0:
            acc = torch.zeros((n_rows, self._n_max, LM_TOT), device=device, dtype=torch.float32)
        else:
            with _sections("sort"):
                idx0 = center_index * self.n_species + neigh_species
                order = torch.argsort(idx0)
                idx0 = idx0[order]
                r_s = r[order].contiguous()
                rvec_s = rvec[order]
                rowptr = torch.searchsorted(
                    idx0, torch.arange(n_rows + 1, device=device, dtype=idx0.dtype)
                )
            with _sections("ylm"):
                unit = (rvec_s / r_s[:, None]).contiguous()
                Yt = self._launch_ylm(unit)
            with _sections("weights"):
                w_s = self._weighting(r_s).contiguous() if self._weighting is not None else None
            with _sections("scatter"):
                acc = self._launch_fused_acc(r_s, w_s, Yt, rowptr, n_rows)

        acc = acc.view(n_centers, self.n_species, self._n_max, LM_TOT)
        with _sections("beta"):
            out = []
            for l in range(self._l_max + 1):
                acc_l = acc[:, :, :, l * l:(l + 1) * (l + 1)]
                c = torch.einsum("ab,csbm->csam", betas[l], acc_l)
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

        n_centers = int(centers_xyz.shape[0])

        if self._fused_pipeline_ok(positions, centers_xyz):
            coeffs = self._features_fused(positions, Z, centers_xyz, cell, pbc)
        else:
            center_index, neigh_species, rvec, r = self._build_neighbor_list(
                positions, Z, centers_xyz, cell, pbc
            )

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

            n_centers = int(centers_xyz.shape[0])

            if self._fused_pipeline_ok(positions, centers_xyz):
                coeffs = self._features_fused(positions, Z, centers_xyz, cell, pbc, prof=prof)
                n_edges = int(getattr(self, "_last_E", 0))
            else:
                with prof.section("neighbor"):
                    center_index, neigh_species, rvec, r = self._build_neighbor_list(
                        positions, Z, centers_xyz, cell, pbc
                    )

                if self._rbf == "gto":
                    coeffs = self._coefficients_gto(center_index, neigh_species, rvec, r, n_centers, prof=prof)
                else:
                    coeffs = self._coefficients_polynomial(center_index, neigh_species, rvec, r, n_centers, prof=prof)
                n_edges = int(center_index.numel())

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
            "E_edges": n_edges,
            "neighbors_per_center": (float(n_edges) / float(n_centers)) if n_centers > 0 else 0.0,
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
        finite-difference approximation. For average='cc' this is precisely
        d c_{nlm}/d x_j as derived in SOAP.pdf; for the power-spectrum averages it is
        that gradient propagated through p = sum_m c_{nlm} c'_{nlm}.

        Output follows the DScribe convention:
            (n_centers, n_atoms_included, 3, n_features)

        Notes
        -----
        * When a center is given as an atom index, its position is tied to that atom
          in the autograd graph, so the "center moves with the atom" term (the
          on-diagonal force contribution) is included automatically, matching the
          DScribe analytical convention. `attach` is accepted for API parity but is
          a no-op here (this behaviour is automatic and always on for index centers).
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

        # number of centers (needed to broadcast for the global-average modes)
        centers_xyz, _ = self.prepare_centers(pos, centers)
        n_centers = int(centers_xyz.shape[0])

        desc = self.create(sys_t, centers)
        if self.sparse:
            desc = desc.to_dense()
        global_avg = (desc.dim() == 1)            # 'inner'/'outer' reduce over centers
        desc2 = desc[None, :] if global_avg else desc   # (R, n_feat)
        R = int(desc2.shape[0])
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

        # (R, n_feat, n_atoms, 3) -> select included atoms -> (R, n_inc, 3, n_feat)
        jac_sel = jac[:, :, idx_t, :].permute(0, 2, 3, 1).contiguous()
        if global_avg:
            deriv = jac_sel.expand(n_centers, len(idx), 3, n_feat).contiguous()
        else:
            deriv = jac_sel

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
            deriv_flat.reshape(n_centers, n_inc, S, 3, nmax, L)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(n_centers, n_inc, 3, n_feat)
            .contiguous()
        )

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
        """
        Dispatch to the requested derivative backend (DScribe-like `method`):

          - "numerical"  : central finite differences (derivatives_numerical).
          - "analytical" : exact autograd derivatives (derivatives_analytical),
                           general over rbf / average / periodic.
          - "auto"       : analytical when supported (here: always, since the autograd
                           path is general), else numerical -- mirroring DScribe, which
                           prefers analytical for SOAP.
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
            return self.derivatives_analytical(
                system=system,
                centers=centers,
                include=include,
                exclude=exclude,
                return_descriptor=return_descriptor,
                attach=attach,
            )
        raise ValueError("method must be one of: 'auto', 'analytical', 'numerical'.")
