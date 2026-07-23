# cuSOAP

GPU-accelerated generator of the Smooth Overlap of Atomic Positions (SOAP)
descriptor, implemented in PyTorch with fused Triton (PTX) kernels. The API and
feature ordering are compatible with DScribe's `SOAP`.

On a 30,000-atom system the fused Triton pipeline generates SOAP vectors about
70x faster than DScribe on a single GPU.

## Features

- DScribe-like `SOAP(...)` constructor and `.create()` / `.derivatives()` API
- GTO and polynomial radial bases, weighting functions, periodic systems,
  all DScribe averaging modes plus `average="cc"` (projection coefficients,
  used for force prediction)
- Fused Triton kernels for neighbor search, real spherical harmonics, and
  density-coefficient accumulation (falls back to pure PyTorch when Triton or
  a GPU is unavailable, e.g. on CPU-only machines)
- Derivatives with respect to atomic coordinates:
  - `method="analytical"`: exact derivatives (autograd and closed-form paths)
  - `method="numerical"`: central finite differences
- Multi-GPU generation via `torch.multiprocessing` (see `examples/test_mp.py`)

## Installation

```bash
pip3 install cuSOAP
```

From source:

```bash
git clone https://github.com/TACC/cuSOAP
cd cuSOAP
pip3 install .
```

Requirements: Python >= 3.9, PyTorch >= 2.0, sphericart + sphericart-torch, ASE.
On Linux + CUDA, Triton is bundled with PyTorch. Optional: `torch_cluster` for a
faster neighbor search on GPU.

## Quickstart

```python
from cusoap import SOAP
from ase.io import read

soap = SOAP(
    species=["H", "O"],
    r_cut=10.0,
    n_max=7,
    l_max=3,
    periodic=False,
    average="outer",   # or "off", "inner", "cc"
)

molecule = read("water.xyz", format="xyz")
descriptor = soap.create(molecule)
derivatives = soap.derivatives(molecule, method="analytical", return_descriptor=False)
```

## Examples

```bash
python3 examples/test_serial.py examples/water.xyz    # single CPU/GPU
python3 examples/test_mp.py examples/water.xyz 4      # one worker per GPU
python3 examples/mini_energy_model.py examples/water.xyz
```

## Citation

cuSOAP: a GPU-accelerated Generator of Smooth Overlap of Atomic Positions
Descriptor (manuscript in preparation).
