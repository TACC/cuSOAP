"""cuSOAP: GPU-accelerated generator of the Smooth Overlap of Atomic Positions
(SOAP) descriptor, implemented in PyTorch with fused Triton (PTX) kernels.

Basic usage:

    from cusoap import SOAP

    soap = SOAP(species=["H", "O"], r_cut=10.0, n_max=7, l_max=3, average="outer")
    desc = soap.create(atoms)                                # descriptor
    deriv = soap.derivatives(atoms, method="analytical",
                             return_descriptor=False)        # d(desc)/d(positions)
"""

from .soap_generator import SOAP

__version__ = "0.1.0"
__all__ = ["SOAP"]
