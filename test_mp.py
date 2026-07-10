from soap_generator import SOAP as PYTORCH_SOAP
from ase.io import read
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import sys
import math

#output an array in a single line
np.set_printoptions(linewidth=np.inf)

#elements
species = ["H","O"]
#feel free to change the parameters below
r_cut = 10.0
n_max = 7
l_max = 3
periodic = False
#average = "off", "inner", "outer" or "cc"
average = "outer"
#rbf = "gto" or "polynomial"
rbf = "gto"


def batch_bounds(num_atoms, centers_batchsize):
    #no batchsize given: the whole molecule is a single batch
    bs = centers_batchsize if centers_batchsize is not None else num_atoms
    num_batches = math.ceil(num_atoms / bs)
    return bs, num_batches


def worker(rank, world_size, xyzfilename, centers_batchsize, queue):
    ndev = torch.cuda.device_count()
    device = f"cuda:{rank % ndev}" if ndev > 0 else "cpu"
    pytorch_soap = PYTORCH_SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
        rbf=rbf,
        device=device,
    )
    molecules = read(xyzfilename, format='xyz', index=":")
    for i, molecule in enumerate(molecules):
        num_atoms = len(molecule)
        bs, num_batches = batch_bounds(num_atoms, centers_batchsize)
        #ibatch parallelized over devices: rank takes ibatch = rank, rank+ws, ...
        for ibatch in range(rank, num_batches, world_size):
            sindex = ibatch * bs
            eindex = min(sindex + bs, num_atoms)
            desc = pytorch_soap.create(system=molecule, centers=range(sindex, eindex))
            deriv = pytorch_soap.derivatives(system=molecule, centers=range(sindex, eindex), method="analytical", return_descriptor=False)
            queue.put((i, ibatch, desc.cpu().numpy(), deriv.cpu().numpy()))


if __name__ == "__main__":
    # Molecule created as an ASE.Atoms
    xyzfilename = sys.argv[1]
    molecules = read(xyzfilename, format='xyz', index=":")

    centers_batchsize = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Parallelizing ibatch over {world_size} device(s)")

    start_time = time.perf_counter()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    spawn_ctx = mp.spawn(worker, args=(world_size, xyzfilename, centers_batchsize, queue),
                         nprocs=world_size, join=False)

    #drain the queue while workers run, then reassemble in ibatch order
    expected = sum(batch_bounds(len(m), centers_batchsize)[1] for m in molecules)
    desc_parts = [dict() for _ in molecules]
    deriv_parts = [dict() for _ in molecules]
    collected = 0
    while collected < expected:
        try:
            i, ibatch, desc, deriv = queue.get(timeout=5)
        except Exception:
            #surface worker crashes instead of blocking forever on an empty queue
            if spawn_ctx.join(timeout=0):
                raise RuntimeError(f"workers exited after {collected}/{expected} results")
            continue
        desc_parts[i][ibatch] = desc
        deriv_parts[i][ibatch] = deriv
        collected += 1
    spawn_ctx.join()

    for i, molecule in enumerate(molecules):
        pytorch_soap_tensor = np.concatenate([desc_parts[i][b] for b in sorted(desc_parts[i])])
        pytorch_soap_derivatives_tensor = np.concatenate([deriv_parts[i][b] for b in sorted(deriv_parts[i])])

    end_time = time.perf_counter()
    duration_pytorch = end_time - start_time
    print(f"PYTORCH_SOAP Execution time: {duration_pytorch:.12f} seconds")
