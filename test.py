from soap_generator import SOAP as PYTORCH_SOAP
from ase.io import read
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import sys
import math

#elements
species = ["H","O"]
#feel free to change the parameters below
r_cut = 10.0
n_max = 10
l_max = 5
periodic = False
#for energy prediction
average = "outer"
#for force prediction
#average = "cc"


def worker(rank, world_size, xyzfilename, work_items, queue, ready_barrier, done_barrier):
    """Compute SOAP vectors/derivatives for work_items[rank::world_size] on cuda:<rank>."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    # Per-process setup (SOAP init, file read) happens before the ready barrier
    # so the parent's timed section only covers the actual computation.
    pytorch_soap = PYTORCH_SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
        device=device,
    )
    molecules = read(xyzfilename, format='xyz', index=":")

    ready_barrier.wait()

    for gidx in range(rank, len(work_items), world_size):
        imol, sindex, eindex = work_items[gidx]
        molecule = molecules[imol]
        centers = list(range(sindex, eindex))
        # construct SOAP tensors
        vec = pytorch_soap.create(system=molecule, centers=centers)
        # construct SOAP derivative tensors
        der = pytorch_soap.derivatives(system=molecule, centers=centers, method="analytical", return_descriptor=False)
        queue.put((gidx, vec.cpu(), der.cpu()))

    # Queued CPU tensors are shared via file descriptors served by this process;
    # stay alive until the parent has drained the queue, or its get() fails.
    done_barrier.wait()


if __name__ == "__main__":
    #command line arguments
    #first argument is the name of the input XYZ file
    xyzfilename = sys.argv[1]
    #second argument is the batchsize of the centers for SOAP vector generation
    #for example, if the system has 100 atoms and the centers_batchsize is 5, a total of 20 batches will be executed.
    if len(sys.argv) < 3:
        is_batched = False
    else:
        is_batched = True
        centers_batchsize = int(sys.argv[2])

    print(is_batched)

    #output an array in a single line
    np.set_printoptions(linewidth=np.inf)

    # Molecules created as ASE.Atoms (read here only to build the work list;
    # each worker re-reads the file itself)
    molecules = read(xyzfilename, format='xyz', index=":")

    # Global work list: one item per (molecule, center batch), preserving the
    # sequential order so results can be reassembled deterministically.
    work_items = []
    for imol, molecule in enumerate(molecules):
        num_atoms = len(molecule)
        bs = centers_batchsize if is_batched else num_atoms
        num_batches = math.ceil(num_atoms / bs)
        for ibatch in range(num_batches):
            sindex = ibatch * bs
            eindex = min(sindex + bs, num_atoms)
            work_items.append((imol, sindex, eindex))

    # One worker process per visible GPU (single CPU worker if no GPU)
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Parallelizing over {world_size} device(s)")

    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    ready_barrier = ctx.Barrier(world_size + 1)
    done_barrier = ctx.Barrier(world_size + 1)

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=worker,
            args=(rank, world_size, xyzfilename, work_items, queue, ready_barrier, done_barrier),
        )
        p.start()
        processes.append(p)

    # Wait until every worker has finished setup, then time only the compute.
    ready_barrier.wait()
    start_time = time.perf_counter()

    results = {}
    for _ in range(len(work_items)):
        gidx, vec, der = queue.get()
        results[gidx] = (vec, der)

    # Release the workers now that all queued tensors have been received.
    done_barrier.wait()

    # Reassemble per molecule in the original batch order (as in the
    # sequential version, the final tensors are those of the last molecule).
    for imol in range(len(molecules)):
        pytorch_soap_batched = [results[g][0] for g, (m, s, e) in enumerate(work_items) if m == imol]
        pytorch_soap_derivatives_batched = [results[g][1] for g, (m, s, e) in enumerate(work_items) if m == imol]

    pytorch_soap = torch.cat(pytorch_soap_batched, dim=0)
    pytorch_soap_derivatives = torch.cat(pytorch_soap_derivatives_batched, dim=0)
    end_time = time.perf_counter()
    duration_pytorch = end_time - start_time
    print(f"PYTORCH_SOAP Execution time: {duration_pytorch:.4f} seconds")

    for p in processes:
        p.join()
