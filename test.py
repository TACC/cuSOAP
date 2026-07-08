from ase.io import read
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import sys
import math

#input arguments
#argument 1: input xyz file name
#batch size for centers

#elements
species = ["H","O"]
#feel free to change the parameters below
r_cut = 10.0
n_max = 10
l_max = 5
periodic = False
#average = "off", "inner", "outer" or "cc"
average = "outer"


def soap_worker(rank, world_size, xyzfilename, soap_params, tasks, result_queue, start_event):
    """One worker per GPU device: builds its own SOAP on cuda:<rank> and
    processes every world_size-th center batch (round-robin over ibatch)."""
    from soap_generator import SOAP as PYTORCH_SOAP

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    pytorch_soap = PYTORCH_SOAP(device=device, **soap_params)
    molecules = read(xyzfilename, format='xyz', index=":")

    # signal that construction/warmup is done, then wait for synchronized start
    result_queue.put(("ready", rank))
    start_event.wait()

    for itask in range(rank, len(tasks), world_size):
        imol, ibatch, sindex, eindex = tasks[itask]
        molecule = molecules[imol]
        desc = pytorch_soap.create(system=molecule, centers=range(sindex, eindex))
        deriv = pytorch_soap.derivatives(system=molecule, method="analytical",
                                         centers=range(sindex, eindex), return_descriptor=False)
        result_queue.put(("result", imol, ibatch, desc.cpu().numpy(), deriv.cpu().numpy()))


def main():
    from dscribe.descriptors import SOAP as DSCRIBE_SOAP

    xyzfilename = sys.argv[1]
    if len(sys.argv) < 3:
       is_batched = False
    else:
       is_batched = True
       centers_batchsize = int(sys.argv[2])

    #output an array in a single line
    np.set_printoptions(linewidth=np.inf)

    soap_params = dict(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
    )

    # Setting up the dscribe_based SOAP descriptor
    dscribe_soap = DSCRIBE_SOAP(**soap_params)

    # Molecule created as an ASE.Atoms
    molecules = read(xyzfilename,format='xyz',index=":")

    start_time = time.perf_counter()
    for i, molecule in enumerate(molecules):
     num_atoms = len(molecule)
     dscribe_soap_list = []
     dscribe_soap_derivatives = []
     for iatom in range(num_atoms):
             if average == "off":
              dscribe_soap_list.append(dscribe_soap.create(molecule, centers=[iatom])[0])
             else:
              dscribe_soap_list.append(dscribe_soap.create(molecule, centers=[iatom]))
             dscribe_soap_derivatives.append(dscribe_soap.derivatives(system=molecule, centers=[iatom], method="numerical")[0][0])
    end_time = time.perf_counter()
    duration_dscribe = end_time - start_time
    print(f"DSCRIBE_SOAP Execution time: {duration_dscribe:.4f} seconds")

    # Build the full list of (molecule, ibatch) tasks to distribute over GPUs
    tasks = []
    num_batches_per_mol = []
    for i, molecule in enumerate(molecules):
        num_atoms = len(molecule)
        bs = centers_batchsize if is_batched else num_atoms
        num_batches = math.ceil(num_atoms / bs)
        num_batches_per_mol.append(num_batches)
        for ibatch in range(num_batches):
            sindex = ibatch * bs
            eindex = min(sindex + bs, num_atoms)
            tasks.append((i, ibatch, sindex, eindex))

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Parallelizing {len(tasks)} center batches over {world_size} GPU device(s)")

    # Spawn one worker per GPU (CUDA requires the 'spawn' start method)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    start_event = ctx.Event()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=soap_worker,
                        args=(rank, world_size, xyzfilename, soap_params, tasks,
                              result_queue, start_event))
        p.start()
        procs.append(p)

    # wait until every worker finished SOAP construction/warmup
    n_ready = 0
    while n_ready < world_size:
        msg = result_queue.get()
        assert msg[0] == "ready"
        n_ready += 1

    # timed section: all workers start together, collect all batch results
    start_time = time.perf_counter()
    start_event.set()
    results = {}
    while len(results) < len(tasks):
        msg = result_queue.get()
        _, imol, ibatch, desc, deriv = msg
        results[(imol, ibatch)] = (desc, deriv)
    end_time = time.perf_counter()
    duration_pytorch = end_time - start_time

    for p in procs:
        p.join()

    print(f"PYTORCH_SOAP Execution time: {duration_pytorch:.4f} seconds")

    #verification between DSCRIBE and PYTORCH
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(len(molecules)):
        batches = [results[(i, ibatch)] for ibatch in range(num_batches_per_mol[i])]
        pytorch_soap_tensor = torch.cat([torch.from_numpy(d) for d, _ in batches]).to(device)
        pytorch_soap_derivatives_tensor = torch.cat([torch.from_numpy(dd) for _, dd in batches]).to(device)

    dscribe_soap_tensor = torch.from_numpy(np.stack(dscribe_soap_list,axis=0)).to(device)
    dscribe_soap_derivatives_tensor = torch.tensor(dscribe_soap_derivatives).to(device)
    max_diff_soap_tensor = torch.max(torch.abs(pytorch_soap_tensor - dscribe_soap_tensor)).item()
    max_soap_tensor = torch.max(torch.abs(pytorch_soap_tensor)).item()
    print("********************************************************************************")
    print("MAX DIFFERENCE on SOAP TENSOR: ",max_diff_soap_tensor," over ",max_soap_tensor)
    max_diff_soap_derivatives = torch.max(torch.abs(pytorch_soap_derivatives_tensor - dscribe_soap_derivatives_tensor)).item()
    max_soap_derivatives_tensor = torch.max(torch.abs(pytorch_soap_derivatives_tensor)).item()
    print("MAX DIFFERENCE on SOAP DERIVATIVES: ",max_diff_soap_derivatives," over ",max_soap_derivatives_tensor)
    print("********************************************************************************")

    performance_ratio = duration_dscribe/duration_pytorch
    print(f"PYTORCH/DSCRIBE Performance Ratio",performance_ratio)


if __name__ == "__main__":
    main()
