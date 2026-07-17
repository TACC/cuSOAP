from cusoap import SOAP as PYTORCH_SOAP
from ase.io import read
import numpy as np
import torch
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

# Setting up the pytorch_based SOAP descriptor
pytorch_soap = PYTORCH_SOAP(
    species=species,
    periodic=periodic,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    average=average,
    rbf = rbf
)

# Molecule created as an ASE.Atoms
xyzfilename = sys.argv[1]
molecules = read(xyzfilename,format='xyz',index=":")

if len(sys.argv) < 3:
   is_batched = False
else:
   is_batched = True
   centers_batchsize = int(sys.argv[2])

start_time = time.perf_counter()
for i, molecule in enumerate(molecules):
 num_atoms = len(molecule)
 if not is_batched:
    pytorch_soap_tensor = pytorch_soap.create(system=molecule)
    pytorch_soap_derivatives_tensor = pytorch_soap.derivatives(system=molecule, method="analytical", return_descriptor=False)
 else:
    pytorch_soap_list = []
    pytorch_soap_derivatives_list = []
    num_batches = math.ceil(num_atoms / centers_batchsize)
    for ibatch in range(num_batches):
        sindex = ibatch * centers_batchsize 
        eindex = min(sindex + centers_batchsize, num_atoms)
        pytorch_soap_list.append(pytorch_soap.create(system=molecule, centers=range(sindex,eindex)))
        pytorch_soap_derivatives_list.append(pytorch_soap.derivatives(system=molecule,centers=range(sindex,eindex), method="analytical", return_descriptor=False))
    pytorch_soap_tensor = torch.cat(pytorch_soap_list)
    pytorch_soap_derivatives_tensor = torch.cat(pytorch_soap_derivatives_list)

end_time = time.perf_counter()
duration_pytorch = end_time - start_time
print(f"PYTORCH_SOAP Execution time: {duration_pytorch:.12f} seconds")


