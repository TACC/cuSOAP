from soap_generator import SOAP as PYTORCH_SOAP
from ase.io import read
import numpy as np
import torch
import time
import sys
import math

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

#data type: float32 or float64
dtype = 'float32'
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

# Setting up the pytorch_based SOAP descriptor
pytorch_soap = PYTORCH_SOAP(
    species=species,
    periodic=periodic,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    average=average,
    dtype=dtype,
)

# Molecule created as an ASE.Atoms
molecules = read(xyzfilename,format='xyz',index=":")

start_time = time.perf_counter()
for i, molecule in enumerate(molecules):
 # construct SOAP tensors
 pytorch_soap_tensor = pytorch_soap.create(molecule)
 # construct SOAP derivative tensors
 pytorch_soap_derivatives_batched = []
 num_atoms = len(molecule)
 if not is_batched:
     centers_batchsize = num_atoms
 num_batches = math.ceil(num_atoms / centers_batchsize) 
 for ibatch in range(num_batches):
     sindex = ibatch * centers_batchsize
     eindex = min(sindex + centers_batchsize, num_atoms)
     pytorch_soap_derivatives_batched.append(pytorch_soap.derivatives(system=molecule, centers=list(range(sindex, eindex)), method="analytical", return_descriptor=False))

# convert the SOAP derivatives list to a Pytorch tensor
pytorch_soap_derivatives = torch.cat(pytorch_soap_derivatives_batched, dim=0)
end_time = time.perf_counter()
duration_pytorch = end_time - start_time
print(f"PYTORCH_SOAP Execution time: {duration_pytorch:.4f} seconds")



