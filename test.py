from soap_generator import SOAP as PYTORCH_SOAP
from dscribe.descriptors import SOAP as DSCRIBE_SOAP
from ase.io import read
import numpy as np
import torch
import time
import sys

#output an array in a single line
np.set_printoptions(linewidth=np.inf)

#elements
species = ["H","O"]
#feel free to change the parameters below
r_cut = 10.0
n_max = 10 
l_max = 5 
periodic = False
#average = "off", "inner", "outer" or "cc"
average = "cc"

# Setting up the pytorch_based SOAP descriptor
pytorch_soap = PYTORCH_SOAP(
    species=species,
    periodic=periodic,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    average=average
)

# Setting up the dscribe_based SOAP descriptor
dscribe_soap = DSCRIBE_SOAP(
    species=species,
    periodic=periodic,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    average=average
)

# Molecule created as an ASE.Atoms
xyzfilename = sys.argv[1]
molecules = read(xyzfilename,format='xyz',index=":")

start_time = time.perf_counter()
for i, molecule in enumerate(molecules):
 num_atoms = len(molecule)
 #print("PROCESSING STRUCTUE: ",i," WITH ",num_atoms," ATOMS")
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

start_time = time.perf_counter()
for i, molecule in enumerate(molecules):
 num_atoms = len(molecule)
 #print("PROCESSING STRUCTUE: ",i," WITH ",num_atoms," ATOMS")

 pytorch_soap_tensor = pytorch_soap.create(system=molecule)
 pytorch_soap_derivatives_tensor = pytorch_soap.derivatives(system=molecule, method="analytical", return_descriptor=False)
 
 #print("PYTORCH_VECTOR: ",pytorch_soap_tensor.size())
 #print("PYTORCH_Derivatives: ",pytorch_soap_derivatives_tensor.size())

end_time = time.perf_counter()
duration_pytorch = end_time - start_time
print(f"PYTORCH_SOAP Execution time: {duration_pytorch:.4f} seconds")

#verification between DSCRIBE and PYTORCH
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dscribe_soap_tensor = torch.from_numpy(np.stack(dscribe_soap_list,axis=0)).to(device)
dscribe_soap_derivatives_tensor = torch.tensor(dscribe_soap_derivatives).to(device)
#print("SIZE: ",dscribe_soap_tensor.size(),dscribe_soap_derivatives_tensor.size())
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


