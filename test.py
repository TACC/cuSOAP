from soap_generator import SOAP as PYTORCH_SOAP
from ase.io import read
import numpy as np

#output an array in a single line
np.set_printoptions(linewidth=np.inf)

#elements
species = ["H"]
#feel free to change the parameters below
r_cut = 10.0
n_max = 1 
l_max = 3 
periodic = False
#for energy prediction
#average = "outer"
#for force prediction
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

# Molecule created as an ASE.Atoms
molecule = read('water.xyz',format='xyz')
num_atoms = len(molecule)
for iatom in range(num_atoms):
 #Create pytorch-based SOAP tensor and its derivatives for the system
 pytorch_soap_tensor = pytorch_soap.create(molecule, centers=[iatom])
 pytorch_soap_derivatives = pytorch_soap.derivatives(system=molecule, centers=[iatom], method="numerical", return_descriptor=False)
 print("SOAP_Tensor: ",pytorch_soap_tensor)
 print("SOAP_Derivatives: ",pytorch_soap_derivatives)
