from soap_generator import SOAP as PYTORCH_SOAP
from dscribe.descriptors import SOAP as DSCRIBE_SOAP
from ase.io import read
import numpy as np
import torch
import time

def calculate_crossing_angle(u, v, eps=1e-8):
    """
    Calculates the crossing angle (in radians) between two 1D tensors on CUDA.
    
    Args:
        u (torch.Tensor): First 1D tensor.
        v (torch.Tensor): Second 1D tensor.
        eps (float): Small epsilon value to prevent division by zero.
        
    Returns:
        torch.Tensor: The angle in radians.
    """
    # 1. Ensure tensors are on the same device (e.g., CUDA)
    u = u.to('cuda' if torch.cuda.is_available() else 'cpu')
    v = v.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Compute dot product and magnitudes
    dot_product = torch.dot(u, v)
    norm_u = torch.norm(u)
    norm_v = torch.norm(v)
    
    # 3. Calculate cosine ratio and clamp to valid range [-1, 1] 
    # (Prevents NaN errors from minor floating-point inaccuracies)
    cos_theta = dot_product / (norm_u * norm_v + eps)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # 4. Compute the angle in degrees using torch.arccos
    angle_in_degrees = torch.arccos(cos_theta) * (180.0 / torch.pi) 

    return angle_in_degrees


#output an array in a single line
np.set_printoptions(linewidth=np.inf)

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
molecules = read('water.xyz',format='xyz',index=":")

start_time = time.perf_counter()
for i, molecule in enumerate(molecules):
 num_atoms = len(molecule)
 print("PROCESSING STRUCTUE: ",i," WITH ",num_atoms," ATOMS")
 dscribe_soap_tensor = dscribe_soap.create(molecule)
end_time = time.perf_counter()
duration_dscribe = end_time - start_time
print(f"DSCRIBE_SOAP Execution time: {duration_dscribe:.4f} seconds")

start_time = time.perf_counter()
for i, molecule in enumerate(molecules):
 num_atoms = len(molecule)
 print("PROCESSING STRUCTUE: ",i," WITH ",num_atoms," ATOMS")
 pytorch_soap_tensor = pytorch_soap.create(molecule)
end_time = time.perf_counter()
duration_pytorch = end_time - start_time
print(f"PYTORCH_SOAP Execution time: {duration_pytorch:.4f} seconds")

performance_ratio = duration_dscribe/duration_pytorch
print(f"PYTORCH/DSCRIBE Performance Ratio",performance_ratio)
