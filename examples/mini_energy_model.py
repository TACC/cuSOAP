"""
Minimal PyTorch SOAP Energy Model demo: two fully connected layers joined by a ReLU.

    input --> Linear(in -> hidden) --> ReLU --> Linear(hidden -> out) --> output
"""

import torch
import torch.nn as nn
from cusoap import SOAP as PYTORCH_SOAP
from ase.io import read
import sys


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = sum(x)
        return x


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #elements
    species = ["H","O"]
    #feel free to change the parameters below
    r_cut = 10.0
    n_max = 10
    l_max = 5
    periodic = False
    #for energy prediction
    average = "outer"

    #define an SOAP generator 
    pytorch_soap = PYTORCH_SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
        device=device,
    )

    # read an XYZ file
    xyzfilename = sys.argv[1]
    molecule = read(xyzfilename, format='xyz')
    # number of atoms
    natoms = len(molecule)
    # construct a SOAP vector for the loaded molecule
    soap_tensors = pytorch_soap.create(system=molecule)
   
    # construct a two-layer MLP for the energy model
    in_dim = soap_tensors.size(1)
    hidden_dim, out_dim = 16, 1
    model = TwoLayerMLP(in_dim, hidden_dim, out_dim).to(device)
    print(model)

    # initiate the model with the constructed SOAP vector
    energy = model(soap_tensors)

    # A minimal training step to show it's wired up correctly.
    target = torch.randn(out_dim, device=device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for step in range(100):
        optimizer.zero_grad()
        pred = model(soap_tensors)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        print(f"step {step}: loss = {loss.item():.4f}")

    # enable the SOAP tensor gradients
    soap_tensors.requires_grad_(True)
    # set the optimized energy model into the evaluation state
    model.eval()
    # set the optimized energy model with the SOAP tensor input
    energy = model(soap_tensors)
    print("ENERGY EVALUATED!")
    # using autograd to get the derivative of energy with respect to the SOAP vetors
    denergy_dsoap = torch.autograd.grad(outputs=energy, inputs=soap_tensors)[0] 
    # calculate the derivative of SOAP vectors with respect to atomic coordinates
    dsoap_dxyz = pytorch_soap.derivatives(system=molecule, method="analytical", return_descriptor=False).permute(0,2,1,3).flatten(0,1) 
    # apply the chain rule to afford atomic forces
    # i.e., denergy_dxyz = denergy_dsoap * dsoap_dxyz
    forces = -1.0 * (denergy_dsoap * dsoap_dxyz).sum(dim=(1,2),keepdim=True).reshape(natoms,3)
    print("FORCES EVALUATED!")

    print("TASK ACCOMPLISEDH!")

if __name__ == "__main__":
    main()
