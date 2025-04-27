import torch
import torch.nn as nn
from itertools import combinations_with_replacement

# Hermite Polynomials and Derivatives 
def H(n: int, x: torch.Tensor):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    elif n == 2:
        return x**2 - 1
    elif n == 3:
        return x**3 - 3*x
    else:
        raise NotImplementedError("Extend H(n) for higher degrees")

def dH(n: int, x: torch.Tensor):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return torch.ones_like(x)
    elif n == 2:
        return 2 * x
    elif n == 3:
        return 3 * x**2 - 3
    else:
        raise NotImplementedError("Extend dH(n) for higher degrees")

def d2H(n: int, x: torch.Tensor):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return torch.zeros_like(x)
    elif n == 2:
        return 2 * torch.ones_like(x)
    elif n == 3:
        return 6 * x
    else:
        raise NotImplementedError("Extend d2H(n) for higher degrees")

# basis function genration
def generate_basis(max_degree: int):
    basis_indices = []
    for total in range(max_degree + 1):
        for n in range(total + 1):
            for m in range(total + 1 - n):
                k = total - n - m
                basis_indices.append((n, m, k))
    return basis_indices

##define the NN

# Coefficient net
class CoefficientNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=10, hidden_dims=[64]):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        layers.append(nn.Linear(hidden_dims[0], output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t):
        return self.net(t)

# test functions taken to be gaussians (3d)
class TestFunctions:
    def __init__(self, num_test=10):
        self.means = torch.randn(num_test, 3)  # Centers
        self.sigmas = torch.ones(num_test, 3)  # Diagonal covariance
    
    def evaluate(self, x: torch.Tensor):
        x_expanded = x.unsqueeze(1)  # [batch, 1, 3]
        means = self.means.unsqueeze(0)  # [1, num_test, 3]
        exponent = -0.5 * torch.sum(((x_expanded - means) / self.sigmas)**2, dim=2)
        return torch.exp(exponent)  # [batch, num_test]