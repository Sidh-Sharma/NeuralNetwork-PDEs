import torch
import argparse
from model import H, CoefficientNet, generate_basis

def analytical_ou_solution(t: torch.Tensor, x: torch.Tensor):
    """Analytical solution for 3D OU process (assuming stationary)."""
    return (2 * torch.pi)**(-1.5) * torch.exp(-0.5 * torch.sum(x**2, dim=1))

def validate(coeff_net, basis_indices):
    # Generate test samples
    x_test = torch.randn(1000, 3)
    t_test = torch.zeros(1000, 1)  # Stationary state (t → ∞ is assumed here)

    # Compute predicted density
    phi = []
    for (n, m, k) in basis_indices:
        Hn = H(n, x_test[:, 0])
        Hm = H(m, x_test[:, 1])
        Hk = H(k, x_test[:, 2])
        phi.append(Hn * Hm * Hk)
    phi = torch.stack(phi, dim=1)          # Shape: [1000, num_basis]
    c = coeff_net(t_test)                  # Shape: [1000, num_basis]
    p_pred = (c * phi).sum(dim=1)            # Shape: [1000]

    # Compute analytical solution
    p_true = analytical_ou_solution(t_test, x_test)  # Shape: [1000]

    # Compute relative error metric (L2 relative error)
    error = torch.mean((p_pred - p_true)**2) / torch.mean(p_true**2)
    print(f"Validation Relative Error: {error.item():.3e}")

    # Additional diagnostics: print mean and std values
    print(f"Predicted density: mean = {p_pred.mean().item():.3e}, std = {p_pred.std().item():.3e}")
    print(f"Analytical density: mean = {p_true.mean().item():.3e}, std = {p_true.std().item():.3e}")

def main():
    parser = argparse.ArgumentParser(description="Validate CoefficientNet model against analytical OU solution")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    args = parser.parse_args()

    # Define basis and network parameters
    max_degree = 3
    basis_indices = generate_basis(max_degree)
    num_basis = len(basis_indices)
    
    # Initialize network and load checkpoint
    coeff_net = CoefficientNet(output_dim=num_basis)
    state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    coeff_net.load_state_dict(state_dict)
    coeff_net.eval()  # Set network to evaluation mode

    # Run validation
    validate(coeff_net, basis_indices)

if __name__ == '__main__':
    main()
