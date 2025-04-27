"""
Neural network initialization utilities (utils/initialization.py)

Provides physics-aware initialization methods for the low-rank decomposition model.
"""

import torch
import numpy as np
from typing import Callable, Tuple
from tqdm import tqdm
from models.basis_network import MultiModeBasisNetwork
from models.basis_network import BasisNetwork
# from models.temporal_network import TemporalNetwork

def initialize_basis_svd(
    basis_network: 'MultiModeBasisNetwork',
    analytical_solution: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    domain_bounds: Tuple[float, float] = (-5.0, 5.0),
    grid_resolution: int = 32,
    time_points: int = 50,
    device: torch.device = None,
    silent: bool = False
) -> None:
    """
    Initialize basis networks using POD modes from analytical solution snapshots.
    
    Theory: Performs SVD on solution snapshots matrix S = UΣV^T,
    initializing each φ_r(x) to match first R columns of U.

    Args:
        basis_network: MultiModeBasisNetwork to initialize
        analytical_solution: Function (x, t) ↦ p(x,t) for ground truth
        domain_bounds: Spatial domain extents (same for all dims)
        grid_resolution: Points per spatial dimension
        time_points: Number of temporal snapshots
        device: Target device for computations
        silent: Disable progress displays
    """
    # Device setup
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Generate temporal grid
    times = torch.linspace(0, 2.0, time_points, device=device)  # OU process reaches ~steady state by t=2
    
    # 2. Create spatial grid ([-5,5]^3)
    axis = torch.linspace(domain_bounds[0], domain_bounds[1], grid_resolution, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)
    grid_flat = grid.reshape(-1, 3)  # (N^3, 3)
    
    # 3. Collect solution snapshots
    snapshot_matrix = []
    if not silent:
        print("Generating analytical snapshots...")
    
    for t in tqdm(times, disable=silent):
        # Compute analytical solution at time t
        p_t = analytical_solution(grid_flat, t * torch.ones(1, device=device))
        snapshot_matrix.append(p_t)
    
    # 4. Form snapshot matrix (N^3 × T)
    S = torch.stack(snapshot_matrix, dim=1)  # (num_points, num_times)
    
    # 5. Compute truncated SVD (economy version)
    num_modes = min(basis_network.num_modes, time_points)
    U, _, _ = torch.pca_lowrank(S, q=num_modes, center=False)
    
    # 6. Fit basis networks to POD modes
    if not silent:
        print(f"Initializing {basis_network.num_modes} basis networks...")
    
    for mode_idx in tqdm(range(basis_network.num_modes), disable=silent):
        # Get target POD mode (flattened spatial pattern)
        target_mode = U[:, mode_idx].contiguous()
        
        # Initialize current basis network
        _fit_single_basis(
            network=basis_network.basis_networks[mode_idx],
            inputs=grid_flat,
            targets=target_mode,
            device=device,
            silent=silent
        )

def _fit_single_basis(
    network: 'BasisNetwork',
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    silent: bool = False,
    epochs: int = 2000,
    lr: float = 3e-4
) -> None:
    """
    Fit a single basis network to approximate target spatial pattern.
    
    Args:
        network: BasisNetwork instance to train
        inputs: Spatial coordinates (N, 3)
        targets: Target mode values (N,)
        device: Computation device
        silent: Disable training progress
        epochs: Training iterations
        lr: Learning rate
    """
    # Move network to device
    network.to(device)
    
    # Optimizer setup
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-6)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    progress = range(epochs)
    if not silent:
        progress = tqdm(progress, desc=f"Training basis {network._get_name()}")
    
    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        preds = network(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        if not silent:
            progress.set_postfix_str(f"Loss: {loss.item():.2e}")

if __name__ == "__main__":
    # Example usage
    from models.basis_network import MultiModeBasisNetwork
    from physics.ou_process import ou_analytical_solution, create_default_ou_params
    
    # Create sample network
    basis_net = MultiModeBasisNetwork(num_modes=4, orthogonalization=True)
    
    # Get default OU parameters
    params = create_default_ou_params()
    
    # Wrap analytical solution with fixed parameters
    def analytical_wrapper(x, t):
        return ou_analytical_solution(x, t, *params)
    
    # Perform SVD initialization
    initialize_basis_svd(
        basis_network=basis_net,
        analytical_solution=analytical_wrapper,
        grid_resolution=16,  # Smaller grid for quick testing
        time_points=20,
        silent=False
    )