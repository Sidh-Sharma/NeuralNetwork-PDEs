"""
Analytical solution generation for 3D OU process (data/analytical_data.py)

Generates ground truth data for validation using closed-form OU process solutions.
"""

import torch
from physics.ou_process import evolve_mean, evolve_covariance_exact, gaussian_pdf, create_default_ou_params
from typing import Tuple, Dict

def generate_grid_points(bounds: Tuple[float, float] = (-5.0, 5.0),
                        resolution: int = 50,
                        device: torch.device = None) -> torch.Tensor:
    """
    Generate 3D grid points for analytical solution evaluation.
    
    Args:
        bounds: Domain bounds for all dimensions
        resolution: Points per dimension
        device: PyTorch device for tensor placement
        
    Returns:
        Grid points tensor (N, 3) where N = resolution^3
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    axis = torch.linspace(bounds[0], bounds[1], resolution, device=device)
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing='ij')
    return torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

def create_validation_dataset(grid: torch.Tensor,
                             time_points: int = 20,
                             t_max: float = 2.0,
                             device: torch.device = None) -> Dict:
    """
    Generate validation dataset with analytical solutions at multiple time points.
    
    Args:
        grid: Spatial grid points (N, 3)
        time_points: Number of temporal snapshots
        t_max: Maximum simulation time
        device: PyTorch device for computations
        
    Returns:
        Dictionary containing:
        - grid: Spatial points (N, 3)
        - times: Temporal points (T,)
        - solutions: Probability densities (T, N)
        - params: OU process parameters
        - moments: Tuple of (means, covariances) for each time
    """
    device = device or grid.device
    params = create_default_ou_params(device)
    mu0, sigma0, A, D = params
    
    # Temporal grid
    times = torch.linspace(0, t_max, time_points, device=device)
    
    # Storage for solutions and moments
    solutions = []
    means = []
    covariances = []

    for t in times:
        # Compute analytical solution
        mu_t = evolve_mean(mu0, A, t.item())
        sigma_t = evolve_covariance_exact(sigma0, A, D, t.item())
        p = gaussian_pdf(grid, mu_t, sigma_t)
        
        # Store results
        solutions.append(p)
        means.append(mu_t)
        covariances.append(sigma_t)

    return {
        "grid": grid,
        "times": times,
        "solutions": torch.stack(solutions),
        "params": params,
        "moments": (torch.stack(means), torch.stack(covariances))
    }

def save_analytical_data(dataset: Dict, filename: str):
    """Save validation dataset to file."""
    torch.save(dataset, filename)

def load_analytical_data(filename: str, device: torch.device = None) -> Dict:
    """Load validation dataset from file."""
    dataset = torch.load(filename, map_location=device)
    return {
        "grid": dataset["grid"].to(device),
        "times": dataset["times"].to(device),
        "solutions": dataset["solutions"].to(device),
        "params": tuple(t.to(device) for t in dataset["params"]),
        "moments": (dataset["moments"][0].to(device), dataset["moments"][1].to(device))
    }

if __name__ == "__main__":
    # Test data generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate grid
    grid = generate_grid_points(device=device)
    
    # Create validation dataset
    dataset = create_validation_dataset(grid, time_points=20)
    
    # Save test dataset
    save_analytical_data(dataset, "validation_data.pt")
    
    print(f"Generated validation dataset with:")
    print(f"- Grid points: {dataset['grid'].shape}")
    print(f"- Time points: {len(dataset['times'])}")
    print(f"- Solutions: {dataset['solutions'].shape}")
    print(f"- Parameters: {[t.shape for t in dataset['params']]}")