"""
Ornstein-Uhlenbeck process definitions and analytical solutions.

This module provides functions to compute the analytical solution of the
3D Ornstein-Uhlenbeck process and its corresponding Fokker-Planck equation.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def evolve_mean(mu_0: torch.Tensor, A: torch.Tensor, t: float) -> torch.Tensor:
    """
    Compute the mean of the OU process at time t.
    
    The mean evolves according to:
    μ(t) = exp(At) * μ_0
    
    Args:
        mu_0: Initial mean vector (3D)
        A: Drift matrix (3x3)
        t: Time point
        
    Returns:
        Mean vector at time t
    """
    # Compute matrix exponential: exp(A*t)
    At = A * t
    exp_At = torch.matrix_exp(At)
    
    # Apply to initial mean
    mu_t = torch.matmul(exp_At, mu_0)
    
    return mu_t


def evolve_covariance(Sigma_0: torch.Tensor, A: torch.Tensor, 
                      D: torch.Tensor, t: float, 
                      dt: float = 0.01) -> torch.Tensor:
    """
    Compute the covariance of the OU process at time t.
    
    The covariance evolves according to the matrix ODE:
    dΣ/dt = AΣ + ΣA^T + 2D
    
    This is solved numerically using Euler integration.
    
    Args:
        Sigma_0: Initial covariance matrix (3x3)
        A: Drift matrix (3x3)
        D: Diffusion matrix (3x3)
        t: Time point
        dt: Time step for numerical integration
        
    Returns:
        Covariance matrix at time t
    """
    # Clone to avoid modifying the input
    Sigma = Sigma_0.clone()
    
    # Number of steps
    n_steps = int(t / dt)
    remaining_time = t - n_steps * dt
    
    # Euler integration
    for _ in range(n_steps):
        # Compute derivative: dΣ/dt = AΣ + ΣA^T + 2D
        A_Sigma = torch.matmul(A, Sigma)
        Sigma_AT = torch.matmul(Sigma, A.transpose(-1, -2))
        dSigma_dt = A_Sigma + Sigma_AT + 2 * D
        
        # Update
        Sigma = Sigma + dSigma_dt * dt
    
    # Handle remaining time if any
    if remaining_time > 0:
        A_Sigma = torch.matmul(A, Sigma)
        Sigma_AT = torch.matmul(Sigma, A.transpose(-1, -2))
        dSigma_dt = A_Sigma + Sigma_AT + 2 * D
        Sigma = Sigma + dSigma_dt * remaining_time
    
    return Sigma


def evolve_covariance_exact(Sigma_0: torch.Tensor, A: torch.Tensor, 
                           D: torch.Tensor, t: float) -> torch.Tensor:
    """
    Compute the exact covariance of the OU process at time t.
    
    For constant A and D, the solution of the Lyapunov equation has a closed form:
    Σ(t) = exp(At)Σ_0 exp(A^T t) + ∫_0^t exp(A(t-s)) 2D exp(A^T(t-s)) ds
    
    For stable A (negative definite), as t→∞, Σ(t) approaches the steady-state solution,
    which satisfies: AΣ + ΣA^T + 2D = 0
    
    Args:
        Sigma_0: Initial covariance matrix (3x3)
        A: Drift matrix (3x3)
        D: Diffusion matrix (3x3)
        t: Time point
        
    Returns:
        Covariance matrix at time t
    """
    # Compute matrix exponentials
    At = A * t
    exp_At = torch.matrix_exp(At)
    exp_ATt = torch.matrix_exp(At.transpose(-1, -2))
    
    # First term: exp(At)Σ_0 exp(A^T t)
    first_term = torch.matmul(torch.matmul(exp_At, Sigma_0), exp_ATt)
    
    # For stable A, we can compute the steady-state covariance
    # Solve the Lyapunov equation AΣ_∞ + Σ_∞A^T + 2D = 0
    # This is an approximation assuming we're close to steady state
    
    # For the integral term, we use the fact that:
    # As t→∞, the solution approaches the steady-state: Σ_∞
    # Σ_∞ satisfies: AΣ_∞ + Σ_∞A^T + 2D = 0
    
    # We can approximate the integral term as:
    # Σ_∞ - exp(At)Σ_∞ exp(A^T t)
    
    # First, compute the steady-state covariance by numerically solving the Lyapunov equation
    # This is a very simplified solver and may not work for all cases
    # For production, consider using scipy.linalg.solve_continuous_lyapunov
    
    # Convert to numpy for the Lyapunov solver
    A_np = A.detach().cpu().numpy()
    D_np = D.detach().cpu().numpy()
    
    # Function to solve the continuous Lyapunov equation
    # AX + XA^T + Q = 0
    def solve_lyapunov(A, Q):
        n = A.shape[0]
        # Construct the Kronecker product system
        I = np.eye(n)
        # Solve (I ⊗ A + A ⊗ I)vec(X) = -vec(Q)
        kron_A = np.kron(I, A) + np.kron(A, I)
        vec_Q = Q.reshape(-1)
        vec_X = np.linalg.solve(kron_A, -vec_Q)
        X = vec_X.reshape(n, n)
        return X
    
    # Solve for the steady-state covariance
    Sigma_inf_np = solve_lyapunov(A_np, 2 * D_np)
    Sigma_inf = torch.tensor(Sigma_inf_np, dtype=Sigma_0.dtype, device=Sigma_0.device)
    
    # Second term: Σ_∞ - exp(At)Σ_∞ exp(A^T t)
    second_term = Sigma_inf - torch.matmul(torch.matmul(exp_At, Sigma_inf), exp_ATt)
    
    # Full solution: exp(At)Σ_0 exp(A^T t) + (Σ_∞ - exp(At)Σ_∞ exp(A^T t))
    Sigma_t = first_term + second_term
    
    # Ensure symmetry (numerical errors might break it)
    Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2))
    
    return Sigma_t


def gaussian_pdf(x: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute multivariate Gaussian probability density function.
    
    Args:
        x: Points where to evaluate the PDF, shape (batch_size, 3) or (batch_size, batch_size, batch_size, 3)
        mu: Mean vector, shape (3,)
        Sigma: Covariance matrix, shape (3, 3)
        
    Returns:
        PDF values at points x, shape matches input x's batch dimensions
    """
    # Get dimensionality and reshape x if needed
    original_shape = x.shape
    if len(original_shape) > 2:
        x = x.reshape(-1, original_shape[-1])
    
    # Center the data
    x_centered = x - mu
    
    # Compute precision matrix (inverse of covariance)
    try:
        L = torch.linalg.cholesky(Sigma)  # Cholesky decomposition for stable inversion
        precision = torch.cholesky_solve(torch.eye(3, device=Sigma.device), L)
    except:
        # If Cholesky fails, use regular inversion
        precision = torch.linalg.inv(Sigma)
    
    # Compute quadratic term (x-μ)ᵀΣ⁻¹(x-μ)
    # We do this manually to handle batched inputs
    quad_term = torch.sum(torch.matmul(x_centered, precision) * x_centered, dim=1)
    
    # Compute normalization constant: 1/((2π)^(d/2) |Σ|^(1/2))
    log_det = torch.logdet(Sigma)
    log_norm = -0.5 * (3 * np.log(2 * np.pi) + log_det)
    
    # Compute log PDF
    log_pdf = log_norm - 0.5 * quad_term
    
    # Convert to PDF
    pdf = torch.exp(log_pdf)
    
    # Reshape to original dimensions
    if len(original_shape) > 2:
        pdf = pdf.reshape(original_shape[:-1])
    
    return pdf


def ou_analytical_solution(x: torch.Tensor, t: float, mu_0: torch.Tensor, 
                           Sigma_0: torch.Tensor, A: torch.Tensor, 
                           D: torch.Tensor) -> torch.Tensor:
    """
    Compute the analytical solution of the FP equation for the OU process.
    
    Args:
        x: Spatial coordinates where to evaluate the solution
        t: Time point
        mu_0: Initial mean
        Sigma_0: Initial covariance
        A: Drift matrix
        D: Diffusion matrix
        
    Returns:
        PDF values at points x and time t
    """
    # Evolve mean and covariance to time t
    mu_t = evolve_mean(mu_0, A, t)
    Sigma_t = evolve_covariance_exact(Sigma_0, A, D, t)
    
    # Compute PDF
    p_xt = gaussian_pdf(x, mu_t, Sigma_t)
    
    return p_xt


def create_default_ou_params(device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create default parameters for the 3D OU process.
    
    Returns:
        Tuple containing (mu_0, Sigma_0, A, D)
    """
    # Initial mean
    mu_0 = torch.tensor([1.0, 0.5, -0.5], device=device)
    
    # Initial covariance (slightly anisotropic)
    Sigma_0 = torch.tensor([
        [0.2, 0.05, 0.02],
        [0.05, 0.25, -0.01],
        [0.02, -0.01, 0.3]
    ], device=device)
    
    # Drift matrix (stable system with some coupling)
    A = torch.tensor([
        [-1.0, 0.2, 0.0],
        [0.1, -0.8, 0.1],
        [0.0, 0.1, -1.2]
    ], device=device)
    
    # Diffusion matrix (positive definite)
    D = torch.tensor([
        [0.3, 0.05, 0.02],
        [0.05, 0.25, 0.03],
        [0.02, 0.03, 0.2]
    ], device=device)
    
    return mu_0, Sigma_0, A, D


def generate_grid_points(bounds: Tuple[float, float], n_points: int, dim: int = 3, device: torch.device = None) -> torch.Tensor:
    """
    Generate a grid of points in the specified domain.
    
    Args:
        bounds: Tuple (min_val, max_val) specifying domain bounds
        n_points: Number of points per dimension
        dim: Dimensionality (default 3)
        device: PyTorch device
        
    Returns:
        Grid points tensor of shape (n_points^dim, dim)
    """
    min_val, max_val = bounds
    
    # Create 1D grid for each dimension
    linspace = torch.linspace(min_val, max_val, n_points, device=device)
    
    # Create meshgrid
    if dim == 3:
        X, Y, Z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
        grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    elif dim == 2:
        X, Y = torch.meshgrid(linspace, linspace, indexing='ij')
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    elif dim == 1:
        grid_points = linspace.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}")
    
    return grid_points


if __name__ == "__main__":
    # Test the implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu_0, Sigma_0, A, D = create_default_ou_params(device)
    
    # Create grid points
    grid_points = generate_grid_points((-5, 5), 50, device=device)
    
    # Compute solution at different times
    t_values = [0.0, 0.5, 1.0, 2.0]
    
    for t in t_values:
        p_xt = ou_analytical_solution(grid_points, t, mu_0, Sigma_0, A, D)
        print(f"t={t}, min={p_xt.min().item()}, max={p_xt.max().item()}, sum={p_xt.sum().item() * (10/50)**3}")