"""
Validation metrics for Fokker-Planck solutions (utils/metrics.py)

Implements physics-aware metrics comparing neural solutions to analytical references.
"""

import torch
from typing import Tuple
from physics.ou_process import gaussian_pdf, evolve_mean, evolve_covariance_exact, generate_grid_points


def relative_l2_error(p_pred: torch.Tensor, 
                     p_true: torch.Tensor,
                     eps: float = 1e-10) -> float:
    """
    Compute relative L² error with numerical safeguards.
    
    Args:
        p_pred: Predicted solution (N,)
        p_true: Reference solution (N,)
        eps: Stabilizing term to prevent division by zero
        
    Returns:
        Relative error: ‖p_pred - p_true‖₂ / (‖p_true‖₂ + eps)
    """
    # Ensure contiguous memory layout
    p_pred = p_pred.contiguous().view(-1)
    p_true = p_true.contiguous().view(-1)
    
    # Compute norms with double precision for accuracy
    diff_norm = torch.linalg.norm(p_pred - p_true, dtype=torch.float64)
    true_norm = torch.linalg.norm(p_true, dtype=torch.float64) + eps
    
    return (diff_norm / true_norm).item()

def gaussian_kl_divergence(mean_pred: torch.Tensor,
                          cov_pred: torch.Tensor,
                          mean_true: torch.Tensor,
                          cov_true: torch.Tensor,
                          reg: float = 1e-6) -> float:
    """
    Compute KL divergence between two multivariate Gaussians with regularization.
    
    Implements:
        D_KL(true || pred) = 0.5[ tr(Σ_pred⁻¹Σ_true) 
                            + (μ_pred - μ_true)ᵀΣ_pred⁻¹(μ_pred - μ_true)
                            - n + ln(det(Σ_pred)/det(Σ_true)) ]
                            
    Args:
        mean_pred: Predicted mean (3,)
        cov_pred: Predicted covariance (3,3)
        mean_true: True mean (3,)
        cov_true: True covariance (3,3)
        reg: Ridge regularization for covariance matrices
        
    Returns:
        KL divergence in nats
    """
    # Add regularization to prevent singular matrices
    I = torch.eye(3, device=cov_pred.device)
    cov_pred_reg = cov_pred + reg * I
    cov_true_reg = cov_true + reg * I
    
    # Compute terms using Cholesky for numerical stability
    try:
        L = torch.linalg.cholesky(cov_pred_reg)
        cov_pred_inv = torch.cholesky_inverse(L)
    except RuntimeError:  # Fallback to pseudo-inverse
        cov_pred_inv = torch.linalg.pinv(cov_pred_reg)
    
    # Term 1: Trace(Σ_pred⁻¹Σ_true)
    trace_term = torch.trace(cov_pred_inv @ cov_true_reg)
    
    # Term 2: Mahalanobis distance
    delta = mean_pred - mean_true
    mahalanobis_term = delta @ cov_pred_inv @ delta
    
    # Term 3: Log determinant ratio
    logdet_pred = torch.logdet(cov_pred_reg)
    logdet_true = torch.logdet(cov_true_reg)
    logdet_ratio = logdet_pred - logdet_true
    
    # Combine all terms
    kl = 0.5 * (trace_term + mahalanobis_term - 3 + logdet_ratio)
    return kl.item()

def max_pointwise_error(p_pred: torch.Tensor,
                      p_true: torch.Tensor) -> float:
    """
    Compute maximum absolute error with input validation.
    
    Args:
        p_pred: Predicted solution (N,)
        p_true: Reference solution (N,)
        
    Returns:
        Maximum absolute error max|p_pred - p_true|
    """
    # Ensure compatible shapes
    if p_pred.shape != p_true.shape:
        raise ValueError(f"Shape mismatch: {p_pred.shape} vs {p_true.shape}")
        
    return torch.max(torch.abs(p_pred - p_true)).item()

def moments_from_samples(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and covariance from solution samples.
    
    Args:
        samples: Solution evaluations (N_points,)
        
    Returns:
        mean: (3,) tensor
        covariance: (3,3) tensor
    """
    # Require spatial coordinates and probability values
    if samples.dim() != 2 or samples.shape[1] != 4:
        raise ValueError("Samples must be shape (N,4) [x,y,z,p]")
    
    # Extract coordinates and probabilities
    coords = samples[:, :3]  # (N,3)
    probs = samples[:, 3]    # (N,)
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Compute weighted mean
    mean = torch.sum(coords * probs.unsqueeze(-1), dim=0)
    
    # Compute weighted covariance
    centered = coords - mean
    cov = (centered.T @ (centered * probs.unsqueeze(-1)))
    
    return mean, cov

def analytical_moments(t: float,
                      mu0: torch.Tensor,
                      sigma0: torch.Tensor,
                      A: torch.Tensor,
                      D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute analytical moments using OU process evolution.
    
    Args:
        t: Evaluation time
        mu0: Initial mean
        sigma0: Initial covariance
        A: Drift matrix
        D: Diffusion matrix
        
    Returns:
        mean_t: Mean at time t
        sigma_t: Covariance at time t
    """
    mu_t = evolve_mean(mu0, A, t)
    sigma_t = evolve_covariance_exact(sigma0, A, D, t)
    return mu_t, sigma_t

if __name__ == "__main__":
    # Test metrics with known Gaussian distributions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test distributions
    mean_true = torch.tensor([1.0, -0.5, 0.3], device=device)
    cov_true = torch.tensor([[0.2, 0.1, 0.0],
                            [0.1, 0.3, 0.0],
                            [0.0, 0.0, 0.4]], device=device)
    
    # Slightly perturbed predicted distribution
    mean_pred = mean_true + 0.05 * torch.randn(3, device=device)
    cov_pred = cov_true + 0.01 * torch.eye(3, device=device)
    
    # Generate samples
    grid = generate_grid_points((-5,5), 50, device=device)
    p_true = gaussian_pdf(grid, mean_true, cov_true)
    p_pred = gaussian_pdf(grid, mean_pred, cov_pred)
    
    # Compute metrics
    print(f"Relative L2: {relative_l2_error(p_pred, p_true):.4f}")
    print(f"KL Divergence: {gaussian_kl_divergence(mean_pred, cov_pred, mean_true, cov_true):.4f} nats")
    print(f"Max Error: {max_pointwise_error(p_pred, p_true):.4f}")