"""
Visualization utilities for 3D Fokker-Planck solutions (utils/visualization.py)

Provides intuitive visualizations of 3D distributions through 2D projections and error analysis.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple
from physics.ou_process import gaussian_pdf, generate_grid_points

def plot_2d_projection(p_values: torch.Tensor,
                      grid: torch.Tensor,
                      fix_dim: int = 2,
                      fix_val: float = 0.0,
                      title: str = "",
                      ax: Optional[plt.Axes] = None,
                      cmap: str = 'viridis') -> plt.Axes:
    """
    Plot 2D projection of 3D distribution by fixing one spatial dimension.
    
    Args:
        p_values: Solution values (N,)
        grid: Spatial coordinates (N, 3)
        fix_dim: Dimension to fix (0=x, 1=y, 2=z)
        fix_val: Value to fix dimension at
        title: Plot title
        ax: Existing axes to plot on
        cmap: Colormap for density values
        
    Returns:
        matplotlib axes with visualization
    """
    # Convert to CPU numpy if needed
    grid_np = grid.cpu().numpy()
    p_np = p_values.cpu().numpy()
    
    # Create mask for fixed dimension
    dim_tolerance = 0.05 * (grid[:, fix_dim].max() - grid[:, fix_dim].min()).item()
    mask = (torch.abs(grid[:, fix_dim] - fix_val) < dim_tolerance).cpu().numpy()
    
    # Create figure if no axes provided
    ax = ax or plt.gca()
    
    # Get free dimensions
    free_dims = [i for i in range(3) if i != fix_dim]
    x = grid_np[mask, free_dims[0]]
    y = grid_np[mask, free_dims[1]]
    z = p_np[mask]
    
    # Use tricontourf for irregular grids
    levels = np.linspace(z.min(), z.max(), 20)
    contour = ax.tricontourf(x, y, z, levels=levels, cmap=cmap)
    
    # Add colorbar and labels
    plt.colorbar(contour, ax=ax, label='Probability Density')
    ax.set_xlabel(['X', 'Y', 'Z'][free_dims[0]])
    ax.set_ylabel(['X', 'Y', 'Z'][free_dims[1]])
    ax.set_title(f"{title}\nFixed {['X','Y','Z'][fix_dim]} = {fix_val:.1f}")
    
    return ax

def plot_error_comparison(p_pred: torch.Tensor,
                         p_true: torch.Tensor,
                         grid: torch.Tensor,
                         fix_dim: int = 2,
                         fix_val: float = 0.0) -> plt.Figure:
    """
    Create side-by-side comparison of predicted and true solutions with error map.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot true solution
    plot_2d_projection(p_true, grid, fix_dim, fix_val, 
                      "Analytical Solution", axes[0])
    
    # Plot predicted solution
    plot_2d_projection(p_pred, grid, fix_dim, fix_val, 
                      "Neural Prediction", axes[1])
    
    # Plot absolute error
    error = (p_pred - p_true).abs()
    plot_2d_projection(error, grid, fix_dim, fix_val, 
                      "Absolute Error", axes[2], cmap='hot')
    
    plt.tight_layout()
    return fig

def plot_training_history(train_loss: list,
                         val_loss: list,
                         metrics: dict = None) -> plt.Figure:
    """
    Plot training and validation curves with optional metrics.
    
    Args:
        train_loss: Training loss history
        val_loss: Validation loss history
        metrics: Dictionary of metric histories
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot losses
    ax1.semilogy(train_loss, label='Training Loss', alpha=0.7)
    ax1.semilogy(val_loss, label='Validation Loss', alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend(loc='upper left')
    
    # Plot metrics if provided
    if metrics:
        ax2 = ax1.twinx()
        colors = ['tab:green', 'tab:purple', 'tab:orange']
        for idx, (name, values) in enumerate(metrics.items()):
            ax2.plot(values, label=name, color=colors[idx % len(colors)], 
                    linestyle='--', alpha=0.7)
        ax2.set_ylabel("Metric Values")
        ax2.legend(loc='upper right')
    
    plt.title("Training Progress")
    return fig

def create_animation(solution_series: list,
                    grid: torch.Tensor,
                    fix_dim: int = 2,
                    fix_val: float = 0.0,
                    interval: int = 100) -> FuncAnimation:
    """
    Create temporal animation of solution evolution.
    
    Args:
        solution_series: List of solution tensors over time
        grid: Spatial coordinates (N, 3)
        fix_dim: Fixed dimension for projection
        fix_val: Value of fixed dimension
        interval: Frame interval in milliseconds
        
    Returns:
        matplotlib FuncAnimation object
    """
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        plot_2d_projection(solution_series[frame], grid, fix_dim, fix_val, 
                          f"Time Step {frame}", ax)
        return ax
    
    return FuncAnimation(fig, update, frames=len(solution_series),
                        interval=interval, blit=False)

if __name__ == "__main__":
    # Test visualization with synthetic data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate test grid
    grid = generate_grid_points((-5, 5), 50, device=device)
    
    # Create Gaussian test data
    mu = torch.tensor([0.0, 0.0, 0.0], device=device)
    sigma = torch.eye(3, device=device)
    p_true = gaussian_pdf(grid, mu, sigma)
    
    # Create perturbed prediction
    p_pred = p_true + 0.1 * torch.randn_like(p_true)
    p_pred = torch.abs(p_pred)  # Ensure non-negative
    
    # Generate plots
    fig_comp = plot_error_comparison(p_pred, p_true, grid)
    fig_train = plot_training_history(
        train_loss=np.logspace(0, -3, 100).tolist(),
        val_loss=np.logspace(0, -3, 100).tolist() * 0.9,
        metrics={'L2 Error': np.linspace(0.5, 0.1, 100).tolist()}
    )
    
    plt.show()