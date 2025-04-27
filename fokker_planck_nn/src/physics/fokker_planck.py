"""
Fokker-Planck equation operators and residual calculations.

This module implements the operators and functions needed to compute
the residual of the Fokker-Planck equation for the 3D Ornstein-Uhlenbeck process.
"""

import torch
import numpy as np
from typing import Callable, Tuple


def compute_gradient(p_func: Callable, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """
    Compute the gradient of a scalar function with respect to x.
    
    Args:
        p_func: Function mapping points x to scalar values
        x: Input tensor of shape (batch_size, 3), requires_grad=True
        create_graph: Whether to create a computational graph (needed for higher derivatives)
        
    Returns:
        Gradient tensor of shape (batch_size, 3)
    """
    # Forward pass through the function
    p = p_func(x)
    
    # Compute gradient with respect to x
    grad_outputs = torch.ones_like(p)
    grad_x = torch.autograd.grad(
        outputs=p,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    return grad_x


def compute_divergence(vector_field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the divergence of a vector field.
    
    Args:
        vector_field: Vector field of shape (batch_size, 3)
        x: Input tensor of shape (batch_size, 3), requires_grad=True
        
    Returns:
        Divergence tensor of shape (batch_size)
    """
    # We need to compute ∇·v = ∂v₁/∂x₁ + ∂v₂/∂x₂ + ∂v₃/∂x₃
    batch_size = x.shape[0]
    divergence = torch.zeros(batch_size, device=x.device)
    
    # Compute each partial derivative independently
    for i in range(3):
        # Create gradient outputs for this component
        grad_outputs = vector_field[:, i]
        
        # Compute gradient of this component with respect to x
        grad_x = torch.autograd.grad(
            outputs=vector_field[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(grad_outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Add the i-th diagonal element to the divergence
        divergence += grad_x[:, i]
    
    return divergence


def compute_laplacian(p_func: Callable, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Laplacian of a scalar function.
    
    Args:
        p_func: Function mapping points x to scalar values
        x: Input tensor of shape (batch_size, 3), requires_grad=True
        
    Returns:
        Laplacian tensor of shape (batch_size)
    """
    # First compute the gradient
    gradient = compute_gradient(p_func, x, create_graph=True)
    
    # Then compute the divergence of the gradient (Laplacian)
    laplacian = compute_divergence(gradient, x)
    
    return laplacian


def fokker_planck_residual(p_model: Callable, x: torch.Tensor, t: torch.Tensor, 
                          A: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Compute the residual of the Fokker-Planck equation for the OU process.
    
    The residual is:
    R = ∂p/∂t - ∇·(Ax·p) - ∇·(D∇p)
    
    Args:
        p_model: Model function mapping (x,t) to probability density
        x: Spatial points tensor of shape (batch_size, 3), requires_grad=True
        t: Time points tensor of shape (batch_size), requires_grad=True
        A: Drift matrix (3x3)
        D: Diffusion matrix (3x3)
        
    Returns:
        Residual tensor of shape (batch_size)
    """
    batch_size = x.shape[0]
    
    # Create inputs that require gradient
    x_requires_grad = x.detach().clone().requires_grad_(True)
    t_requires_grad = t.detach().clone().requires_grad_(True)
    
    # Define lambda functions for partial evaluations
    p_func_x = lambda x_input: p_model(x_input, t_requires_grad)
    p_func_t = lambda t_input: p_model(x_requires_grad, t_input)
    
    # Compute p(x,t)
    p = p_model(x_requires_grad, t_requires_grad)
    
    # 1. Compute time derivative: ∂p/∂t
    dp_dt = compute_time_derivative(p_func_t, t_requires_grad)
    
    # 2. Compute first part: ∇·(Ax·p)
    div_Axp = compute_div_drift_term(p_func_x, x_requires_grad, A)
    
    # 3. Compute second part: ∇·(D∇p)
    div_D_grad_p = compute_div_diffusion_term(p_func_x, x_requires_grad, D)
    
    # Compute residual: ∂p/∂t - ∇·(Ax·p) - ∇·(D∇p)
    residual = dp_dt - div_Axp - div_D_grad_p
    
    return residual


def compute_time_derivative(p_func_t: Callable, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the time derivative of the probability density.
    
    Args:
        p_func_t: Function mapping time t to probability density
        t: Time points tensor, requires_grad=True
        
    Returns:
        Time derivative tensor
    """
    # Forward pass
    p = p_func_t(t)
    
    # Compute gradient with respect to t
    grad_outputs = torch.ones_like(p)
    dp_dt = torch.autograd.grad(
        outputs=p,
        inputs=t,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    return dp_dt


def compute_div_drift_term(p_func: Callable, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇·(Ax·p) term of the Fokker-Planck equation.
    
    Args:
        p_func: Function mapping points x to probability density
        x: Spatial points tensor, requires_grad=True
        A: Drift matrix (3x3)
        
    Returns:
        Divergence of drift term
    """
    batch_size = x.shape[0]
    
    # Compute p(x,t)
    p = p_func(x)
    
    # Compute Ax
    Ax = torch.matmul(x, A.transpose(0, 1))
    
    # Compute gradient of p
    grad_p = compute_gradient(p_func, x)
    
    # First term: (Ax)·∇p
    term1 = torch.sum(Ax * grad_p, dim=1)
    
    # Second term: p·∇·(Ax) = p·tr(A)
    # For constant A, ∇·(Ax) = tr(A)
    trace_A = torch.trace(A)
    term2 = p * trace_A
    
    # Combine: ∇·(Ax·p) = (Ax)·∇p + p·∇·(Ax)
    div_Axp = term1 + term2
    
    return div_Axp


def compute_div_diffusion_term(p_func: Callable, x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇·(D∇p) term of the Fokker-Planck equation.
    
    Args:
        p_func: Function mapping points x to probability density
        x: Spatial points tensor, requires_grad=True
        D: Diffusion matrix (3x3)
        
    Returns:
        Divergence of diffusion term
    """
    batch_size = x.shape[0]
    
    # For constant D, we can simplify:
    # ∇·(D∇p) = tr(D·H_p) = tr(D) * Laplacian(p)
    
    # 1. Compute the Laplacian of p
    laplacian_p = compute_laplacian(p_func, x)
    
    # 2. For more general cases, we would need to compute D:H_p where H_p is the Hessian
    # But for constant D, we can use the trace formula
    trace_D = torch.trace(D)
    div_D_grad_p = trace_D * laplacian_p
    
    return div_D_grad_p


def approximate_residual_with_finite_diff(p_model: Callable, x: torch.Tensor, 
                                         t: torch.Tensor, A: torch.Tensor, 
                                         D: torch.Tensor, h: float = 1e-3, 
                                         dt: float = 1e-3) -> torch.Tensor:
    """
    Compute the FP residual using finite differences (for testing).
    
    Args:
        p_model: Model function mapping (x,t) to probability density
        x: Spatial points tensor of shape (batch_size, 3)
        t: Time points tensor of shape (batch_size)
        A: Drift matrix (3x3)
        D: Diffusion matrix (3x3)
        h: Spatial step for finite differences
        dt: Time step for finite differences
        
    Returns:
        Residual tensor of shape (batch_size)
    """
    batch_size = x.shape[0]
    residual = torch.zeros(batch_size, device=x.device)
    
    # Create unit vectors
    e1 = torch.tensor([1., 0., 0.], device=x.device)
    e2 = torch.tensor([0., 1., 0.], device=x.device)
    e3 = torch.tensor([0., 0., 1.], device=x.device)
    
    # 1. Compute time derivative: ∂p/∂t
    p = p_model(x, t)
    p_forward = p_model(x, t + dt)
    dp_dt = (p_forward - p) / dt
    
    # 2. Compute spatial derivatives for the drift term
    # We'll use central differences
    for i in range(batch_size):
        xi = x[i]
        
        # Evaluate p at shifted points for each dimension
        p_center = p_model(xi.unsqueeze(0), t[i].unsqueeze(0))
        
        # Second derivatives for Laplacian (needed for diffusion term)
        laplacian = 0
        for d, e in enumerate([e1, e2, e3]):
            x_forward = xi + h * e
            x_backward = xi - h * e
            
            p_forward = p_model(x_forward.unsqueeze(0), t[i].unsqueeze(0))
            p_backward = p_model(x_backward.unsqueeze(0), t[i].unsqueeze(0))
            
            # Second derivative: (p(x+h) - 2p(x) + p(x-h)) / h²
            d2p_dx2 = (p_forward + p_backward - 2 * p_center) / (h * h)
            laplacian += D[d, d] * d2p_dx2
            
            # For the drift term
            # First derivative: (p(x+h) - p(x-h)) / 2h
            dp_dx = (p_forward - p_backward) / (2 * h)
            
            # Ax term for this dimension
            Ax_i = torch.sum(A[d] * xi)
            
            # Add contribution to drift term
            residual[i] -= (Ax_i * dp_dx)
        
        # Add Laplacian term for diffusion
        residual[i] -= laplacian
        
        # Add time derivative
        residual[i] += dp_dt[i]
    
    return residual


def conservation_of_mass_residual(p_model: Callable, domain_bounds: Tuple[float, float], 
                                grid_size: int, t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute the residual for conservation of mass: ∫p(x,t)dx = 1.
    
    Args:
        p_model: Model function mapping (x,t) to probability density
        domain_bounds: Tuple (min_val, max_val) for integration domain
        grid_size: Number of points per dimension for integration
        t: Time points tensor
        device: PyTorch device
        
    Returns:
        Residual tensor for conservation of mass
    """
    min_val, max_val = domain_bounds
    
    # Create integration grid
    linspace = torch.linspace(min_val, max_val, grid_size, device=device)
    X, Y, Z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    
    # Cell volume
    dx = (max_val - min_val) / (grid_size - 1)
    cell_volume = dx ** 3
    
    # Evaluate model at grid points for each time
    residuals = []
    for t_i in t:
        # Expand t_i to match grid_points
        t_expanded = t_i.expand(grid_points.shape[0])
        
        # Evaluate model
        p_values = p_model(grid_points, t_expanded)
        
        # Compute integral using rectangle rule
        integral = torch.sum(p_values) * cell_volume
        
        # Residual: |∫p dx - 1|
        residual = torch.abs(integral - 1.0)
        residuals.append(residual)
    
    return torch.stack(residuals)


def decay_boundary_residual(p_model: Callable, domain_bounds: Tuple[float, float], 
                           num_points: int, t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute a residual enforcing decay at the domain boundaries.
    
    Args:
        p_model: Model function mapping (x,t) to probability density
        domain_bounds: Tuple (min_val, max_val) for domain
        num_points: Number of boundary points to sample per face
        t: Time points tensor
        device: PyTorch device
        
    Returns:
        Residual tensor for boundary decay
    """
    min_val, max_val = domain_bounds
    
    # Create points for boundary sampling (6 faces of a cube)
    boundary_points = []
    
    # Sample points on each face
    for d in range(3):  # x, y, z dimension
        for val in [min_val, max_val]:  # min and max boundary
            # Create random points on this face
            points = torch.rand(num_points, 3, device=device) * (max_val - min_val) + min_val
            # Set the coordinate for this dimension to the boundary value
            points[:, d] = val
            boundary_points.append(points)
    
    # Combine all points
    boundary_points = torch.cat(boundary_points, dim=0)
    
    # Evaluate model at boundary points for each time
    residuals = []
    for t_i in t:
        # Expand t_i to match boundary_points
        t_expanded = t_i.expand(boundary_points.shape[0])
        
        # Evaluate model
        p_values = p_model(boundary_points, t_expanded)
        
        # Compute L2 norm of values at boundary
        # Since we want p ≈ 0 at boundaries
        residual = torch.mean(p_values ** 2)
        residuals.append(residual)
    
    return torch.stack(residuals)


if __name__ == "__main__":
    # Test the implementation with a simple Gaussian
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test function: Gaussian PDF
    def test_gaussian(x, t=None):
        mu = torch.tensor([0.0, 0.0, 0.0], device=device)
        sigma = 1.0 + 0.5 * t.view(-1, 1) if t is not None else 1.0
        norm = 1.0 / ((2 * np.pi * sigma**2) ** 1.5)
        return norm * torch.exp(-torch.sum((x - mu)**2, dim=1) / (2 * sigma**2))
    
    # Test points
    x = torch.randn(10, 3, device=device, requires_grad=True)
    t = torch.ones(10, device=device, requires_grad=True)
    
    # Test matrices
    A = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], device=device)
    
    D = torch.tensor([
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5]
    ], device=device)
    
    # Test gradient and divergence
    grad_p = compute_gradient(lambda x: test_gaussian(x), x)
    print("Gradient shape:", grad_p.shape)
    
    # Test Laplacian
    laplacian_p = compute_laplacian(lambda x: test_gaussian(x), x)
    print("Laplacian shape:", laplacian_p.shape)
    
    # Test residual
    residual = fokker_planck_residual(test_gaussian, x, t, A, D)
    print("FP residual shape:", residual.shape)
    print("FP residual mean:", residual.mean().item())
    
    # Test mass conservation
    mass_residual = conservation_of_mass_residual(
        test_gaussian, (-5.0, 5.0), 20, t[:2], device
    )
    print("Mass conservation residual:", mass_residual)
    
    # Test boundary decay
    boundary_residual = decay_boundary_residual(
        test_gaussian, (-5.0, 5.0), 100, t[:2], device
    )
    print("Boundary decay residual:", boundary_residual)