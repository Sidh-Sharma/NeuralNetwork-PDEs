"""
Physics-informed loss components (training/losses.py)

Implements composite loss function for Fokker-Planck equation solution:
- PDE residual loss
- Mass conservation loss
- Boundary condition loss
"""

import torch
from typing import Callable, Tuple
from physics.fokker_planck import fokker_planck_residual

class CompositeLoss:
    """
    Manages physics-informed loss components with adaptive weighting.
    
    Implements:
    - L_total = λ_pde * L_pde + λ_mass * L_mass + λ_bc * L_bc
    """
    
    def __init__(self,
                 A: torch.Tensor,
                 D: torch.Tensor,
                 domain_bounds: Tuple[float, float] = (-5.0, 5.0),
                 mc_samples: int = 100000,
                 bc_samples: int = 1000,
                 device: torch.device = None):
        """
        Initialize loss components.
        
        Args:
            A: Drift matrix (3x3)
            D: Diffusion matrix (3x3)
            domain_bounds: Spatial domain extents
            mc_samples: Monte Carlo samples for mass conservation
            bc_samples: Samples per boundary face
            device: Target device for computations
        """
        self.A = A
        self.D = D
        self.domain_bounds = domain_bounds
        self.mc_samples = mc_samples
        self.bc_samples = bc_samples
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Precompute mass conservation grid
        self._setup_mass_integration()
        self._setup_boundary_samples()

    def _setup_mass_integration(self):
        """Precompute integration constants for mass conservation."""
        # Generate fixed MC samples for consistent integration
        self.mc_points = torch.rand((self.mc_samples, 3), device=self.device)
        self.mc_points = self.mc_points * (self.domain_bounds[1] - self.domain_bounds[0]) + self.domain_bounds[0]
        
        # Cell volume for integration
        self.volume = (self.domain_bounds[1] - self.domain_bounds[0])**3
        self.density_coeff = self.volume / self.mc_samples

    def _setup_boundary_samples(self):
        """Precompute boundary sample templates."""
        self.boundary_templates = []
        for dim in range(3):
            for val in [self.domain_bounds[0], self.domain_bounds[1]]:
                # Generate template with fixed dimension
                points = torch.rand((self.bc_samples, 3), device=self.device)
                points = points * (self.domain_bounds[1] - self.domain_bounds[0]) + self.domain_bounds[0]
                points[:, dim] = val
                self.boundary_templates.append(points)

    def _mass_conservation(self, model: Callable, t: torch.Tensor) -> torch.Tensor:
        """Compute mass conservation loss ∫p(x,t)dx = 1."""
        # Evaluate model on precomputed MC points
        t_expanded = t[0].expand(self.mc_samples)  # Assume single time per batch
        p = model(self.mc_points, t_expanded)
        
        # Compute integral approximation
        integral = torch.sum(p) * self.density_coeff
        return (integral - 1.0)**2

    def _boundary_condition(self, model: Callable, t: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss p(∂Ω,t) ≈ 0."""
        losses = []
        t = t[0].expand(self.bc_samples)  # Assume single time per batch
        
        for boundary_points in self.boundary_templates:
            p = model(boundary_points, t)
            losses.append(torch.mean(p**2))
            
        return torch.mean(torch.stack(losses))

    def __call__(self, 
                model: Callable,
                x: torch.Tensor,
                t: torch.Tensor,
                lambda_pde: float = 1.0,
                lambda_mass: float = 10.0,
                lambda_bc: float = 0.1) -> Tuple[torch.Tensor, dict]:
        """
        Compute composite loss with current weights.
        
        Args:
            model: Neural solution model
            x: Spatial coordinates (batch_size, 3)
            t: Temporal coordinates (batch_size,)
            
        Returns:
            total_loss: Weighted sum of loss components
            loss_dict: Individual loss components
        """
        # Ensure tensor requirements for gradient computation
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        
        # Compute PDE residual loss
        residual = fokker_planck_residual(model, x, t, self.A, self.D)
        L_pde = torch.mean(residual**2)
        
        # Compute mass conservation loss
        L_mass = self._mass_conservation(model, t)
        
        # Compute boundary condition loss
        L_bc = self._boundary_condition(model, t)
        
        # Combine losses
        total_loss = (lambda_pde * L_pde + 
                     lambda_mass * L_mass + 
                     lambda_bc * L_bc)
        
        return total_loss, {
            "total": total_loss,
            "pde": L_pde,
            "mass": L_mass,
            "bc": L_bc
        }

