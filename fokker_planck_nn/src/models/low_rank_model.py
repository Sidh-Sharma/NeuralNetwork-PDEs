"""
Low-rank decomposition model combining spatial and temporal networks.
"""

import torch
import torch.nn as nn
from typing import Tuple
from .basis_network import MultiModeBasisNetwork
from .temporal_network import TemporalNetwork

class LowRankModel(nn.Module):
    """
    Implements the low-rank decomposition p(x,t) = Σ c_r(t)φ_r(x)
    with automatic derivative computation for physics constraints.
    
    Architecture:
    - Spatial basis networks: MultiModeBasisNetwork (multiple BasisNetwork instances)
    - Temporal coefficient network: TemporalNetwork
    """

    def __init__(self,
                 num_modes: int = 4,
                 spatial_hidden: int = 64,
                 temporal_hidden: int = 64,
                 spatial_activation: str = 'gelu',
                 temporal_activation: str = 'tanh'):
        """
        Initialize low-rank model components.
        
        Args:
            num_modes: Number of modes in decomposition (R)
            spatial_hidden: Hidden dimension for spatial networks
            temporal_hidden: Hidden dimension for temporal network
            spatial_activation: Activation for spatial networks
            temporal_activation: Activation for temporal network
        """
        super().__init__()
        
        # Spatial basis function networks
        self.basis_net = MultiModeBasisNetwork(
            num_modes=num_modes,
            input_dim=3,
            hidden_dim=spatial_hidden,
            num_layers=3,
            activation=spatial_activation,
            output_activation='sin',  # Periodic boundary benefits
            orthogonalization=True
        )
        
        # Temporal coefficient network
        self.temporal_net = TemporalNetwork(
            num_modes=num_modes,
            hidden_dim=temporal_hidden,
            num_layers=2,
            activation=temporal_activation
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability density p(x,t) = Σ c_r(t)φ_r(x)
        
        Args:
            x: Spatial coordinates (batch_size, 3)
            t: Temporal coordinates (batch_size, 1) or (batch_size,)
            
        Returns:
            p: Probability density values (batch_size)
        """
        # Ensure proper tensor shapes
        t = t.view(-1, 1)  # Force (batch_size, 1)
        
        # Get spatial basis functions (batch_size, num_modes)
        phi = self.basis_net(x)
        
        # Get temporal coefficients (batch_size, num_modes)
        c = self.temporal_net(t)
        
        # Combine modes (batch_size)
        return torch.sum(c * phi, dim=1)

    def compute_derivatives(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute derivatives required for PDE residual calculation.
        """
        # Enable gradient tracking with fresh tensors
        x = x.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)
        
        # Forward pass
        p = self(x, t)
        
        # Compute time derivative
        (dp_dt,) = torch.autograd.grad(
            outputs=p,
            inputs=t,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True  # Keep graph for subsequent gradients
        )
        
        # Compute spatial gradient
        (grad_p,) = torch.autograd.grad(
            outputs=p,
            inputs=x,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True
        )
        
        # Compute Laplacian
        laplacian_p = torch.zeros_like(p)
        for i in range(3):
            (d2p_dxi2,) = torch.autograd.grad(
                outputs=grad_p[:, i],
                inputs=x,
                grad_outputs=torch.ones_like(grad_p[:, i]),
                create_graph=True,
                retain_graph=True if i < 2 else False  # Release graph after last
            )
            laplacian_p += d2p_dxi2[:, i]

        return p, dp_dt, grad_p, laplacian_p


if __name__ == "__main__":
    # Test implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model instance
    model = LowRankModel(
        num_modes=4,
        spatial_hidden=64,
        temporal_hidden=64
    ).to(device)
    
    # Test inputs
    batch_size = 5
    x = torch.randn(batch_size, 3, device=device)
    t = torch.rand(batch_size, 1, device=device)
    
    # Forward pass test
    p = model(x, t)
    print("\nForward pass test:")
    print(f"Input x shape: {x.shape}")
    print(f"Input t shape: {t.shape}")
    print(f"Output p shape: {p.shape}")
    print(f"Sample outputs: {p[:2].detach().cpu().numpy()}")
    
    # Derivative computation test
    print("\nDerivative computation test:")
    p, dp_dt, grad_p, laplacian_p = model.compute_derivatives(x, t)
    print(f"dp_dt shape: {dp_dt.shape}")
    print(f"grad_p shape: {grad_p.shape}")
    print(f"laplacian_p shape: {laplacian_p.shape}")
    print(f"Max laplacian value: {laplacian_p.abs().max().item():.4f}")