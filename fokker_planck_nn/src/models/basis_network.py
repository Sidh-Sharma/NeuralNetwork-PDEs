"""
Basis network implementation for low-rank decomposition of the solution.

This module defines the neural network architecture for the spatial basis functions
in the low-rank decomposition of the Fokker-Planck equation solution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union, Callable


class BasisNetwork(nn.Module):
    """
    Neural network representing a spatial basis function φᵣ(x).
    
    This network maps points in 3D space to scalar values, representing
    one component of the low-rank decomposition of the solution.
    """
    
    def __init__(self, 
                 input_dim: int = 3, 
                 hidden_dim: int = 64, 
                 num_layers: int = 3,
                 activation: str = 'gelu',
                 output_activation: str = 'sin',
                 init_method: str = 'xavier_normal'):
        """
        Initialize the basis network.
        
        Args:
            input_dim: Input dimension (default 3 for 3D space)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            init_method: Weight initialization method
        """
        super(BasisNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Output activation function
        if output_activation == 'sin':
            self.output_activation = torch.sin
        elif output_activation == 'tanh':
            self.output_activation = torch.tanh
        elif output_activation == 'none' or output_activation is None:
            self.output_activation = lambda x: x
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights(init_method)
    
    def _initialize_weights(self, method: str):
        """
        Initialize network weights.
        
        Args:
            method: Weight initialization method
        """
        for layer in self.layers:
            if method == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight)
            elif method == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                raise ValueError(f"Unsupported initialization method: {method}")
            
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        # Initialize output layer
        if method == 'xavier_normal':
            nn.init.xavier_normal_(self.output_layer.weight)
        elif method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.output_layer.weight)
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='linear')
        elif method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='linear')
        
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size)
        """
        # Forward pass through hidden layers
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            out = self.activation(out)
        
        # Forward pass through output layer
        out = self.output_layer(out)
        out = self.output_activation(out)
        
        # Return flattened output
        return out.squeeze(-1)


class MultiModeBasisNetwork(nn.Module):
    """
    Container for multiple basis networks representing multiple spatial modes.
    """
    
    def __init__(self, 
                 num_modes: int = 4,
                 input_dim: int = 3, 
                 hidden_dim: int = 64, 
                 num_layers: int = 3,
                 activation: str = 'gelu',
                 output_activation: str = 'sin',
                 init_method: str = 'xavier_normal',
                 orthogonalization: bool = False):
        """
        Initialize multiple basis networks.
        
        Args:
            num_modes: Number of basis modes to use
            input_dim: Input dimension (default 3 for 3D space)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            init_method: Weight initialization method
            orthogonalization: Whether to enforce orthogonality between modes
        """
        super(MultiModeBasisNetwork, self).__init__()
        
        self.num_modes = num_modes
        self.input_dim = input_dim
        self.orthogonalization = orthogonalization
        
        # Create multiple basis networks
        self.basis_networks = nn.ModuleList([
            BasisNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation,
                output_activation=output_activation,
                init_method=init_method
            )
            for _ in range(num_modes)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all basis networks.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
                
        Returns:
            Output tensor of shape (batch_size, num_modes)
        """
        # Evaluate each basis network
        outputs = []
        for basis_net in self.basis_networks:
            outputs.append(basis_net(x).unsqueeze(1))
        
        # Stack outputs along mode dimension
        modes = torch.cat(outputs, dim=1)
        
        # Optionally apply orthogonalization
        if self.orthogonalization and self.training:
            batch_size = x.shape[0]
            orthogonal_modes = []
            
            # Modified Gram-Schmidt process
            for r in range(self.num_modes):
                # Start with current mode
                v = modes[:, r].clone()
                
                # Subtract projections onto previous modes
                for j in range(r):
                    proj = torch.sum(v * orthogonal_modes[j], dim=0) / batch_size
                    v = v - proj * orthogonal_modes[j]
                
                # Normalize
                norm = torch.sqrt(torch.sum(v**2) / batch_size)
                if norm > 1e-8:
                    v = v / norm
                    
                orthogonal_modes.append(v)
            
            # Replace modes with orthogonalized versions
            modes = torch.stack(orthogonal_modes, dim=1)
        
        return modes
    
    def add_mode(self):
        """Add a new basis network with the same configuration."""
        self.num_modes += 1
        self.basis_networks.append(
            BasisNetwork(
                input_dim=self.input_dim,
                hidden_dim=self.basis_networks[0].hidden_dim,
                num_layers=len(self.basis_networks[0].layers),
                activation='gelu',  # Use same activation as others
                output_activation='sin' if self.basis_networks[0].output_activation == torch.sin else 'none',
                init_method='xavier_normal'
            )
        )


def initialize_from_pod(basis_network: MultiModeBasisNetwork, 
                       pod_modes: List[Callable], 
                       domain_bounds: Tuple[float, float], 
                       grid_size: int = 20,
                       device: torch.device = None):
    """
    Initialize basis networks from POD modes using least squares fitting.
    
    Args:
        basis_network: MultiModeBasisNetwork to initialize
        pod_modes: List of callable POD mode functions
        domain_bounds: Tuple (min_val, max_val) for the domain
        grid_size: Number of points per dimension for the grid
        device: PyTorch device
    """
    min_val, max_val = domain_bounds
    
    # Create grid for fitting
    linspace = torch.linspace(min_val, max_val, grid_size, device=device)
    X, Y, Z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    
    # Number of modes to initialize
    num_modes = min(len(pod_modes), basis_network.num_modes)
    
    # Initialize each basis network
    for i in range(num_modes):
        # Compute POD mode values on grid
        pod_values = pod_modes[i](grid_points)
        
        # Fit the basis network to this POD mode
        fit_network_to_function(basis_network.basis_networks[i], grid_points, pod_values)


def fit_network_to_function(network: BasisNetwork, 
                           points: torch.Tensor, 
                           values: torch.Tensor,
                           num_epochs: int = 1000,
                           lr: float = 1e-3,
                           verbose: bool = False):
    """
    Fit a single basis network to match given function values using least squares.
    
    Args:
        network: BasisNetwork to fit
        points: Input points of shape (num_points, input_dim)
        values: Target values of shape (num_points)
        num_epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print progress
    """
    # Use MSE loss
    criterion = nn.MSELoss()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = network(points)
        
        # Compute loss
        loss = criterion(outputs, values)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # Test the implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a basis network
    basis_net = BasisNetwork(input_dim=3, hidden_dim=64, num_layers=3)
    basis_net.to(device)
    
    # Create input
    x = torch.randn(10, 3, device=device)

        # Forward pass through single basis network
    output = basis_net(x)
    print("Single BasisNetwork output shape:", output.shape)
    
    # Create a multi-mode basis network
    multi_basis_net = MultiModeBasisNetwork(
        num_modes=4,
        input_dim=3,
        hidden_dim=64,
        num_layers=3,
        activation='gelu',
        output_activation='sin',
        init_method='xavier_normal',
        orthogonalization=True
    )
    multi_basis_net.to(device)
    
    # Forward pass through multi-mode basis network
    multi_output = multi_basis_net(x)
    print("MultiModeBasisNetwork output shape:", multi_output.shape)
    
    # Check orthogonality during training mode
    multi_basis_net.train()
    orthogonalized_output = multi_basis_net(x)
    print("Orthogonalized MultiModeBasisNetwork output shape:", orthogonalized_output.shape)
    
    # Quick check: print a few outputs
    print("Sample outputs (first sample):", orthogonalized_output[0])

    # Switch to eval mode and check outputs again
    multi_basis_net.eval()
    eval_output = multi_basis_net(x)
    print("Evaluation mode output shape:", eval_output.shape)
    print("Sample outputs (first sample) in eval mode:", eval_output[0])

    # Initialize from POD modes (dummy example)
    pod_modes = [lambda x: torch.sin(x[:, 0]) * torch.cos(x[:, 1]) * torch.sin(x[:, 2]) for _ in range(4)]
    initialize_from_pod(multi_basis_net, pod_modes, domain_bounds=(-1, 1), grid_size=10, device=device)
    print("Initialized MultiModeBasisNetwork from POD modes.")
    # Check the initialized output
    initialized_output = multi_basis_net(x)
    print("Initialized output shape:", initialized_output.shape)
    print("Sample initialized outputs (first sample):", initialized_output[0])
