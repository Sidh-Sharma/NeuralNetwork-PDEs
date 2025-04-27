"""
Temporal coefficient network implementation for low-rank decomposition.

This module defines the neural network architecture for the time-dependent coefficients
in the low-rank decomposition of the Fokker-Planck equation solution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

class TemporalNetwork(nn.Module):
    """
    Neural network representing time-dependent coefficients c_r(t).
    
    Maps scalar time values to R-dimensional coefficient vectors through:
    - Two hidden layers with tanh activation
    - Linear output layer for coefficient magnitudes
    
    Input shape: (batch_size, 1)
    Output shape: (batch_size, num_modes)
    """
    
    def __init__(self, 
                 num_modes: int = 4,
                 hidden_dim: int = 64, 
                 num_layers: int = 2,
                 activation: str = 'tanh',
                 init_method: str = 'xavier_normal'):
        """
        Initialize temporal network components.
        
        Args:
            num_modes: Number of coefficient outputs (R in decomposition)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function for hidden layers
            init_method: Weight initialization method
        """
        super().__init__()
        
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Validate activation choice
        valid_activations = ['tanh', 'gelu', 'relu']
        if activation not in valid_activations:
            raise ValueError(f"Invalid activation: {activation}. Choose from {valid_activations}")
            
        # Define activation function
        self.activation = getattr(nn, activation.title())()

        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer (time → hidden_dim)
        self.layers.append(nn.Linear(1, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output layer (hidden_dim → num_modes)
        self.output_layer = nn.Linear(hidden_dim, num_modes)
        
        # Initialize weights
        self._initialize_weights(init_method)
        
    def _initialize_weights(self, method: str):
        """
        Initialize network weights using specified method.
        
        Args:
            method: One of ['xavier_normal', 'xavier_uniform', 'kaiming_normal']
        """
        valid_methods = ['xavier_normal', 'xavier_uniform', 'kaiming_normal']
        if method not in valid_methods:
            raise ValueError(f"Invalid init method: {method}. Choose from {valid_methods}")

        for layer in self.layers:
            if method.startswith('xavier'):
                nn.init.xavier_normal_(layer.weight) if 'normal' in method else nn.init.xavier_uniform_(layer.weight)
            elif method.startswith('kaiming'):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='tanh')
            
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
                
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal network.
        
        Args:
            t: Input time tensor of shape (batch_size, 1)
            
        Returns:
            Coefficient tensor of shape (batch_size, num_modes)
        """
        out = t
        for layer in self.layers:
            out = layer(out)
            out = self.activation(out)
            
        # Final linear projection
        coefficients = self.output_layer(out)
        
        return coefficients


if __name__ == "__main__":
    # Test implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create network instance
    temp_net = TemporalNetwork(
        num_modes=4,
        hidden_dim=64,
        num_layers=2,
        activation='tanh',
        init_method='xavier_normal'
    ).to(device)
    
    # Test input
    t = torch.tensor([[0.0], [0.5], [1.0]], device=device)  # (3, 1)
    
    # Forward pass
    coefficients = temp_net(t)
    print("Temporal network test:")
    print(f"Input shape: {t.shape}")
    print(f"Output shape: {coefficients.shape}")
    print(f"Sample coefficients:\n{coefficients.detach().cpu().numpy()}")