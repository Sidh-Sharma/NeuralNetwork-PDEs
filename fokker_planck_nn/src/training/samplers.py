"""
Adaptive sampling strategies for 3D Fokker-Planck training (training/samplers.py)

Implements physics-aware spatiotemporal sampling with phase-dependent strategies.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from collections import deque

class AdaptiveSampler3D:
    """
    Manages adaptive spatial sampling strategies for 3D domain.
    
    Implements:
    - Uniform sampling (initial phase)
    - Residual-based adaptive sampling (mid phase)
    - Curvature-aware sampling (final phase)
    """
    
    def __init__(self, 
                 domain_bounds: Tuple[float, float] = (-5.0, 5.0),
                 initial_samples: int = 10000,
                 adaptive_buffer_size: int = 5000,
                 device: torch.device = None):
        """
        Initialize 3D spatial sampler.
        
        Args:
            domain_bounds: Spatial domain extents (same for all dimensions)
            initial_samples: Initial uniform sample pool size
            adaptive_buffer_size: Max stored adaptive samples
            device: Target device for generated samples
        """
        self.domain_bounds = domain_bounds
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sampling buffers
        self.uniform_pool = self._create_uniform_pool(initial_samples)
        self.adaptive_buffer = deque(maxlen=adaptive_buffer_size)
        
        # Phase-dependent parameters
        self.phase = "initial"
        self._phase_strategies = {
            "initial": self._initial_sampling,
            "mid": self._mid_phase_sampling,
            "final": self._final_phase_sampling
        }
    
    def _create_uniform_pool(self, n_samples: int) -> torch.Tensor:
        """Generate initial uniform samples across domain."""
        samples = torch.rand((n_samples, 3), device=self.device)
        return samples * (self.domain_bounds[1] - self.domain_bounds[0]) + self.domain_bounds[0]
    
    def update_residuals(self, 
                       residuals: torch.Tensor, 
                       samples: torch.Tensor) -> None:
        """
        Update adaptive buffer with high-residual samples.
        
        Args:
            residuals: PDE residuals from last batch (shape: [batch_size])
            samples: Corresponding spatial coordinates (shape: [batch_size, 3])
        """
        if self.phase == "initial":
            return  # No adaptation during initial phase
        
        # Select top 20% high-residual samples
        _, idx = torch.topk(residuals, k=int(0.2 * len(residuals)))
        self.adaptive_buffer.extend(samples[idx].cpu().unbind())
    
    def sample_spatial(self, 
                      batch_size: int, 
                      phase: Optional[str] = None) -> torch.Tensor:
        """
        Generate spatial samples according to current phase strategy.
        
        Returns:
            Spatial coordinates (batch_size, 3) on target device
        """
        strategy = self._phase_strategies[phase or self.phase]
        return strategy(batch_size)
    
    def _initial_sampling(self, batch_size: int) -> torch.Tensor:
        """Uniform sampling with boundary emphasis."""
        # 70% uniform, 30% near boundaries
        n_boundary = int(0.3 * batch_size)
        main_samples = self.uniform_pool[torch.randperm(len(self.uniform_pool))[:batch_size - n_boundary]]
        
        # Boundary samples (random face selection)
        boundary_samples = torch.zeros((n_boundary, 3), device=self.device)
        dims = torch.randint(0, 3, (n_boundary,))
        sides = torch.randint(0, 2, (n_boundary,)) * 2 - 1  # -1 or 1
        boundary_samples[torch.arange(n_boundary), dims] = sides * self.domain_bounds[1]
        
        return torch.cat([main_samples, boundary_samples])
    
    def _mid_phase_sampling(self, batch_size: int) -> torch.Tensor:
        """Hybrid uniform + adaptive sampling."""
        if len(self.adaptive_buffer) < 1000:  # Not enough adaptive samples yet
            return self._initial_sampling(batch_size)
        
        # 50% adaptive, 50% uniform
        n_adaptive = batch_size // 2
        adaptive_samples = torch.stack(np.random.sample(self.adaptive_buffer, n_adaptive)).to(self.device)
        
        uniform_samples = self.uniform_pool[torch.randperm(len(self.uniform_pool))[:batch_size - n_adaptive]]
        
        return torch.cat([adaptive_samples, uniform_samples])
    
    def _final_phase_sampling(self, batch_size: int) -> torch.Tensor:
        """Focus sampling near high-curvature regions."""
        # Current implementation uses adaptive buffer only
        # (In practice could add curvature estimation)
        if len(self.adaptive_buffer) < batch_size:
            return self._mid_phase_sampling(batch_size)
        
        return torch.stack(np.random.sample(self.adaptive_buffer, batch_size)).to(self.device)
    
    def set_phase(self, phase: str) -> None:
        """Update sampling phase."""
        valid_phases = ["initial", "mid", "final"]
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase: {phase}. Choose from {valid_phases}")
        self.phase = phase


class TimeSampler:
    """
    Manages temporal sampling strategies with phase adaptation.
    
    Implements:
    - Linear time sampling (initial phase)
    - Residual-weighted sampling (mid phase)
    - Steady-state focused sampling (final phase)
    """
    
    def __init__(self, 
                 t_range: Tuple[float, float] = (0.0, 2.0),
                 device: torch.device = None):
        """
        Initialize temporal sampler.
        
        Args:
            t_range: Time domain bounds
            device: Target device for generated samples
        """
        self.t_range = t_range
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phase = "initial"
        self.time_weights = None
    
    def sample_temporal(self, 
                       batch_size: int,
                       phase: Optional[str] = None) -> torch.Tensor:
        """
        Generate temporal samples according to current phase strategy.
        
        Returns:
            Time points (batch_size,) on target device
        """
        phase = phase or self.phase
        if phase == "initial":
            return self._linear_sampling(batch_size)
        elif phase == "mid":
            return self._weighted_sampling(batch_size)
        else:  # final phase
            return self._steady_state_sampling(batch_size)
    
    def _linear_sampling(self, batch_size: int) -> torch.Tensor:
        """Uniform time sampling across full range."""
        return torch.rand(batch_size, device=self.device) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
    
    def _weighted_sampling(self, batch_size: int) -> torch.Tensor:
        """Error-weighted time sampling."""
        if self.time_weights is None:
            return self._linear_sampling(batch_size)
        
        # Normalize weights
        probs = self.time_weights / self.time_weights.sum()
        indices = torch.multinomial(probs, batch_size, replacement=True)
        return self.time_points[indices]
    
    def _steady_state_sampling(self, batch_size: int) -> torch.Tensor:
        """Focus sampling near steady-state (t ≈ 2.0)."""
        base_t = torch.full((batch_size,), self.t_range[1], device=self.device)
        noise = torch.randn(batch_size, device=self.device) * 0.1  # N(2.0, 0.1)
        return torch.clamp(base_t + noise, *self.t_range)
    
    def update_time_weights(self, 
                          time_points: torch.Tensor,
                          residuals: torch.Tensor) -> None:
        """
        Update temporal sampling weights based on residuals.
        
        Args:
            time_points: Sampled time points (shape: [N])
            residuals: Corresponding PDE residuals (shape: [N])
        """
        # Bin residuals by time
        time_bins = torch.linspace(*self.t_range, 50)
        binned_residuals = torch.zeros_like(time_bins)
        
        for i, t in enumerate(time_bins):
            mask = (torch.abs(time_points - t) < 0.05)
            if mask.any():
                binned_residuals[i] = residuals[mask].mean()
        
        # Exponential smoothing
        if self.time_weights is None:
            self.time_weights = binned_residuals
        else:
            self.time_weights = 0.3 * binned_residuals + 0.7 * self.time_weights
        
        self.time_points = time_bins


if __name__ == "__main__":
    # Test sampler functionality
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Spatial sampler tests
    spatial_sampler = AdaptiveSampler3D(device=device)
    print("Initial spatial samples:", spatial_sampler.sample_spatial(1000).shape)
    
    # Simulate residual update
    fake_residuals = torch.rand(1000, device=device)
    fake_samples = spatial_sampler.sample_spatial(1000)
    spatial_sampler.update_residuals(fake_residuals, fake_samples)
    
    # Temporal sampler tests
    time_sampler = TimeSampler(device=device)
    print("Initial time samples:", time_sampler.sample_temporal(1000).shape)
    
    # Test time weight update
    fake_time_points = torch.linspace(0, 2, 1000, device=device)
    fake_time_residuals = torch.exp(-fake_time_points)  # Higher errors at early times
    time_sampler.update_time_weights(fake_time_points, fake_time_residuals)
    print("Weighted time samples:", time_sampler.sample_temporal(1000, "mid").shape)