"""
Training orchestration for Fokker-Planck neural solver (training/trainer.py)

Manages the complete training lifecycle including phase transitions, mode growth, and validation.
"""

import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from .samplers import AdaptiveSampler3D, TimeSampler
from .losses import CompositeLoss
from utils.metrics import relative_l2_error, gaussian_kl_divergence, max_pointwise_error
from utils.visualization import plot_training_history

logger = logging.getLogger(__name__)

class FPTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        A: torch.Tensor,
        D: torch.Tensor,
        domain_bounds: Tuple[float, float] = (-5.0, 5.0),
        config: Optional[Dict] = None
    ):
        """
        Initialize the Fokker-Planck equation trainer.
        
        Args:
            model: Low-rank decomposition model
            A: Drift matrix
            D: Diffusion matrix
            domain_bounds: Spatial domain bounds
            config: Training configuration dictionary
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.config = self._default_config()
        if config:
            self.config.update(config)

        # Initialize components
        self.sampler = AdaptiveSampler3D(domain_bounds, device=self.device)
        self.time_sampler = TimeSampler(t_range=(0.0, 2.0), device=self.device)
        self.loss_fn = CompositeLoss(A, D, domain_bounds, device=self.device)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config["restart_cycle"],
            eta_min=self.config["min_lr"]
        )

        # Training state
        self.epoch = 0
        self.best_loss = float("inf")
        self.train_history = []
        self.val_history = []
        self.metric_history = []

        # Create output directory
        self.save_dir = Path(self.config["save_dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _default_config(self) -> Dict:
        """Return default training configuration."""
        return {
            "max_epochs": 5000,
            "batch_size": 4096,
            "lr": 3e-4,
            "min_lr": 1e-6,
            "weight_decay": 1e-6,
            "grad_clip": 1.0,
            "phase_epochs": {"initial": 500, "mid": 3000, "final": 1500},
            "restart_cycle": 500,
            "val_interval": 100,
            "save_interval": 500,
            "patience": 200,
            "save_dir": "./checkpoints"
        }

    def train(self):
        """Execute full training lifecycle."""
        try:
            for phase in ["initial", "mid", "final"]:
                self._transition_phase(phase)
                
                for epoch in tqdm(range(self.config["phase_epochs"][phase]), desc=f"{phase.capitalize()} Phase"):
                    self.epoch += 1
                    train_loss = self._train_epoch()
                    
                    # Validation and checkpointing
                    if self.epoch % self.config["val_interval"] == 0:
                        val_metrics = self.validate()
                        self._update_metrics(val_metrics)
                        self._checkpoint()
                        self._handle_mode_growth(val_metrics)
                        
                    # Update learning rate
                    self.scheduler.step()

        except KeyboardInterrupt:
            logger.info("Training interrupted. Saving current state...")
            self._checkpoint(final=True)

    def _train_epoch(self) -> float:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        phase = self._current_phase()

        for _ in range(self.config["batch_per_epoch"]):
            # Sample batch
            x = self.sampler.sample_spatial(self.config["batch_size"], phase)
            t = self.time_sampler.sample_temporal(self.config["batch_size"], phase)
            
            # Forward pass and loss calculation
            self.optimizer.zero_grad()
            loss, loss_dict = self.loss_fn(self.model, x, t, **self._phase_loss_weights(phase))
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.optimizer.step()
            
            total_loss += loss.item()

        return total_loss / self.config["batch_per_epoch"]

    def validate(self) -> Dict:
        """Evaluate model on validation dataset."""
        self.model.eval()
        val_data = self._load_validation_data()
        metrics = {"l2": [], "kl": [], "max_err": []}

        with torch.no_grad():
            for t_idx, t in enumerate(val_data["times"]):
                # Forward pass
                p_pred = self.model(val_data["grid"], t.expand(val_data["grid"].shape[0]))
                p_true = val_data["solutions"][t_idx]
                
                # Get predicted moments
                mean_pred, cov_pred = self._model_moments(p_pred, val_data["grid"])
                
                # Get true moments - correctly access the tuple elements
                mean_true = val_data["moments"][0][t_idx]  # First element is means tensor
                cov_true = val_data["moments"][1][t_idx]   # Second element is covariances tensor
                
                # Compute metrics
                metrics["l2"].append(relative_l2_error(p_pred, p_true))
                metrics["kl"].append(gaussian_kl_divergence(mean_pred, cov_pred, mean_true, cov_true))
                metrics["max_err"].append(max_pointwise_error(p_pred, p_true))

        return {k: np.mean(v) for k, v in metrics.items()}

    def _transition_phase(self, phase: str):
        """Handle phase transition logic."""
        logger.info(f"Transitioning to {phase} phase")
        self.sampler.set_phase(phase)
        self.time_sampler.phase = phase
        self._adjust_learning_rate(phase)

    def _handle_mode_growth(self, metrics: Dict):
        """Add new modes if validation performance plateaus."""
        if len(self.metric_history) < self.config["patience"]:
            return

        recent_metrics = self.metric_history[-self.config["patience"] :]
        improvements = [recent_metrics[i]["l2"] - recent_metrics[i+1]["l2"] 
                       for i in range(len(recent_metrics)-1)]
        
        if all(imp < 0.01 for imp in improvements):
            logger.info("Validation plateau detected. Adding new mode.")
            self.model.add_mode()
            self._reinitialize_optimizer()

    def _checkpoint(self, final: bool = False):
        """Save training state."""
        checkpoint = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": self.metric_history,
            "config": self.config
        }
        
        fname = "final.pt" if final else f"checkpoint_epoch{self.epoch}.pt"
        torch.save(checkpoint, self.save_dir / fname)
        
        if self.epoch % self.config["save_interval"] == 0:
            plot_training_history(self.train_history, self.val_history)

    def _adjust_learning_rate(self, phase: str):
        """Phase-specific learning rate adjustments."""
        if phase == "final":
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= 0.1

    def _reinitialize_optimizer(self):
        """Reinitialize optimizer after architecture changes."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )

    def _phase_loss_weights(self, phase: str) -> Dict:
        """Get loss weights for current phase."""
        return {
            "initial": {"lambda_pde": 1.0, "lambda_mass": 0.1, "lambda_bc": 0.0},
            "mid": {"lambda_pde": 1.0, "lambda_mass": 10.0, "lambda_bc": 0.1},
            "final": {"lambda_pde": 0.5, "lambda_mass": 50.0, "lambda_bc": 0.5}
        }[phase]

    def _current_phase(self) -> str:
        """Determine current training phase."""
        total_epochs = sum(self.config["phase_epochs"].values())
        if self.epoch < self.config["phase_epochs"]["initial"]:
            return "initial"
        elif self.epoch < total_epochs - self.config["phase_epochs"]["final"]:
            return "mid"
        return "final"

    def _load_validation_data(self) -> Dict:
        """Load or generate validation dataset."""
        if not hasattr(self, '_val_data_cache'):
            # Import analytical data utilities
            from data.analytical_data import load_analytical_data, create_validation_dataset, generate_grid_points
            
            # Look for saved validation data first - note the correct path
            validation_file = Path("data/validation.pt")
            
            if validation_file.exists():
                logger.info(f"Loading validation data from {validation_file}")
                self._val_data_cache = load_analytical_data(validation_file, self.device)
            else:
                logger.warning("Validation data not found at data/validation.pt - generating on-the-fly")
                # Generate grid based on config
                grid = generate_grid_points(
                    bounds=(-5.0, 5.0),
                    resolution=self.config.get("grid_resolution", 50),
                    device=self.device
                )
                
                # Create validation dataset
                self._val_data_cache = create_validation_dataset(
                    grid=grid,
                    time_points=self.config.get("time_points", 20),
                    t_max=2.0,
                    device=self.device
                )
                
                # Save for future use
                logger.info("Saving generated validation data for future use")
                Path("data").mkdir(exist_ok=True)
                torch.save(self._val_data_cache, validation_file)
        
        return self._val_data_cache

    def _model_moments(self, p: torch.Tensor, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute model moments from predicted distribution."""
        # Normalize probability density
        p_sum = torch.sum(p)
        if p_sum > 0:  # Avoid division by zero
            p_norm = p / p_sum
        else:
            logger.warning("Zero probability sum encountered in _model_moments")
            p_norm = torch.ones_like(p) / p.shape[0]
        
        # Compute mean
        mean = torch.sum(grid * p_norm.unsqueeze(1), dim=0)
        
        # Compute covariance
        centered = grid - mean
        cov = torch.zeros((3, 3), device=self.device)
        
        for i in range(grid.shape[0]):
            # Outer product for each point, weighted by probability
            outer = centered[i].unsqueeze(1) @ centered[i].unsqueeze(0)
            cov += p_norm[i] * outer
        
        return mean, cov

    def _update_metrics(self, metrics: Dict) -> None:
        """Update and track validation metrics history."""
        # Store the full metrics in history
        self.metric_history.append(metrics)
        
        # Update specific metrics for plotting
        self.val_history.append(metrics["l2"])  # Track L2 error for plotting
        
        # Log the validation results
        logger.info(f"Epoch {self.epoch} validation metrics: " +
                    f"L2={metrics['l2']:.6f}, KL={metrics['kl']:.6f}, " +
                    f"MaxErr={metrics['max_err']:.6f}")

if __name__ == "__main__":
    # Example usage
    from models.low_rank_model import LowRankModel
    from physics.ou_process import create_default_ou_params

    # Initialize model and parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu0, sigma0, A, D = create_default_ou_params(device)
    model = LowRankModel(num_modes=4).to(device)

    # Configure trainer
    config = {
        "max_epochs": 5000,
        "batch_size": 8192,
        "val_interval": 100
    }
    
    trainer = FPTrainer(model, A, D, config=config)
    trainer.train()