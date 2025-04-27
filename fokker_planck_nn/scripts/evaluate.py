"""
Model evaluation script with quantitative metrics and visualizations (scripts/evaluate.py)
"""

import argparse
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict

# Local imports
from src.models.low_rank_model import LowRankModel
from src.data.analytical_data import load_analytical_data
from src.utils.metrics import (
    relative_l2_error,
    gaussian_kl_divergence,
    max_pointwise_error,
    moments_from_samples
)
from src.utils.visualization import plot_error_comparison, create_animation

def configure_logging(output_dir: Path) -> logging.Logger:
    """Set up logging system"""
    logger = logging.getLogger('FP-Evaluation')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(output_dir / 'evaluation.log')
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def load_model(checkpoint_path: str, device: torch.device) -> LowRankModel:
    """Load trained model with architecture parameters from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Reconstruct model with saved parameters
        model = LowRankModel(
            num_modes=checkpoint['config']['model']['num_modes'],
            spatial_hidden=checkpoint['config']['model']['spatial_hidden'],
            temporal_hidden=checkpoint['config']['model']['temporal_hidden']
        ).to(device)
        
        model.load_state_dict(checkpoint['model'])
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Checkpoint file not found: {checkpoint_path}")
    except KeyError as e:
        raise RuntimeError(f"Missing key in checkpoint: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def compute_metrics(model: LowRankModel, val_data: Dict) -> Dict:
    """Calculate validation metrics across all time points"""
    metrics = {
        'relative_l2': [],
        'kl_divergence': [],
        'max_error': []
    }
    
    model.eval()
    with torch.no_grad():
        for t_idx, t in enumerate(val_data['times']):
            # Prepare inputs
            grid = val_data['grid']
            t_expanded = t.expand(grid.shape[0])
            
            # Compute predictions
            p_pred = model(grid, t_expanded)
            
            # Create samples tensor [N, 4] = [x, y, z, p]
            samples = torch.cat([grid, p_pred.unsqueeze(1)], dim=1)
            
            # Calculate moments
            mean_pred, cov_pred = moments_from_samples(samples)
            mean_true = val_data['moments'][0][t_idx]
            cov_true = val_data['moments'][1][t_idx]
            
            # Store metrics
            metrics['relative_l2'].append(relative_l2_error(p_pred, val_data['solutions'][t_idx]))
            metrics['kl_divergence'].append(gaussian_kl_divergence(
                mean_pred, cov_pred, mean_true, cov_true))
            metrics['max_error'].append(max_pointwise_error(p_pred, val_data['solutions'][t_idx]))
    
    return {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in metrics.items()}

def save_results(model: LowRankModel, val_data: Dict, metrics: Dict, output_dir: Path):
    """Save metrics and generate visualizations"""
    # Save numerical metrics
    with (output_dir / 'metrics.json').open('w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate error comparison plot
    with torch.no_grad():
        final_pred = model(val_data['grid'], val_data['times'][-1].expand(val_data['grid'].shape[0]))
        plot_error_comparison(
            p_pred=final_pred,
            p_true=val_data['solutions'][-1],
            grid=val_data['grid']
        ).savefig(output_dir / 'final_error_comparison.png')
        
        # Create solution animation
        predictions = [model(val_data['grid'], t.expand(val_data['grid'].shape[0])) 
                     for t in val_data['times']]
        create_animation(
            predictions,
            val_data['grid']
        ).save(output_dir / 'solution_evolution.mp4')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--data', required=True, help='Validation dataset path')
    parser.add_argument('--output', default='results', help='Output directory')
    args = parser.parse_args()
    
    # Setup environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    
    try:
        # Load model and data
        model = load_model(args.checkpoint, device)
        val_data = load_analytical_data(args.data, device)
        
        # Compute metrics
        logger.info("Computing validation metrics...")
        metrics = compute_metrics(model, val_data)
        
        # Save results
        logger.info("Generating output files...")
        save_results(model, val_data, metrics, output_dir)
        
        logger.info(f"Evaluation complete. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
