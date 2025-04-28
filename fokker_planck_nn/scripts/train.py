"""
Main training script for 3D Fokker-Planck neural solver
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime

# Local imports
from src.models.low_rank_model import LowRankModel
from src.physics.ou_process import create_default_ou_params
from src.training.trainer import FPTrainer

def configure_logging(save_dir: Path) -> logging.Logger:
    """Setup unified logging system"""
    logger = logging.getLogger('FP-Training')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(save_dir / 'training.log')
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def parse_config(config_path: str) -> dict:
    """Load and validate configuration file"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required = ['model', 'training', 'validation']
    for section in required:
        if section not in config:
            raise ValueError(f"Config missing required section: {section}")
    
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to YAML config file')
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume training')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                      help='Override device selection')
    args = parser.parse_args()
    
    # Load configuration
    config = parse_config(args.config)
    
    # Device setup
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    save_dir = Path(config['training']['save_dir']) / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging
    logger = configure_logging(save_dir)
    
    # Model setup
    model = LowRankModel(
        num_modes=config['model']['num_modes'],
        spatial_hidden=config['model']['spatial_hidden'],
        temporal_hidden=config['model']['temporal_hidden']
    ).to(device)
    
    # Load physics parameters
    _,_, A, D = create_default_ou_params(device)
    
    # Initialize trainer
    trainer = FPTrainer(model, A, D, config=config['training'])
    
    # Resume training if specified
    if args.resume:
        try:
            checkpoint = torch.load(args.resume)
            trainer.load_checkpoint(checkpoint)
            logger.info(f"Resumed training from {args.resume}")
        except FileNotFoundError:
            logger.error(f"Checkpoint not found: {args.resume}")
            return
    
    # Start training
    logger.info("Commencing training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted - saving final state")
        trainer.save_checkpoint(final=True)

if __name__ == "__main__":
    main()