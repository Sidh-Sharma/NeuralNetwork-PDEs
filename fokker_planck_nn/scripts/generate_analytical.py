"""
Precompute analytical solutions for validation
"""

import argparse
import torch
from pathlib import Path

# Local imports
from data.analytical_data import (
    generate_grid_points,
    create_validation_dataset,
    save_analytical_data
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    parser.add_argument('--resolution', type=int, default=50,
                      help='Spatial resolution per dimension')
    parser.add_argument('--time_points', type=int, default=20,
                      help='Number of temporal snapshots')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                      help='Force computation device')
    args = parser.parse_args()
    
    # Device setup
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate grid
    grid = generate_grid_points(
        bounds=(-5, 5),
        resolution=args.resolution,
        device=device
    )
    
    # Create validation dataset
    dataset = create_validation_dataset(
        grid=grid,
        time_points=args.time_points,
        t_max=2.0,
        device=device
    )
    
    # Save results
    save_analytical_data(dataset, args.output)
    print(f"Saved validation data to {args.output}")

if __name__ == "__main__":
    main()