import os
import torch
import pandas as pd
import numpy as np
from itertools import product
from pinn_core import PINN, BurgersEquation, train_nn, evaluate_model
from pinn_visualise import compare_with_fdm_solution, compute_error_metrics

# Use GPU if available
device = torch.device("mps" if torch.mps.is_available() else "cpu")


def grid_search(train_data, val_data, hyperparams, problem, save_dir="saved_models"):
    """
    Perform grid search over hyperparameters for PINN optimization.
    
    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        hyperparams: Dictionary of hyperparameter lists to search over.
        problem: BurgersEquation instance to generate reference solutions.
        save_dir: Directory to save model checkpoints.

    Returns:
        best_config: Identifier string of the best configuration.
        best_state: State dictionary of the best model.
        df_results: DataFrame containing all results.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_val_loss = np.inf
    best_config = None
    best_state = None
    best_metrics = None
    results = []

    # Generate a reference solution for error evaluation
    _, _, reference_solution = problem.finite_difference_solution(nx=201, nt=201)

    # Generate all hyperparameter combinations
    search_space = list(product(
        hyperparams["architecture"],
        hyperparams["weight_f"],
        hyperparams["weight_ic"],
        hyperparams["weight_bc"],
        hyperparams["adam_lr"],
        hyperparams["adam_epochs"],
        hyperparams["lbfgs_lr"],
        hyperparams["lbfgs_epochs"],
        hyperparams["activation"]
    ))

    total_combinations = len(search_space)
    print(f"Starting grid search over {total_combinations} configurations...")

    for i, (arch, wf, wic, wbc, adam_lr, adam_epochs, lbfgs_lr, lbfgs_epochs, activation) in enumerate(search_space):
        progress = (i + 1) / total_combinations * 100
        print(f"\n[{progress:.1f}%] Training configuration {i+1}/{total_combinations}")

        # Define model
        layers = arch
        model = PINN(layers, activation=activation).to(device)

        # Configuration identifier
        arch_str = "x".join(map(str, arch))
        config_id = (
            f"arch{arch_str}_wf{wf}_wic{wic}_wbc{wbc}_"
            f"adamlr{adam_lr}_adamep{adam_epochs}_"
            f"lbfgslr{lbfgs_lr}_lbfgsep{lbfgs_epochs}_act{activation}"
        )

        print(f"Training model: {config_id}")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # Train model
        start_time.record()
        net, loss_history = train_nn(
            model, train_data, problem.nu,
            adam_epochs=adam_epochs, adam_lr=adam_lr,
            lbfgs_epochs=lbfgs_epochs, lbfgs_lr=lbfgs_lr,
            weight_f=wf, weight_ic=wic, weight_bc=wbc
        )
        end_time.record()
        torch.cuda.synchronize()
        training_time = start_time.elapsed_time(end_time) / 1000  # Convert ms to seconds

        # Evaluate on validation set
        val_loss, val_loss_f, val_loss_ic, val_loss_bc = evaluate_model(net, val_data, problem.nu, wf, wic, wbc)

        # Compare with FDM solution
        error_metrics, _ = compare_with_fdm_solution(net, problem, visualize=False)

        # Store results
        current_metrics = {
            "val_loss": val_loss,
            "val_loss_f": val_loss_f,
            "val_loss_ic": val_loss_ic,
            "val_loss_bc": val_loss_bc,
            "training_time": training_time,
            **error_metrics
        }
        results.append({
            "architecture": arch_str,
            "weight_f": wf,
            "weight_ic": wic,
            "weight_bc": wbc,
            "adam_lr": adam_lr,
            "adam_epochs": adam_epochs,
            "lbfgs_lr": lbfgs_lr,
            "lbfgs_epochs": lbfgs_epochs,
            "activation": activation,
            **current_metrics
        })

        # Save partial results
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(os.path.join(save_dir, "grid_search_partial_results.csv"), index=False)

        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config_id
            best_state = net.state_dict()
            best_metrics = current_metrics

            # Save best model checkpoint
            torch.save({
                "model_state_dict": best_state,
                "config": {
                    "architecture": arch,
                    "activation": activation,
                    "training_params": {
                        "weight_f": wf, "weight_ic": wic, "weight_bc": wbc,
                        "adam_lr": adam_lr, "adam_epochs": adam_epochs,
                        "lbfgs_lr": lbfgs_lr, "lbfgs_epochs": lbfgs_epochs
                    }
                },
                "metrics": best_metrics
            }, os.path.join(save_dir, f"best_model_{config_id}.pt"))
            print(f"New best model found! Validation loss: {best_val_loss:.4e}")

    # Save final results
    df_results = pd.DataFrame(results)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    df_results.to_csv(os.path.join(save_dir, f"grid_search_results_{timestamp}.csv"), index=False)

    print("\nGrid search completed.")
    print(f"Best configuration: {best_config}")
    print(f"Best validation loss: {best_val_loss:.4e}")

    # Save summary of best model
    with open(os.path.join(save_dir, "best_model_summary.txt"), "w") as f:
        f.write(f"Best configuration: {best_config}\n")
        f.write(f"Best validation loss: {best_val_loss:.4e}\n")
        f.write("\nDetailed metrics:\n")
        for metric, value in best_metrics.items():
            f.write(f"  {metric}: {value:.6e}\n")

    return best_config, best_state, df_results
