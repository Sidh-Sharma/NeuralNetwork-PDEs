import os
import torch
import numpy as np
import argparse
from datetime import datetime
from pinn_core import PINN, BurgersEquation, train_nn, evaluate_model
from pinn_gridsearch import grid_search
from pinn_visualise import visualise_solution, compare_with_fdm_solution

# Set device
device = torch.device("mps" if torch.mps.is_available() else "cpu")

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="PINN for Burgers' Equation")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "gridsearch", "visualise"],
                        help="Execution mode: train, gridsearch, or visualise")
    parser.add_argument("--model_path", type=str, help="Path to trained model for visualisation")
    parser.add_argument("--n_train", type=int, default=10000, help="Training data points")
    parser.add_argument("--n_val", type=int, default=2000, help="Validation data points")
    args = parser.parse_args()

    # Problem setup
    nu = 0.01 / np.pi
    problem = BurgersEquation(nu=nu)

    if args.mode == "train":
        print("Training PINN...")
        train_data = problem.generate_training_data(n_points=args.n_train)

        # Define model
        layers = [2, 16, 32, 32, 16, 1]
        net = PINN(layers, activation="gelu").to(device)

        # Train the model
        trained_net, loss_history = train_nn(
            net, train_data,
            adam_epochs=10000, adam_lr=5e-4,
            lbfgs_epochs=100, lbfgs_lr=0.5,
            weight_f=1.0, weight_ic=2.5, weight_bc=5.0,
            print_every=1000
        )

        # Save trained model
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f"saved_models/model_{timestamp}.pt"
        torch.save(trained_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Compare with FDM solution
        compare_with_fdm_solution(trained_net, problem, save_path=f"results/comparison_{timestamp}")
        visualise_solution(trained_net, x_min=-1, x_max=1, t_min=0, t_max=1, save_path=f"results/solution_{timestamp}")

    elif args.mode == "gridsearch":
        print("Running hyperparameter grid search...")

        # Generate training & validation data
        train_data = problem.generate_training_data(n_points=args.n_train)
        val_data = problem.generate_training_data(n_points=args.n_val)

        # Define hyperparameter grid
        hyperparams = {
            'architecture': [[2, 16, 32, 32, 16, 1], [2, 16, 32, 64, 32, 16, 1]],
            'weight_f': [1.0, 1.5, 2.0], 'weight_ic': [2.5, 5.0], 'weight_bc': [2.5, 5.0],
            'adam_lr': [4e-3, 5e-4], 'adam_epochs': [10000, 15000],
            'lbfgs_lr': [0.1], 'lbfgs_epochs': [500],
            'activation': ['gelu']
        }

        # Run grid search
        best_config, best_state, results = grid_search(train_data, val_data, hyperparams, problem)
        print(f"Best configuration: {best_config}")

        # Save best model and results
        torch.save(best_state, "saved_models/best_model_gridsearch.pt")
        results.to_csv("results/grid_search_results.csv", index=False)

    elif args.mode == "visualise":
        if not args.model_path:
            print("No model path provided. Training a new model with default parameters...")

            train_data = problem.generate_training_data(n_points=10000)

            # Define default model
            layers = [2, 16, 32, 16, 1]
            net = PINN(layers, activation="gelu").to(device)

            # Train the model with default settings
            trained_net, loss_history = train_nn(
                net, train_data, nu,
                adam_epochs=5000, adam_lr=5e-4,
                lbfgs_epochs=50, lbfgs_lr=0.5,
                weight_f=1.0, weight_ic=2.0, weight_bc=2.0
            )

            # Save the model temporarily
            model_path = "saved_models/temp_visualisation_model.pt"
            torch.save(trained_net.state_dict(), model_path)
            print(f"Temporary model trained and saved to {model_path}")

        else:
            print(f"Loading model from {args.model_path}...")
            model_path = args.model_path
            net = PINN([2, 16, 32, 32, 16, 1]).to(device)
            net.load_state_dict(torch.load(model_path))

        compare_with_fdm_solution(net, problem, save_path="results/visualisation")
        true_solution = problem.finite_difference_solution(nx=401, nt=501)[2]
        visualise_solution(net, x_min=-1, x_max=1, t_min=0, t_max=1, true_solution=true_solution, save_path="results/visualisation")



if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    main()
