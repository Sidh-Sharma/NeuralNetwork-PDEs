import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import seaborn as sns

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Use GPU if available
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Problem parameters
nu = 0.01 / np.pi  # viscosity
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0

class BurgersEquation:
    
    def __init__(self, nu=0.01/np.pi, x_domain=(-1.0, 1.0), t_domain=(0.0, 1.0)):
        self.nu = nu
        self.x_min, self.x_max = x_domain
        self.t_min, self.t_max = t_domain
    
    def generate_training_data(self, n_points=10000):
        x_f = torch.FloatTensor(n_points, 1).uniform_(self.x_min, self.x_max).to(device)
        t_f = torch.FloatTensor(n_points, 1).uniform_(self.t_min, self.t_max).to(device)
        
        # Initial condition points: u(x,0) = -sin(πx)
        n0 = n_points // 10
        x0 = torch.linspace(self.x_min, self.x_max, n0).view(-1, 1).to(device)
        t0 = torch.zeros_like(x0).to(device)
        u0 = -torch.sin(np.pi * x0)
        
        # Boundary condition points: u(-1,t) = u(1,t) = 0
        nb = n_points // 10
        t_b = torch.linspace(self.t_min, self.t_max, nb).view(-1, 1).to(device)
        x_b_left = self.x_min * torch.ones_like(t_b).to(device)
        x_b_right = self.x_max * torch.ones_like(t_b).to(device)
        
        return x_f, t_f, x0, t0, u0, x_b_left, t_b, x_b_right
    
    def finite_difference_solution(self, nx=401, nt=501):
        
        x = np.linspace(self.x_min, self.x_max, nx)
        t = np.linspace(self.t_min, self.t_max, nt)
        dx = (self.x_max - self.x_min) / (nx - 1)
        dt = (self.t_max - self.t_min) / (nt - 1)
        
        u = np.zeros((nx, nt))
        
        u[:, 0] = -np.sin(np.pi * x)
        u[0, :] = 0  # u(-1,t) = 0
        u[-1, :] = 0  # u(1,t) = 0
        
        # Use Crank-Nicolson scheme for better stability
        r = self.nu * dt / (dx**2)
        
        alpha = -r / 2
        beta = 1 + r
        
        # Time-stepping solution
        for n in range(0, nt-1):
            
            u_n = u[:, n].copy()
            
            b = np.zeros(nx)
            b[1:-1] = u_n[1:-1] + 0.5 * dt * (
                self.nu * (u_n[2:] - 2*u_n[1:-1] + u_n[0:-2]) / dx**2 - 
                u_n[1:-1] * (u_n[2:] - u_n[0:-2]) / (2*dx)
            )
            
            b[0] = 0
            b[-1] = 0
            
            A = np.zeros((nx, nx))
            
            for i in range(1, nx-1):
                A[i, i-1] = alpha - dt * u_n[i] / (4*dx)  
                A[i, i] = beta                           
                A[i, i+1] = alpha + dt * u_n[i] / (4*dx) 
            
            A[0, 0] = 1
            A[-1, -1] = 1
            

            u[:, n+1] = np.linalg.solve(A, b)
        
        return x, t, u

class PINN(nn.Module):
    
    def __init__(self, layers, activation='gelu'):

        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        num_layers = len(layers)
        
        for i in range(num_layers - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)
            
        self.num_layers = num_layers
        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if i == len(self.layers) - 2:
                x = torch.tanh(x)  # Last hidden layer always uses tanh
            elif self.activation == 'gelu':
                x = torch.nn.functional.gelu(x)
            else:
                x = torch.tanh(x)
                
        return self.layers[-1](x)

def compute_derivatives(net, x, t):
    X = torch.cat([x, t], dim=1).requires_grad_(True)
    u = net(X)
    
    # First derivatives using autograd
    grad_u = autograd.grad(u, X, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    
    # Second derivative u_xx
    u_xx = autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    
    return u, u_x, u_t, u_xx

def loss_function(net, x_f, t_f, x0, t0, u0, x_b_left, t_b, x_b_right,
                  nu, weight_f=1.0, weight_ic=5.0, weight_bc=5.0):
    x_f_temp = x_f.detach().requires_grad_(True)
    t_f_temp = t_f.detach().requires_grad_(True)
    
    # PDE residual loss
    u, u_x, u_t, u_xx = compute_derivatives(net, x_f_temp, t_f_temp)
    f_pred = u_t + u * u_x - nu * u_xx  # Burgers equation residual
    loss_f = torch.mean(f_pred ** 2)
    
    # Initial condition loss
    u_pred_0 = net(torch.cat([x0, t0], dim=1))
    loss_ic = torch.mean((u_pred_0 - u0) ** 2)
    
    # Boundary conditions loss
    u_pred_left = net(torch.cat([x_b_left, t_b], dim=1))
    u_pred_right = net(torch.cat([x_b_right, t_b], dim=1))
    loss_bc = torch.mean(u_pred_left ** 2) + torch.mean(u_pred_right ** 2)
    
    # Total weighted loss
    total_loss = weight_f * loss_f + weight_ic * loss_ic + weight_bc * loss_bc
    return total_loss, loss_f, loss_ic, loss_bc

def train_nn(net, data, adam_epochs, adam_lr, lbfgs_epochs, lbfgs_lr,
                 weight_f, weight_ic, weight_bc, print_every=500, early_stopping=True, patience=1000):

    x_f, t_f, x0, t0, u0, x_b_left, t_b, x_b_right = data
    loss_history = []
    best_loss = float('inf')
    no_improvement_count = 0
    start_time = datetime.now()
    
    # Phase 1: Adam optimizer
    print("Phase 1: Training with Adam optimizer")
    optimizer_adam = optim.Adam(net.parameters(), lr=adam_lr)
    
    adam_start_time = time.time()
    for epoch in range(adam_epochs):
        optimizer_adam.zero_grad()
        loss, loss_f, loss_ic, loss_bc = loss_function(
            net, x_f, t_f, x0, t0, u0,
            x_b_left, t_b, x_b_right, nu,
            weight_f, weight_ic, weight_bc
        )
        loss.backward()
        optimizer_adam.step()
        loss_history.append(loss.item())
        
        if epoch % print_every == 0 or epoch == adam_epochs:
            elapsed = datetime.now() - start_time
            print(f'Adam Epoch {epoch:05d} | Time: {elapsed} | '
                  f'Loss: {loss.item():.4e} | '
                  f'PDE: {loss_f.item():.4e} | '
                  f'IC: {loss_ic.item():.4e} | '
                  f'BC: {loss_bc.item():.4e}')
    
    adam_time = time.time() - adam_start_time
    adam_transition = len(loss_history)
    
    # Phase 2: L-BFGS optimizer
    print("\nPhase 2: Refinement with L-BFGS optimizer")
    optimizer_lbfgs = optim.LBFGS(net.parameters(),
                                  lr=lbfgs_lr,
                                  max_iter=20,
                                  max_eval=25,
                                  history_size=50,
                                  line_search_fn='strong_wolfe')
    
    lbfgs_start_time = time.time()
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, _, _, _ = loss_function(
            net, x_f, t_f, x0, t0, u0,
            x_b_left, t_b, x_b_right, nu,
            weight_f, weight_ic, weight_bc
        )
        loss.backward()
        return loss
    
    for epoch in range(lbfgs_epochs):
        loss = optimizer_lbfgs.step(closure)
        loss_history.append(loss.item())
        if epoch % print_every == 0 or epoch == lbfgs_epochs - 1:
            print(f'L-BFGS Epoch {epoch:04d} | Loss: {loss.item():.4e}')
    
    lbfgs_time = time.time() - lbfgs_start_time
    total_time = time.time() - adam_start_time
    
    training_metrics = {
        "adam_time": adam_time,
        "lbfgs_time": lbfgs_time,
        "total_time": total_time,
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history)
    }
    
    return net, loss_history, training_metrics

def evaluate_model(net, data, weight_f, weight_ic, weight_bc):
    
    net.eval()
    with torch.enable_grad():
        loss, loss_f, loss_ic, loss_bc = loss_function(
            net, *data, nu, weight_f, weight_ic, weight_bc
        )
    net.train()
    return loss.item(), loss_f.item(), loss_ic.item(), loss_bc.item()

def predict_solution(net, x, t):

    net.eval()
    X = torch.cat([x, t], dim=1)
    with torch.no_grad():
        u_pred = net(X)
    return u_pred

def compute_error_metrics(u_true, u_pred):
    u_true_flat = u_true.flatten()
    u_pred_flat = u_pred.flatten()
    
    # Calculate standard metrics
    mse = mean_squared_error(u_true_flat, u_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(u_true_flat, u_pred_flat)
    r2 = r2_score(u_true_flat, u_pred_flat)
    
    # Relative L2 error
    rel_l2 = np.linalg.norm(u_true_flat - u_pred_flat) / np.linalg.norm(u_true_flat)
    
    # Max absolute error
    max_error = np.max(np.abs(u_true_flat - u_pred_flat))

    epsilon = 1e-10
    cv_rmse = rmse / np.mean(np.abs(u_true_flat) + epsilon)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Relative L2": rel_l2,
        "Max Error": max_error,
        "CV_RMSE": cv_rmse
    }

def visualize_solution(net, nx=401, nt=501, true_solution=None, save_path=None):

    x = torch.linspace(x_min, x_max, nx).view(-1, 1).to(device)
    t = torch.linspace(t_min, t_max, nt).view(-1, 1).to(device)
    
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    input_pts = torch.cat([X_flat, T_flat], dim=1)
    
    with torch.no_grad():
        u_pred = net(input_pts).cpu().numpy()
    u_grid = u_pred.reshape(nx, nt)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(T.cpu().numpy(), X.cpu().numpy(), u_grid, 50, cmap='viridis')
    plt.colorbar(contour, label='u(x,t)')
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Position (x)', fontsize=12)
    plt.title(f'PINN Solution to Burgers Equation (ν={nu:.4f})', fontsize=14)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}_contour.png", dpi=300, bbox_inches='tight')
    plt.close()

    t_snapshots = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(t_snapshots), figsize=(15, 5), sharex=True, sharey=True)
    
    for i, t_val in enumerate(t_snapshots):
        t_idx = int(t_val * (nt - 1))
        ax = axes[i]
        
        ax.plot(x.cpu().numpy(), u_grid[:, t_idx], 'b-', linewidth=2, label='PINN')
        
        if true_solution is not None:
            ax.plot(x.cpu().numpy(), true_solution[:, t_idx], 'r--', linewidth=2, label='True')
        
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)
        ax.set_title(f't = {t_val:.2f}', fontsize=12)
        
        if i == 0:
            ax.set_ylabel('u(x,t)', fontsize=12)
            ax.legend(fontsize=10)
            
        ax.set_xlabel('x', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_snapshots.png", dpi=300, bbox_inches='tight')
    plt.close()

    if true_solution is not None:
        error = np.abs(u_grid - true_solution)
        
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(T.cpu().numpy(), X.cpu().numpy(), error, 50, cmap='inferno')
        plt.colorbar(contour, label='Absolute Error |u_true - u_pred|')
        plt.xlabel('Time (t)', fontsize=12)
        plt.ylabel('Position (x)', fontsize=12)
        plt.title('Error Distribution', fontsize=14)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(f"{save_path}_error.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(error.flatten(), bins=50, color='darkred', alpha=0.7)
        plt.xlabel('Absolute Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Error Distribution Histogram', fontsize=14)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(f"{save_path}_error_hist.png", dpi=300, bbox_inches='tight')
        plt.close()

def compare_with_fdm_solution(net, problem, nx=401, nt=501, visualize=True, save_path=None):
    print("Computing finite difference solution...")
    x_fdm, t_fdm, u_fdm = problem.finite_difference_solution(nx=nx, nt=nt)

    print("Generating PINN predictions on the FDM grid...")
    x_tensor = torch.tensor(x_fdm, dtype=torch.float32).view(-1, 1).to(device)
    
    u_pinn = np.zeros((nx, nt))
    for j in range(nt):
        t_val = t_fdm[j]
        t_tensor = torch.ones_like(x_tensor) * t_val
        
        # Predict in batches to avoid memory issues
        inputs = torch.cat([x_tensor, t_tensor], dim=1)
        with torch.no_grad():
            u_pinn[:, j] = net(inputs).cpu().numpy().flatten()
    

    metrics = compute_error_metrics(u_fdm, u_pinn)

    print("\nComparison with Finite Difference Solution:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6e}")
    
    if visualize:

        t_snapshots = [0.0, 0.25, 0.5, 0.75, 1.0]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, t_val in enumerate(t_snapshots):
            t_idx = int(t_val * (nt - 1))
            ax = axes[i]
            
            ax.plot(x_fdm, u_fdm[:, t_idx], 'r-', label='FDM', linewidth=2)
            ax.plot(x_fdm, u_pinn[:, t_idx], 'b--', label='PINN', linewidth=2)
            
            ax.set_ylim([-1.1, 1.1])
            ax.grid(True, alpha=0.3)
            ax.set_title(f't = {t_val:.2f}', fontsize=12)
            
            if i == 0:
                ax.set_ylabel('u(x,t)', fontsize=12)
                ax.legend(fontsize=10)
                
            ax.set_xlabel('x', fontsize=12)
        
        ax = axes[5]
        t_idx_mid = int(0.5 * (nt - 1))  # Error at t=0.5
        ax.plot(x_fdm, np.abs(u_fdm[:, t_idx_mid] - u_pinn[:, t_idx_mid]), 'k-', linewidth=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title('Error at t=0.5', fontsize=12)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('|Error|', fontsize=12)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_fdm_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # FDM solution
        im0 = axes[0].contourf(t_fdm, x_fdm, u_fdm, 50, cmap='viridis')
        axes[0].set_title('FDM Solution', fontsize=14)
        axes[0].set_xlabel('t', fontsize=12)
        axes[0].set_ylabel('x', fontsize=12)
        plt.colorbar(im0, ax=axes[0], label='u(x,t)')
        
        # PINN solution
        im1 = axes[1].contourf(t_fdm, x_fdm, u_pinn, 50, cmap='viridis')
        axes[1].set_title('PINN Solution', fontsize=14)
        axes[1].set_xlabel('t', fontsize=12)
        axes[1].set_ylabel('x', fontsize=12)
        plt.colorbar(im1, ax=axes[1], label='u(x,t)')
        
        # Error
        error = np.abs(u_fdm - u_pinn)
        im2 = axes[2].contourf(t_fdm, x_fdm, error, 50, cmap='inferno')
        axes[2].set_title('Absolute Error', fontsize=14)
        axes[2].set_xlabel('t', fontsize=12)
        axes[2].set_ylabel('x', fontsize=12)
        plt.colorbar(im2, ax=axes[2], label='|Error|')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_solution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return metrics, u_fdm

def grid_search(train_data, val_data, hyperparams, problem, save_dir='saved_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_val_loss = np.inf
    best_config = None
    best_state = None
    best_metrics = None

    results = []
    
    # Generate a reference solution for consistent benchmarking
    _, _, reference_solution = problem.finite_difference_solution(nx=201, nt=201)
    
    for (arch, wf, wic, wbc, adam_lr, adam_epochs, lbfgs_lr, lbfgs_epochs, activation) in product(
            hyperparams['architecture'],
            hyperparams['weight_f'],
            hyperparams['weight_ic'],
            hyperparams['weight_bc'],
            hyperparams['adam_lr'],
            hyperparams['adam_epochs'],
            hyperparams['lbfgs_lr'],
            hyperparams['lbfgs_epochs'],
            hyperparams['activation']):

        layers = arch
        model = PINN(layers, activation=activation).to(device)
        
        # Create config identifier
        arch_str = 'x'.join(map(str, arch))
        config_id = (f"arch{arch_str}_wf{wf}_wic{wic}_wbc{wbc}_"
                     f"adamlr{adam_lr}_adamep{adam_epochs}_"
                     f"lbfgslr{lbfgs_lr}_lbfgsepochs{lbfgs_epochs}_act{activation}")

        print(f"\nTraining configuration: {config_id}")

        # Train the model
        net, loss_history, training_metrics = train_nn(
            model, train_data,
            adam_epochs=adam_epochs,
            adam_lr=adam_lr,
            lbfgs_epochs=lbfgs_epochs,
            lbfgs_lr=lbfgs_lr,
            weight_f=wf,
            weight_ic=wic,
            weight_bc=wbc,
            print_every=1000  
        )

        # Evaluate validation loss
        val_loss, val_loss_f, val_loss_ic, val_loss_bc = evaluate_model(
            net, val_data, wf, wic, wbc
        )

        error_metrics, _ = compare_with_fdm_solution(net, problem, visualize=False)

        current_metrics = {
            "val_loss": val_loss,
            "val_loss_f": val_loss_f,
            "val_loss_ic": val_loss_ic,
            "val_loss_bc": val_loss_bc,
            **error_metrics,
            **training_metrics
        }

        results.append({
            "architecture": arch,
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

        if current_metrics['val_loss'] < best_val_loss:
            best_val_loss = current_metrics['val_loss']
            best_config = config_id
            best_state = net.state_dict()
            best_metrics = current_metrics

            torch.save(best_state, os.path.join(save_dir, f"best_model_{config_id}.pt"))

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(save_dir, "grid_search_result_2.csv"), index=False)

    print(f"\nBest configuration: {best_config}")
    print(f"Best validation loss: {best_val_loss:.4e}")

    return best_config, best_state, df_results




def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    problem = BurgersEquation(nu=nu)
    
    train_data = problem.generate_training_data(n_points=10000)
    val_data = problem.generate_training_data(n_points=2000)  # Validation set
    
    import argparse
    parser = argparse.ArgumentParser(description="PINN for Burgers Equation")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "gridsearch", "visualize"],
                        help="Execution mode: train, gridsearch, or visualize")
    parser.add_argument("--model_path", type=str, 
                        help="Path to model for visualization mode")
    args = parser.parse_args()

    if args.mode == "train":

        layers = [2, 16, 32, 16, 1]  
        net = PINN(layers, activation='gelu').to(device)

        trained_net, loss_history, metrics = train_nn(
            net, train_data,
            adam_epochs=10000,
            adam_lr=5e-4,
            lbfgs_epochs=100,
            lbfgs_lr=0.5,
            weight_f=1.0,
            weight_ic=2.0,
            weight_bc=2.0,
            print_every=1000
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f"saved_models/model_{timestamp}.pt"
        torch.save(trained_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        metrics, fdm_solution = compare_with_fdm_solution(
            trained_net, problem, 
            save_path=f"results/comparison_{timestamp}"
        )
        visualize_solution(
            trained_net, 
            true_solution=fdm_solution,
            save_path=f"results/solution_{timestamp}"
        )

    elif args.mode == "gridsearch":

        hyperparams = {
            'architecture': [
                [2, 16, 32, 32, 16, 1],  
                [2, 16, 32, 64, 32, 16, 1]  
                # [2, 16, 32, 16, 1],  
            ],
            'weight_f': [1.0],
            'weight_ic': [2.5],
            'weight_bc': [5.0],
            'adam_lr': [5e-4],
            'adam_epochs': [15000],
            'lbfgs_lr': [0.1],
            'lbfgs_epochs': [500],
            'activation': ['gelu']
        }
        
        best_config, best_state, results = grid_search(
            train_data, val_data, hyperparams, problem
        )
        print("\nGrid search completed. Best configuration:")
        print(best_config)

        torch.save(best_state, "saved_models/best_model_gridsearch.pt")
        results.to_csv("results/grid_search_results.csv", index=False)

    elif args.mode == "visualize":
        if not args.model_path:
            raise ValueError("Must provide --model_path for visualization mode")

        net = PINN([2, 16, 32, 64, 32, 16, 1]).to(device)
        net.load_state_dict(torch.load(args.model_path))

        metrics, fdm_solution = compare_with_fdm_solution(
            net, problem, 
            save_path="results/visualization"
        )
        visualize_solution(
            net, 
            true_solution=fdm_solution,
            save_path="results/visualization"
        )

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=True)
    np.set_printoptions(precision=4, suppress=True)
    
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {time.time()-start_time:.2f} seconds")