import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Use GPU if available
device = torch.device("mps" if torch.mps.is_available() else "cpu")


def compute_error_metrics(u_true, u_pred):
    """Compute error metrics between the true and predicted solutions."""
    u_true_flat = u_true.flatten()
    u_pred_flat = u_pred.flatten()

    mse = mean_squared_error(u_true_flat, u_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(u_true_flat, u_pred_flat)
    r2 = r2_score(u_true_flat, u_pred_flat)
    rel_l2 = np.linalg.norm(u_true_flat - u_pred_flat) / np.linalg.norm(u_true_flat)
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
        "CV_RMSE": cv_rmse,
    }


def visualise_solution(net, x_min, x_max, t_min, t_max, nx=401, nt=501, true_solution=None, save_path=None):
    """Generate contour plots and snapshots for the PINN solution."""
    x = torch.linspace(x_min, x_max, nx).view(-1, 1).to(device)
    t = torch.linspace(t_min, t_max, nt).view(-1, 1).to(device)

    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    X_flat, T_flat = X.reshape(-1, 1), T.reshape(-1, 1)
    input_pts = torch.cat([X_flat, T_flat], dim=1)

    with torch.no_grad():
        u_pred = net(input_pts).cpu().numpy().reshape(nx, nt)

    # Contour plot of the predicted solution
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(T.cpu().numpy(), X.cpu().numpy(), u_pred, 50, cmap="viridis")
    plt.colorbar(contour, label="u(x,t)")
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Position (x)", fontsize=12)
    plt.title("PINN Solution", fontsize=14)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(f"{save_path}_contour.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot snapshots at different time steps
    t_snapshots = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(t_snapshots), figsize=(15, 5), sharex=True, sharey=True)

    for i, t_val in enumerate(t_snapshots):
        t_idx = int(t_val * (nt - 1))
        ax = axes[i]
        ax.plot(x.cpu().numpy(), u_pred[:, t_idx], "b-", linewidth=2, label="PINN")

        if true_solution is not None:
            ax.plot(x.cpu().numpy(), true_solution[:, t_idx], "r--", linewidth=2, label="True")

        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"t = {t_val:.2f}", fontsize=12)

        if i == 0:
            ax.set_ylabel("u(x,t)", fontsize=12)
            ax.legend(fontsize=10)

        ax.set_xlabel("x", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_snapshots.png", dpi=300, bbox_inches="tight")
    plt.close()


def compare_with_fdm_solution(net, problem, nx=401, nt=501, visualise=True, save_path=None):
    """Compare PINN solution with Finite Difference Method (FDM) solution."""
    print("Computing finite difference solution...")
    x_fdm, t_fdm, u_fdm = problem.finite_difference_solution(nx=nx, nt=nt)

    print("Generating PINN predictions on the FDM grid...")
    x_tensor = torch.tensor(x_fdm, dtype=torch.float32).view(-1, 1).to(device)

    u_pinn = np.zeros((nx, nt))
    for j in range(nt):
        t_val = t_fdm[j]
        t_tensor = torch.ones_like(x_tensor) * t_val
        inputs = torch.cat([x_tensor, t_tensor], dim=1)

        with torch.no_grad():
            u_pinn[:, j] = net(inputs).cpu().numpy().flatten()

    # Compute error metrics
    metrics = compute_error_metrics(u_fdm, u_pinn)

    print("\nComparison with Finite Difference Solution:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6e}")

    if visualise:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # FDM solution
        im0 = axes[0].contourf(t_fdm, x_fdm, u_fdm, 50, cmap="viridis")
        axes[0].set_title("FDM Solution", fontsize=14)
        axes[0].set_xlabel("t", fontsize=12)
        axes[0].set_ylabel("x", fontsize=12)
        plt.colorbar(im0, ax=axes[0], label="u(x,t)")

        # PINN solution
        im1 = axes[1].contourf(t_fdm, x_fdm, u_pinn, 50, cmap="viridis")
        axes[1].set_title("PINN Solution", fontsize=14)
        axes[1].set_xlabel("t", fontsize=12)
        axes[1].set_ylabel("x", fontsize=12)
        plt.colorbar(im1, ax=axes[1], label="u(x,t)")

        # Error distribution
        error = np.abs(u_fdm - u_pinn)
        im2 = axes[2].contourf(t_fdm, x_fdm, error, 50, cmap="inferno")
        axes[2].set_title("Absolute Error", fontsize=14)
        axes[2].set_xlabel("t", fontsize=12)
        axes[2].set_ylabel("x", fontsize=12)
        plt.colorbar(im2, ax=axes[2], label="|Error|")

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_solution_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    return metrics, u_fdm
