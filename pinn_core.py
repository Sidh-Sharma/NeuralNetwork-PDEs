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
                 weight_f, weight_ic, weight_bc, print_every=500):

    x_f, t_f, x0, t0, u0, x_b_left, t_b, x_b_right = data
    loss_history = []

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
    
    return net, loss_history

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