import torch
from torch.utils.data import DataLoader
from model import CoefficientNet, generate_basis, H, dH, d2H, TestFunctions

# Hyperparameters 
max_degree = 3  # Total degree for basis functions
num_test = 20    # Number of test functions
lambda_norm = 10
lambda_pos = 1.0
batch_size = 1024
T_max = 1.0      # Time domain [0, T_max]

#initialisation
basis_indices = generate_basis(max_degree)
num_basis = len(basis_indices)
coeff_net = CoefficientNet(output_dim=num_basis)
test_funcs = TestFunctions(num_test=num_test)
optimizer = torch.optim.Adam(coeff_net.parameters())


for epoch in range(50000):
    # Sample time and space
    t = torch.rand(batch_size, 1, requires_grad=True) * T_max  # [batch, 1]
    x = torch.randn(batch_size, 3)         # [batch, 3]
    
    # Compute basis functions and FP terms
    phi = []
    fp_terms = []
    
    for (n, m, k) in basis_indices:
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        
        # Basis function
        Hn = H(n, x1)
        Hm = H(m, x2)
        Hk = H(k, x3)
        phi_i = Hn * Hm * Hk
        phi.append(phi_i)
        
        # Fokker-Planck operator terms
        d2Hn = d2H(n, x1)
        d2Hm = d2H(m, x2)
        d2Hk = d2H(k, x3)
        term_laplacian = d2Hn*Hm*Hk + Hn*d2Hm*Hk + Hn*Hm*d2Hk
        
        dHn = dH(n, x1)
        dHm = dH(m, x2)
        dHk = dH(k, x3)
        term_xdotgrad = x1*dHn*Hm*Hk + x2*Hn*dHm*Hk + x3*Hn*Hm*dHk
        
        term_total = term_laplacian + term_xdotgrad + 3 * phi_i
        fp_terms.append(term_total)
    
    phi = torch.stack(phi, dim=1)          # [batch, num_basis]
    fp_terms = torch.stack(fp_terms, dim=1)  # [batch, num_basis]
    
    # Compute coefficients and predicted density
    c = coeff_net(t)                       # [batch, num_basis]
    p = (c * phi).sum(dim=1)               # [batch]
    
    # Compute dp/dt using autograd
    t.requires_grad_(True)
    dp_dt = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True, allow_unused=True)[0]
    
    # Compute Fokker-Planck residual
    L_p = (c * fp_terms).sum(dim=1)        # [batch]
    residual = dp_dt.squeeze() - L_p
    
    # Weak residual loss (project onto test functions)
    psi = test_funcs.evaluate(x)           # [batch, num_test]
    L_weak = (psi * residual.unsqueeze(1)).mean(dim=0).pow(2).sum()
    
    # Physical constraints
    L_norm = (p.mean() - 1.0)**2           # Normalization
    L_pos = torch.relu(-p).mean()           # Positivity
    
    total_loss = L_weak + lambda_norm * L_norm + lambda_pos * L_pos
    
    # Optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Checkpointing
    if epoch % 5000 == 0:
        torch.save(coeff_net.state_dict(), f"checkpoint_epoch{epoch}.pt")
        print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")