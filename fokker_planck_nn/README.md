# Theoretical Plan

---

#  Neural Solution of 3D OU Fokker–Planck Equation with Basis Learning

---

## 1. Problem Statement

We aim to **numerically solve** the **Fokker–Planck equation** for the **3D Ornstein–Uhlenbeck (OU) process**, which models the evolution of a probability density \(p(x, t)\).

The PDE is:

$$
\frac{\partial p}{\partial t} = \nabla \cdot (A x p) + \nabla \cdot (D \nabla p)
$$

where:

- \(x \in \mathbb{R}^3\),
- \(A\) is a \(3 \times 3\) **drift** matrix (typically negative definite),
- \(D\) is a \(3 \times 3\) **diffusion** (positive semi-definite) matrix.

This equation describes how the probability distribution "flows" and "spreads" over time.

---

## 2. Expected Output

- Neural network(s) that approximate the full solution \(p(x, t)\) **over a finite domain** (say \(x_i \in [-5,5]\)) and time \(t \in [0, T]\).
- We want to learn a **low-rank decomposition**:

$$
p(x, t) \approx \sum_{r=1}^R c_r(t) \, \phi_r(x)
$$

where:

- \(\phi_r(x)\) are learned **basis functions** (spatial networks),
- \(c_r(t)\) are learned **time-dependent coefficients** (temporal network).

**Goal**: Learn both **basis** and **coefficients** in a **physics-informed** manner (by minimizing the PDE residuals).

---

## 3. Analytical Solution (for Validation)

For the 3D OU process:

- The solution remains a **Gaussian** at all times.
- If the initial \(p(x, 0)\) is Gaussian with mean \(\mu_0\) and covariance \(\Sigma_0\), then:

$$
p(x,t) = \frac{1}{(2\pi)^{3/2} \sqrt{\det(\Sigma(t))}} \exp\left( -\frac{1}{2}(x - \mu(t))^T \Sigma(t)^{-1} (x - \mu(t)) \right)
$$

where:

- \(\mu(t) = e^{At} \mu_0\),
- \(\Sigma(t)\) satisfies the matrix ODE:

$$
\frac{d\Sigma}{dt} = A\Sigma + \Sigma A^T + 2D
$$

which is a **Lyapunov equation**.

Thus, we have a closed-form expression for \(p(x,t)\) for any \(t\), which we can use for **validation**.

---

# 4. Full Implementation Plan

---

## A. Architecture Design

| Component                     | Design                                                |
| ----------------------------- | ----------------------------------------------------- |
| Spatial Network \(\phi_r(x)\) | Small MLPs, \(x \in \mathbb{R}^3 \mapsto \mathbb{R}\) |
| Temporal Network \(c(t)\)     | 1D MLP, \(t \mapsto \mathbb{R}^R\)                    |
| Support Network (Optional)    | (x,t) -> scalar indicating region of support          |

---

**Spatial Network (basis network)**

```plaintext
Input: x ∈ ℝ³
Layers:
- Dense(64) + gelu
- Dense(64) + gelu
- Dense(64) + sin
- Dense(1)  (linear output)
```

**Temporal Network (coefficient network)**

```plaintext
Input: t ∈ ℝ
Layers:
- Dense(64) + tanh
- Dense(64) + tanh
- Dense(R)  (linear output)
```

---

## B. Initializations

- **Spatial Networks**:

  - Small random weights (Xavier normal initialization).
  - If available, **POD modes** from snapshots used to warm-start (use SVD on snapshots of true solution).

- **Temporal Network**:

  - Xavier normal initialization.

- **Support Mask**:

  - Train a small separate network to predict "region of significant \(p(x,t)\)".
  - Or simply threshold \(p(x,t)\) from true solution if available.

---

## C. Loss Formulations

### (1) PDE Residual Loss (Strong Form)

At random sample points \((x,t)\):

$$
\mathcal{L}_{\text{PDE}} = \mathbb{E}_{x,t} \left| \frac{\partial p_\theta}{\partial t} - \nabla \cdot (A x p_\theta) - \nabla \cdot (D \nabla p_\theta) \right|^2
$$

where \(p_\theta(x,t) = \sum_{r=1}^R c_r(t) \phi_r(x)\).

- Use automatic differentiation for \(\partial p / \partial t\) and \(\nabla p\).

---

### (2) Mass Conservation Loss

Ensure:

$$
\int_{\mathbb{R}^3} p(x,t) \, dx = 1
\quad \forall t
$$

Approximate by Monte Carlo integration:

$$
\mathcal{L}_{\text{mass}} = \mathbb{E}_t \left( \int p_\theta(x,t) \, dx - 1 \right)^2
$$

---

### (3) Boundary Condition Loss

If needed, enforce decay near domain edges:

$$
\mathcal{L}_{\text{boundary}} = \mathbb{E}_{x \in \partial \Omega, t} \left| p(x,t) \right|^2
$$

(Though for OU, support decays naturally.)

---

### (4) Total Loss

$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{PDE}} + \lambda_2 \mathcal{L}_{\text{mass}} + \lambda_3 \mathcal{L}_{\text{boundary}}
$$

with \(\lambda_i\) as hyperparameters (for now start with all 1).

---

## D. Optimizers and Training Strategy

| Phase                   | Optimizer | Learning Rate                     | Notes                                   |
| ----------------------- | --------- | --------------------------------- | --------------------------------------- |
| Initial Basis Training  | Adam      | 1e-3                              | Until residuals plateau                 |
| Incremental Mode Growth | Adam      | 1e-3 → decay after every new mode |                                         |
| Final Fine-tuning       | AdamW     | 5e-4                              | Reduce LR slowly for better convergence |

**Additional**:

- Optional LR scheduler: ReduceLROnPlateau (patience = 20 epochs).
- Weight decay for regularization: \(1e-5\).

---

## E. Metrics for Evaluation

At validation times \(t = t_1, t_2, ..., t_N\):

- **Relative ************************************\(L^2\)************************************ Error**:

$$
\text{Relative } L^2 = \frac{\| p_{\text{true}} - p_{\text{pred}} \|_2}{\| p_{\text{true}} \|_2}
$$

- **Wasserstein Distance** (later)
- **KL Divergence**:

$$
D_{\text{KL}}(p_{\text{true}} \| p_{\text{pred}})
$$

- **Maximum Pointwise Error**:

$$
\max_{x} |p_{\text{true}}(x,t) - p_{\text{pred}}(x,t)|
$$

---

## F. Visualizations

For a few \(t\):

- Plot **side-by-side 2D slices** (say at fixed \(x_3=0\)):
  - Analytical \(p(x,t)\) vs Predicted \(p(x,t)\).
- Plot **contours** or **heatmaps**.
- Plot **error heatmaps** \(|p_{\text{true}} - p_{\text{pred}}|\).



---

# Exact Implementation Steps

---

1. **Preprocessing**:

   - Define domain bounds and time interval.
   - Precompute analytical solution snapshots (for support and validation).
   -  SVD on snapshots for POD initial basis.

2. **Phase 1: Support Creation**:

   - Train a tiny network or threshold analytical \(p(x,t)\).

3. **Phase 2: Basis Initialization**:

   - Train initial \(R\) spatial networks and the temporal network together.
   - Physics-informed training: minimize total loss.

4. **Phase 3: Mode Growth**:

   - If validation error above tolerance, add 1–2 new basis networks.
   - Expand temporal network output layer.
   - Retrain.

5. **Phase 4: Fine-Tuning**:

   - Fix number of modes.
   - Fine-tune all networks jointly with a smaller learning rate.

6. **Evaluation**:

   - Compute all metrics at selected times.
   - Visualize side-by-side slices.

7. **Save models, results, plots**.

---



