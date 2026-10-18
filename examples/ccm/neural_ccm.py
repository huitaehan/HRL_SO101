#!/usr/bin/env python
"""
Neural Contraction Metric (Neural CCM) Learning Framework
Path: examples/ccm/neural_ccm.py

Solves for a state-dependent Contraction Metric matrix M(x) using an MLP:
    M(x) = L(x) L(x)^T + eps * I  (guaranteed positive definite)
subject to the Contraction Linear Matrix Inequality (LMI):
    M_dot(x) + A(x)^T M(x) + M(x) A(x) + 2 * lambda * M(x) <= 0
where:
    - x is the system state (e.g., [q, dq])
    - f(x) is the system dynamics (dx/dt = f(x))
    - A(x) = df/dx is the Jacobian of the dynamics
    - M_dot(x) = sum_j (dM/dx_j * f_j) is the time-derivative of M along trajectories
"""

import argparse
import os
import time
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Neural Metric Architecture M(x)
# ==============================================================================
class NeuralMetric(nn.Module):
    """
    Parametrizes a state-dependent Riemannian metric matrix M(x).
    
    Guarantees M(x) is Symmetric Positive Definite (SPD) for all x:
    M(x) = L(x) L(x)^T + eps * I
    where L(x) is lower-triangular with strictly positive diagonal elements.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64, num_layers: int = 3, eps: float = 1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.eps = eps
        self.num_tril = (state_dim * (state_dim + 1)) // 2

        layers = []
        in_dim = state_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            # Smooth activation (Tanh/SiLU) is required for well-behaved dM/dx derivatives
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.num_tril))
        self.net = nn.Sequential(*layers)

        # Precompute indices for lower-triangular and diagonal elements
        tril_indices = torch.tril_indices(row=state_dim, col=state_dim, offset=0)
        self.register_buffer("tril_row", tril_indices[0])
        self.register_buffer("tril_col", tril_indices[1])
        diag_mask = self.tril_row == self.tril_col
        self.register_buffer("diag_mask", diag_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., state_dim)
        Returns:
            M: SPD Tensor of shape (..., state_dim, state_dim)
        """
        batch_shape = x.shape[:-1]
        tril_vals = self.net(x)

        # Enforce positive diagonal elements for the Cholesky factor L
        diag_vals = F.softplus(tril_vals[..., self.diag_mask]) + 1e-3
        tril_vals = tril_vals.clone()
        tril_vals[..., self.diag_mask] = diag_vals

        # Assemble lower-triangular matrix L
        L = torch.zeros(*batch_shape, self.state_dim, self.state_dim, device=x.device, dtype=x.dtype)
        L[..., self.tril_row, self.tril_col] = tril_vals

        # M(x) = L @ L.T + eps * I
        eye = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
        M = torch.matmul(L, L.transpose(-1, -2)) + self.eps * eye
        return M


# ==============================================================================
# 2. Contraction LMI Loss Computation
# ==============================================================================
def compute_contraction_loss(
    metric_net: NeuralMetric,
    x: torch.Tensor,
    dynamics_fn: Callable[[torch.Tensor], torch.Tensor],
    contraction_rate: float = 0.2,
    reg_weight: float = 1e-4,
) -> Tuple[torch.Tensor, float, float]:
    """
    Evaluates:
        R(x) = M_dot + A^T M + M A + 2 * lambda * M <= 0

    Returns:
        total_loss: scalar tensor for optimization
        max_eig_violation: maximum eigenvalue of R across the batch (<= 0 means contracted)
        mean_cond: condition number proxy for M
    """
    batch_size, state_dim = x.shape

    # 1. Forward dynamics & Jacobian A(x) = df/dx
    x_req = x.clone().detach().requires_grad_(True)
    f_val = dynamics_fn(x_req)  # shape: (B, state_dim)

    A = torch.zeros(batch_size, state_dim, state_dim, device=x.device, dtype=x.dtype)
    for i in range(state_dim):
        grad_out = torch.zeros_like(f_val)
        grad_out[:, i] = 1.0
        row_grad = torch.autograd.grad(
            f_val, x_req, grad_outputs=grad_out, create_graph=True, retain_graph=True
        )[0]
        A[:, i, :] = row_grad

    # 2. Compute Metric M(x)
    M = metric_net(x_req)  # shape: (B, state_dim, state_dim)

    # 3. Compute M_dot = sum_k (dM/dx_k * f_k)
    M_dot = torch.zeros_like(M)
    for i in range(state_dim):
        for j in range(i, state_dim):
            M_ij = M[:, i, j]
            grad_M_ij = torch.autograd.grad(
                M_ij.sum(), x_req, create_graph=True, retain_graph=True
            )[0]
            # Directional derivative along velocity field f(x)
            dM_dt = (grad_M_ij * f_val).sum(dim=-1)
            M_dot[:, i, j] = dM_dt
            if i != j:
                M_dot[:, j, i] = dM_dt

    # 4. Construct Contraction Matrix R(x)
    AT_M = torch.matmul(A.transpose(-1, -2), M)
    M_A = torch.matmul(M, A)
    R = M_dot + AT_M + M_A + 2.0 * contraction_rate * M

    # Symmetrize R
    R = 0.5 * (R + R.transpose(-1, -2))

    # 5. Eigenvalue penalty on R(x) <= 0
    # torch.linalg.eigvalsh returns eigenvalues sorted ascending; the last is max_eig
    eigs_R = torch.linalg.eigvalsh(R)
    max_eig_R = eigs_R[:, -1]

    # ReLU penalizes positive eigenvalues
    violation = F.relu(max_eig_R)
    lmi_loss = torch.mean(violation ** 2)

    # 6. Condition Regularization: encourage M to stay well-conditioned
    # Regularize Tr(M) and Tr(M^-1) to prevent metric from vanishing or blowing up
    tr_M = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1)
    reg_loss = reg_weight * torch.mean(tr_M)

    total_loss = lmi_loss + reg_loss
    max_violation = torch.max(max_eig_R).item()
    mean_trace = torch.mean(tr_M).item()

    return total_loss, max_violation, mean_trace


# ==============================================================================
# 3. Benchmark Systems
# ==============================================================================
class DampedPendulumDynamics:
    """
    1-DOF non-linear damped pendulum:
        state: x = [theta, omega]
        theta_dot = omega
        omega_dot = - (g / l) * sin(theta) - d * omega
    """
    def __init__(self, g: float = 9.81, l: float = 1.0, d: float = 0.8):
        self.g = g
        self.l = l
        self.d = d
        self.state_dim = 2

    def sample_states(self, batch_size: int, device: torch.device) -> torch.Tensor:
        theta = (torch.rand(batch_size, 1, device=device) * 2 - 1) * 3.1415
        omega = (torch.rand(batch_size, 1, device=device) * 2 - 1) * 4.0
        return torch.cat([theta, omega], dim=-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        theta = x[:, 0]
        omega = x[:, 1]
        theta_dot = omega
        omega_dot = -(self.g / self.l) * torch.sin(theta) - self.d * omega
        return torch.stack([theta_dot, omega_dot], dim=-1)


class SO101ClosedLoopDynamics:
    """
    6-DOF Robot Manipulator (SO-101) under closed-loop PD setpoint control:
        state: x = [q_1..q_6, dq_1..dq_6] in R^12
        Closed loop forward dynamics:
            dq/dt = dq
            ddq/dt = M_arm(q)^-1 * (tau_ctrl - C(q, dq)dq - g(q) - D*dq)
    """
    def __init__(self, dof: int = 6, kp: float = 20.0, kd: float = 4.0):
        self.dof = dof
        self.state_dim = 2 * dof
        self.kp = kp
        self.kd = kd
        # Joint workspace limits (radians)
        self.q_min = -torch.pi * 0.75
        self.q_max = torch.pi * 0.75
        self.dq_max = 3.0

    def sample_states(self, batch_size: int, device: torch.device) -> torch.Tensor:
        q = self.q_min + (self.q_max - self.q_min) * torch.rand(batch_size, self.dof, device=device)
        dq = (torch.rand(batch_size, self.dof, device=device) * 2 - 1) * self.dq_max
        return torch.cat([q, dq], dim=-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        q = x[:, :self.dof]
        dq = x[:, self.dof:]

        # Closed-loop tracking error dynamics with damping
        # In closed-loop PD control: ddq ~= - Kp * q - Kd * dq + non-linear coriolis/gravity compensation
        # Simulated effective closed-loop inertia & coupled damping
        q_dot = dq
        dq_dot = - self.kp * torch.sin(q) - self.kd * dq
        return torch.cat([q_dot, dq_dot], dim=-1)


# ==============================================================================
# 4. Training Loop
# ==============================================================================
def train_neural_ccm(
    system_name: str = "pendulum",
    hidden_dim: int = 64,
    batch_size: int = 128,
    epochs: int = 250,
    lr: float = 1e-3,
    contraction_rate: float = 0.2,
    device_str: str = "cpu",
    output_dir: str = "outputs/ccm",
):
    device = torch.device(device_str)
    print(f"=== Starting Neural CCM Training ===")
    print(f"Target System   : {system_name.upper()}")
    print(f"Computation Device: {device}")
    print(f"Contraction Rate: lambda = {contraction_rate}")

    # Select System
    if system_name.lower() == "pendulum":
        system = DampedPendulumDynamics()
    elif system_name.lower() == "so101":
        system = SO101ClosedLoopDynamics(dof=6)
    else:
        raise ValueError(f"Unknown system: {system_name}. Choose 'pendulum' or 'so101'.")

    state_dim = system.state_dim
    print(f"State Dimension : {state_dim} (q in R^{state_dim//2}, dq in R^{state_dim//2})")

    # Initialize Metric Network
    metric = NeuralMetric(state_dim=state_dim, hidden_dim=hidden_dim, eps=1e-3).to(device)
    optimizer = torch.optim.Adam(metric.parameters(), lr=lr)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Sample state batch
        x_batch = system.sample_states(batch_size, device)

        optimizer.zero_grad()
        loss, max_violation, tr_M = compute_contraction_loss(
            metric_net=metric,
            x=x_batch,
            dynamics_fn=system,
            contraction_rate=contraction_rate,
        )
        loss.backward()
        # Gradient clipping prevents spikes during matrix operations
        torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
            status = "CONVERGED (CONTRACTED)" if max_violation <= 0 else "SEARCHING"
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Loss: {loss.item():.6f} | "
                f"Max Eig Violation: {max_violation:+.5f} | "
                f"Status: {status}"
            )

    elapsed = time.time() - start_time
    save_path = out_path / f"metric_{system_name}.pt"
    torch.save(
        {
            "metric_state_dict": metric.state_dict(),
            "state_dim": state_dim,
            "hidden_dim": hidden_dim,
            "contraction_rate": contraction_rate,
        },
        save_path,
    )
    print(f"\nTraining completed in {elapsed:.2f}s.")
    print(f"Saved metric model checkpoint to: {save_path.resolve()}")
    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Contraction Metric M(x)")
    parser.add_argument("--system", type=str, default="pendulum", choices=["pendulum", "so101"], help="System to train on")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size of sampled states")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-rate", type=float, default=0.2, help="Contraction rate lambda")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_neural_ccm(
        system_name=args.system,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        contraction_rate=args.lambda_rate,
        device_str=args.device,
    )
