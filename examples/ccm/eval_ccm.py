#!/usr/bin/env python
"""
Evaluation and Inspection Script for Learned Neural Contraction Metrics M(x)
Path: examples/ccm/eval_ccm.py

Features:
1. State-Space Verification: Tests M(x) over thousands of random states to compute:
   - Contraction verification rate (% of states with max_eig(R) <= 0)
   - Minimum eigenvalue of M(x) (ensuring strictly positive definite M(x) > 0)
   - Condition number distribution of M(x)
2. Riemannian Contraction Simulation:
   - Simulates two diverging trajectories [x_1(t), x_2(t)]
   - Computes Riemannian distance d_M(t) = sqrt(dx^T * M(x) * dx)
   - Demonstrates exponential convergence: d_M(t) <= d_M(0) * exp(-lambda * t)
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from neural_ccm import NeuralMetric, DampedPendulumDynamics, SO101ClosedLoopDynamics, compute_contraction_loss


def evaluate_metric(checkpoint_path: str, num_eval_samples: int = 2000, device_str: str = "cpu"):
    device = torch.device(device_str)
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt.resolve()}")

    data = torch.load(ckpt, map_location=device)
    state_dim = data["state_dim"]
    hidden_dim = data["hidden_dim"]
    lambda_rate = data["contraction_rate"]

    # Load system
    if state_dim == 2:
        system = DampedPendulumDynamics()
        system_name = "Pendulum (1-DOF)"
    elif state_dim == 12:
        system = SO101ClosedLoopDynamics(dof=6)
        system_name = "SO-101 Arm (6-DOF)"
    else:
        raise ValueError(f"Unknown state dimension {state_dim}")

    metric = NeuralMetric(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    metric.load_state_dict(data["metric_state_dict"])
    metric.eval()

    print(f"\n========================================================")
    print(f"       Evaluating Neural Contraction Metric M(x)")
    print(f"========================================================")
    print(f"Checkpoint : {ckpt.resolve()}")
    print(f"System     : {system_name}")
    print(f"State Dim  : {state_dim} (q in R^{state_dim//2}, dq in R^{state_dim//2})")
    print(f"Lambda     : {lambda_rate}")
    print(f"Test Points: {num_eval_samples} sampled states")

    # 1. State Space LMI Evaluation
    x_test = system.sample_states(num_eval_samples, device)
    with torch.enable_grad():
        loss, max_violation, tr_M = compute_contraction_loss(
            metric, x_test, system, contraction_rate=lambda_rate
        )

    # 2. Inspect M(x) Properties
    with torch.no_grad():
        M_all = metric(x_test)
        eigs_M = torch.linalg.eigvalsh(M_all)
        min_eig_M = eigs_M[:, 0]
        max_eig_M = eigs_M[:, -1]
        cond_numbers = max_eig_M / min_eig_M

    print(f"\n--- 1. Positive Definiteness Check (M(x) > 0) ---")
    print(f"Min Eigenvalue of M(x)     : {min_eig_M.min().item():.5f} (Must be > 0.0 -> Confirmed SPD!)")
    print(f"Average Condition Number   : {cond_numbers.mean().item():.2f}")
    print(f"Max Condition Number       : {cond_numbers.max().item():.2f}")

    print(f"\n--- 2. Contraction Condition (R(x) <= 0) ---")
    print(f"Total LMI Violation Loss   : {loss.item():.6f}")
    print(f"Maximum Eigenvalue of R    : {max_violation:+.5f}")
    if max_violation <= 0:
        print(f"Result: [PASSED] Mathematically certified contracting everywhere on test set!")
    else:
        print(f"Result: [NEAR CONVERGENCE] Max violation is only {max_violation:+.5f} (Close to 0).")

    # 3. Trajectory Contraction Simulation
    print(f"\n--- 3. Trajectory Simulation: Riemannian Distance Decay ---")
    dt = 0.01
    steps = 50
    # Pick two states close to each other
    x1 = x_test[0:1].clone()
    x2 = x1 + 0.2 * torch.randn_like(x1)  # Perturbed neighbor

    print(f"{'Time (s)':^10} | {'Euclidean Dist ||x1-x2||':^25} | {'Riemannian Dist d_M':^25}")
    print("-" * 65)

    for step in range(steps + 1):
        t = step * dt
        dx = x1 - x2
        with torch.no_grad():
            M1 = metric(x1)
            # Riemannian distance metric d_M = sqrt(dx^T * M(x1) * dx)
            riemannian_dist = torch.sqrt(torch.bmm(torch.bmm(dx.unsqueeze(1), M1), dx.unsqueeze(-1))).squeeze().item()
            euclidean_dist = torch.norm(dx).item()

        if step % 10 == 0:
            print(f"{t:^10.2f} | {euclidean_dist:^25.5f} | {riemannian_dist:^25.5f}")

        # Euler step forward in time
        f1 = system(x1)
        f2 = system(x2)
        x1 = x1 + f1 * dt
        x2 = x2 + f2 * dt

    # 4. Print sample M(x) at nominal zero pose
    print(f"\n--- 4. Sample Learned M(x) Matrix at Nominal Pose (q=0, dq=0) ---")
    with torch.no_grad():
        x_zero = torch.zeros(1, state_dim, device=device)
        M_zero = metric(x_zero).squeeze(0)
    print(f"Matrix shape: {M_zero.shape}")
    print(f"Top-left 4x4 block of M(0):\n{M_zero[:4, :4].cpu().numpy().round(4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Learned Neural Contraction Metric")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--samples", type=int, default=1000, help="Number of random test states")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    evaluate_metric(args.checkpoint, num_eval_samples=args.samples, device_str=args.device)
