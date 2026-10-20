#!/usr/bin/env python
"""
Visualization and Inspection Utility for Learned Contraction Metric M(x)
Path: examples/ccm/visualize_ccm.py

Generates:
1. Full 12x12 Metric Heatmap: Visualizes M(x) and joint coupling structure at different poses.
2. Riemannian Contraction Curve: Plots exponential convergence d_M(t) vs Euclidean distance.
3. Interactive Console Inspection: Explains how to query M(x) for any robot state.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from neural_ccm import NeuralMetric, SO101ClosedLoopDynamics, DampedPendulumDynamics


def visualize_metric(checkpoint_path: str, output_dir: str = "outputs/ccm"):
    ckpt = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(ckpt, map_location="cpu")
    state_dim = data["state_dim"]
    hidden_dim = data["hidden_dim"]
    lambda_rate = data["contraction_rate"]

    metric = NeuralMetric(state_dim=state_dim, hidden_dim=hidden_dim)
    metric.load_state_dict(data["metric_state_dict"])
    metric.eval()

    print(f"\n========================================================")
    print(f"       Visualizing Trained Metric M(x)")
    print(f"========================================================")
    print(f"Checkpoint : {ckpt.resolve()}")
    print(f"State Dim  : {state_dim}")

    # 1. Print full 12x12 matrix for nominal pose
    x_zero = torch.zeros(1, state_dim)
    with torch.no_grad():
        M_zero = metric(x_zero).squeeze(0).numpy()

    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper"
    ]
    state_labels = [f"q_{name[:4]}" for name in joint_names] + [f"dq_{name[:4]}" for name in joint_names]

    print(f"\nFull {state_dim}x{state_dim} Learned M(0) at Nominal Pose (q=0, dq=0):")
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(M_zero)

    # 2. Try generating matplotlib plots if installed
    try:
        import matplotlib.pyplot as plt

        # Plot 1: Heatmap of M(0)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(M_zero, cmap="viridis", interpolation="nearest")
        ax.set_xticks(range(state_dim))
        ax.set_yticks(range(state_dim))
        ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(state_labels, fontsize=9)
        ax.set_title(f"Learned Riemannian Metric Matrix $M(x=0)$\n(SO-101 6-DOF)", fontsize=13, pad=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Metric Magnitude")
        plt.tight_layout()
        heatmap_file = out_dir / "metric_heatmap.png"
        fig.savefig(heatmap_file, dpi=300)
        plt.close(fig)
        print(f"\n[Saved] Matrix heatmap plot -> {heatmap_file.resolve()}")

        # Plot 2: Riemannian Contraction Curve
        if state_dim == 12:
            system = SO101ClosedLoopDynamics(dof=6)
        else:
            system = DampedPendulumDynamics()

        # Simulate trajectory decay
        dt = 0.01
        steps = 60
        x1 = system.sample_states(1, torch.device("cpu"))
        x2 = x1 + 0.25 * torch.randn_like(x1)

        t_hist, d_riem_hist, d_euc_hist = [], [], []
        for step in range(steps):
            t = step * dt
            dx = x1 - x2
            with torch.no_grad():
                M1 = metric(x1)
                d_riem = torch.sqrt(torch.bmm(torch.bmm(dx.unsqueeze(1), M1), dx.unsqueeze(-1))).item()
                d_euc = torch.norm(dx).item()
            t_hist.append(t)
            d_riem_hist.append(d_riem)
            d_euc_hist.append(d_euc)

            f1 = system(x1)
            f2 = system(x2)
            x1 = x1 + f1 * dt
            x2 = x2 + f2 * dt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(t_hist, d_riem_hist, label="Riemannian Distance $d_M(t)$ (Contracts)", color="tab:blue", lw=2.2)
        ax.plot(t_hist, d_euc_hist, label="Euclidean Distance $\|x_1 - x_2\|$", color="tab:gray", ls="--", lw=1.8)
        # Theoretical exponential bound
        bound = d_riem_hist[0] * np.exp(-lambda_rate * np.array(t_hist))
        ax.plot(t_hist, bound, label=f"Theoretical Bound $e^{{-\\lambda t}}$ ($\\lambda={lambda_rate}$)", color="tab:red", ls=":", lw=1.5)

        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Distance between Trajectories", fontsize=11)
        ax.set_title("Trajectory Contraction Verification on SO-101", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
        plt.tight_layout()
        curve_file = out_dir / "contraction_curve.png"
        fig.savefig(curve_file, dpi=300)
        plt.close(fig)
        print(f"[Saved] Trajectory contraction curve -> {curve_file.resolve()}")

    except ImportError:
        print("\nNote: matplotlib not installed in current environment. To generate PNG plots, run:")
        print("pip install matplotlib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Trained Contraction Metric M(x)")
    parser.add_argument("--checkpoint", type=str, default="outputs/ccm/metric_so101.pt", help="Path to checkpoint file")
    parser.add_argument("--output-dir", type=str, default="outputs/ccm", help="Directory to save plots")
    args = parser.parse_args()

    visualize_metric(args.checkpoint, args.output_dir)
