r"""
SmolVLA + ICL Simulation Testbed with 3D Graphics Visualization
Path: examples/smolvla/simulate_smolvla_icl.py

Runs a closed-loop simulation on MuJoCo (CPU-only):
1. Initializes the TrackingPendulum (or Reacher) environment.
2. Supports live 3D window visualization (`--render`) or MP4 video recording (`--save-video`).
3. SmolVLAAdapter generates action proposals.
4. The ICL Differentiable QP layer filters/projects the actions.
5. Steps the MuJoCo physics engine and logs state & interventions.

Usage:
    # 1. Live 3D graphical window:
    python examples/smolvla/simulate_smolvla_icl.py --render
    python examples/smolvla/simulate_smolvla_icl.py --env reacher --render

    # 2. Save video recording (MP4):
    python examples/smolvla/simulate_smolvla_icl.py --save-video
    python examples/smolvla/simulate_smolvla_icl.py --env reacher --save-video
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

# Ensure paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ICL_SRC = WORKSPACE_ROOT / "implicit_contraction_layers" / "src"

if str(WORKSPACE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
if str(ICL_SRC) not in sys.path:
    sys.path.insert(0, str(ICL_SRC))

from lerobot.policies.smolvla.adapter_icl import SmolVLAAdapter
from icl_core.qp import solve_box_halfspace
from icl_core.pendulum import TrackingPendulum
from icl_core.reacher import TrackingReacher
from icl_core.utils import load_config


class CustomTrackingPendulum(TrackingPendulum):
    def __init__(self, settings, render_mode=None):
        env = gym.make(
            settings["id"],
            reset_noise_scale=settings["reset_noise_scale"],
            max_episode_steps=settings["episode_steps"],
            render_mode=render_mode,
        )
        super(TrackingPendulum, self).__init__(env)
        self.settings = settings
        self.target = settings["target_position"]
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)


class CustomTrackingReacher(TrackingReacher):
    def __init__(self, settings, render_mode=None):
        env = gym.make(settings["id"], max_episode_steps=settings["episode_steps"], render_mode=render_mode)
        super(TrackingReacher, self).__init__(env)
        self.settings = settings
        self.target = np.asarray(settings["target"], np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)


class SimpleICLFilter(nn.Module):
    """Applies a box-constrained halfspace QP to project action proposals."""
    def __init__(self, action_lower, action_upper):
        super().__init__()
        self.lower = torch.as_tensor(action_lower, dtype=torch.float32)
        self.upper = torch.as_tensor(action_upper, dtype=torch.float32)
        self.weights = torch.ones_like(self.lower)

    def forward(self, proposal: np.ndarray, normal: np.ndarray, bound: float) -> tuple[np.ndarray, bool, bool]:
        prop_t = torch.as_tensor(proposal, dtype=torch.float32)
        norm_t = torch.as_tensor(normal, dtype=torch.float32)
        bnd_t = torch.as_tensor(bound, dtype=torch.float32)

        action_t, feasible, _ = solve_box_halfspace(
            prop_t, norm_t, bnd_t, self.lower, self.upper, self.weights
        )
        was_projected = not torch.allclose(prop_t, action_t, atol=1e-4)
        return action_t.numpy(), bool(feasible), was_projected


class NominalLinearPolicy(nn.Module):
    """A baseline stabilizing feedback policy."""
    def __init__(self, action_dim: int, gain: float = 1.2):
        super().__init__()
        self.gain = gain
        self.action_dim = action_dim

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = batch["observation.state"]
        if self.action_dim == 1:
            # Angle and angular velocity feedback
            action = -(state[:, 1:2] * 2.5 + state[:, 3:4] * 0.8)
        else:
            action = -state[:, :self.action_dim] * self.gain
        return action


def run_simulation(env_name: str = "pendulum", num_steps: int = 150, render: bool = False, save_video: bool = False):
    print("=" * 70)
    print(f" SmolVLA + ICL Simulation on MuJoCo: [{env_name.upper()}]")
    if render:
        print(" [GUI Active] Rendering live 3D window...")
    if save_video:
        print(" [Video Active] Recording frames for MP4 export...")
    print("=" * 70)

    render_mode = "human" if render else ("rgb_array" if save_video else None)

    # 1. Setup Environment
    if env_name == "pendulum":
        config_path = WORKSPACE_ROOT / "implicit_contraction_layers/experiments/inverted_pendulum_from_scratch/config.yaml"
        config = load_config(config_path)
        env = CustomTrackingPendulum(config["tracking_pendulum"], render_mode=render_mode)
        action_dim = 1
        action_limit = 1.0
    else:
        config_path = WORKSPACE_ROOT / "implicit_contraction_layers/experiments/reacher_from_scratch/config.yaml"
        config = load_config(config_path)
        env = CustomTrackingReacher(config["tracking_reacher"], render_mode=render_mode)
        action_dim = 2
        action_limit = 1.0

    # 2. Setup SmolVLAAdapter wrapping policy
    base_policy = NominalLinearPolicy(action_dim=action_dim)
    adapter = SmolVLAAdapter(
        policy=base_policy,
        action_dim=action_dim,
        action_lower=-action_limit,
        action_upper=action_limit,
    )

    # 3. Setup ICL Safety QP Filter
    icl_filter = SimpleICLFilter(
        action_lower=[-action_limit] * action_dim,
        action_upper=[action_limit] * action_dim,
    )

    # 4. Run Closed-Loop Episode
    state, info = env.reset(seed=42)
    adapter.reset()

    total_reward = 0.0
    qp_interventions = 0
    history = []
    frames = []

    dt = getattr(env, "dt", 0.02)
    print(f"\nSimulating {num_steps} steps (dt={dt}s)...")

    for t in range(num_steps):
        # A. Adapter proposes nominal action
        u_proposal = adapter.act(state)

        # B. ICL Contraction Filter checks constraints
        normal = np.ones(action_dim, dtype=np.float32)
        bound = 0.6  # Safety acceleration envelope

        u_safe, feasible, modified = icl_filter(u_proposal, normal, bound)
        if modified:
            qp_interventions += 1

        # C. Step physics in MuJoCo
        next_state, reward, terminated, truncated, step_info = env.step(u_safe)
        total_reward += reward
        state = next_state

        history.append({
            "step": t,
            "proposal": u_proposal.copy(),
            "safe_action": u_safe.copy(),
            "reward": reward,
            "modified": modified,
        })

        # D. Rendering
        if render:
            env.render()
            time.sleep(dt)  # Sync to real-time physical playback speed
        elif save_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if t % 25 == 0 or t == num_steps - 1:
            print(f"  Step {t:3d} | State: {np.round(state[:2], 3)} | Action: {np.round(u_safe, 3)} | Filter Active: {modified}")

        if terminated or truncated:
            break

    env.close()

    # 5. Save Video/GIF if requested
    if save_video and frames:
        import imageio
        out_dir = WORKSPACE_ROOT / "outputs" / "simulations"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Try MP4 with libx264, fallback to GIF
        try:
            video_path = out_dir / f"sim_{env_name}.mp4"
            imageio.mimsave(str(video_path), frames, fps=int(1.0 / dt), codec="libx264")
        except Exception:
            video_path = out_dir / f"sim_{env_name}.gif"
            imageio.mimsave(str(video_path), frames, duration=dt)
        print(f"\n [Saved Video] Recording exported to: {video_path}")

    # 6. Summary
    print("\n" + "-" * 70)
    print(f" Episode Completed in {len(history)} steps")
    print(f" Total Cumulative Reward    : {total_reward:.2f}")
    print(f" QP Safety Interventions    : {qp_interventions} / {len(history)} ({(qp_interventions / len(history))*100:.1f}%)")
    print(f" Final System State         : {np.round(state, 4)}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["pendulum", "reacher"], default="pendulum")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--render", action="store_true", help="Open interactive 3D graphical window")
    parser.add_argument("--save-video", action="store_true", help="Record and save MP4 video file")
    args = parser.parse_args()
    run_simulation(env_name=args.env, num_steps=args.steps, render=args.render, save_video=args.save_video)
