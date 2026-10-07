#!/usr/bin/env python

"""
Robust Safety & Non-Stationary Dynamics Evaluation Pipeline for SO-101 (SO-ARM100)

This script evaluates trained LeRobot policies on the real SO-101 robot arm.
By default, it runs standard policy evaluation. When --perturb-enabled is passed,
it injects programmatic mid-episode dynamic shifts (Torque Degradation, Latency, etc.)
and logs tracking error before and after the shift.
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import load_pretrained_policy_processors
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true or false).")


def parse_args():
    parser = argparse.ArgumentParser(description="SO-101 Policy Evaluation & Perturbation Pipeline")
    parser.add_argument("--policy-path", type=str, required=True, help="Local checkpoint path or Hugging Face hub ID")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port for SO-101 arm (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index for OpenCV (default: 0)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode (~10s at 30Hz)")
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency in Hz")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device for inference (cpu or cuda)")

    # Dynamic Perturbation Option: explicitly set true or false
    parser.add_argument(
        "--perturb-enabled",
        type=str2bool,
        default=False,
        help="Activate (true) or deactivate (false) dynamic perturbations (default: false)",
    )
    parser.add_argument("--perturb-step", type=int, default=100, help="Step index T_perturb to trigger dynamic shift")
    parser.add_argument("--torque-limit", type=int, default=250, help="Degraded Torque_Limit (0-1000, 1000 is default)")
    parser.add_argument("--joints", nargs="+", default=["shoulder_lift", "elbow_flex"], help="Joints to perturb")
    parser.add_argument("--latency-ms", type=float, default=0.0, help="Optional latency injection in ms per step")
    parser.add_argument("--fifo-buffer-size", type=int, default=0, help="Optional FIFO action queue buffer lag (e.g. 3 frames)")
    parser.add_argument("--virtual-gravity-sag-deg", type=float, default=0.0, help="Optional virtual gravity angle sag offset in degrees")
    parser.add_argument("--output-dir", type=str, default="outputs/robust_safety_eval", help="Path to save log metrics")

    return parser.parse_args()


def compute_tracking_error(target_action: dict, present_obs: dict) -> float:
    """Computes Euclidean distance between target joint angles and present joint angles."""
    errors = []
    for k, v in target_action.items():
        if k in present_obs:
            errors.append((v - present_obs[k]) ** 2)
    return math.sqrt(sum(errors)) if errors else 0.0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 1. Configure SO-101 Follower Arm & Camera
    camera_cfg = {}
    if args.camera_index is not None:
        camera_cfg = {
            "laptop": OpenCVCameraConfig(
                index_or_path=args.camera_index,
                width=640,
                height=480,
                fps=args.fps,
            )
        }

    robot_config = SOFollowerRobotConfig(
        port=args.port,
        cameras=camera_cfg,
        perturb_enabled=args.perturb_enabled,
        perturb_step=args.perturb_step,
        perturb_joints=args.joints,
        perturb_torque_limit=args.torque_limit,
        perturb_latency_ms=args.latency_ms,
        perturb_fifo_buffer_size=args.fifo_buffer_size,
        perturb_virtual_gravity_sag_deg=args.virtual_gravity_sag_deg,
    )
    robot = SOFollower(robot_config)

    logger.info(f"Connecting to SO-101 robot on {args.port}...")
    robot.connect()

    # 2. Load Policy and Pre/Post-processors
    logger.info(f"Loading pretrained policy from {args.policy_path}...")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()

    logger.info("Loading policy observation & action processors...")
    preprocessor, postprocessor = load_pretrained_policy_processors(args.policy_path)

    all_episode_metrics = []
    step_duration = 1.0 / args.fps

    try:
        for ep in range(args.episodes):
            logger.info(f"\n{'='*20} Starting Episode {ep + 1}/{args.episodes} {'='*20}")
            if args.perturb_enabled:
                logger.info(f"Perturbation ACTIVE: triggers at step {args.perturb_step} on {args.joints}")
            else:
                logger.info("Standard evaluation mode (No perturbations, 100% full torque)")

            robot.reset_perturbation()
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            pre_perturb_errors = []
            post_perturb_errors = []

            for step in range(args.max_steps):
                t_start = time.perf_counter()

                # Read raw observation from hardware (motor angles + camera image)
                obs_raw = robot.get_observation()

                # Build state vector in joint order
                state_vec = np.array([obs_raw.get(f"{m}.pos", 0.0) for m in JOINT_NAMES], dtype=np.float32)

                obs_dict = {
                    "observation.state": state_vec,
                }
                if "laptop" in obs_raw:
                    obs_dict["observation.images.laptop"] = obs_raw["laptop"]

                # Preprocess observation for model inference
                obs_tensor = prepare_observation_for_inference(obs_dict, device=device)
                obs_processed = preprocessor(obs_tensor)

                with torch.inference_mode():
                    action_tensor = policy.select_action(obs_processed)
                    action_tensor = postprocessor(action_tensor)

                # Format action dictionary for robot motors
                action_values = action_tensor.squeeze().cpu().numpy()
                action_cmd = {f"{m}.pos": float(action_values[i]) for i, m in enumerate(JOINT_NAMES)}

                # Send action to SO-101 arm
                sent_action = robot.send_action(action_cmd)

                # Measure tracking error (target vs actual present state)
                error = compute_tracking_error(sent_action, obs_raw)
                if step < args.perturb_step:
                    pre_perturb_errors.append(error)
                else:
                    post_perturb_errors.append(error)

                # Maintain control loop timing
                elapsed = time.perf_counter() - t_start
                if elapsed < step_duration:
                    time.sleep(step_duration - elapsed)

            pre_mse = float(np.mean(pre_perturb_errors)) if pre_perturb_errors else 0.0
            post_mse = float(np.mean(post_perturb_errors)) if post_perturb_errors else 0.0
            degradation_ratio = (post_mse / pre_mse) if pre_mse > 0 else 1.0

            ep_summary = {
                "episode": ep + 1,
                "pre_perturb_mse": pre_mse,
                "post_perturb_mse": post_mse,
                "degradation_ratio": degradation_ratio,
            }
            all_episode_metrics.append(ep_summary)
            logger.info(
                f"Episode {ep + 1} Complete | Pre-Shift MSE: {pre_mse:.4f} | "
                f"Post-Shift MSE: {post_mse:.4f} | Ratio: {degradation_ratio:.2f}x"
            )

    finally:
        logger.info("Disconnecting robot...")
        robot.disconnect()

    # Save summary report
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "episodes": all_episode_metrics,
                "overall_pre_mse": float(np.mean([m["pre_perturb_mse"] for m in all_episode_metrics])) if all_episode_metrics else 0.0,
                "overall_post_mse": float(np.mean([m["post_perturb_mse"] for m in all_episode_metrics])) if all_episode_metrics else 0.0,
            },
            f,
            indent=4,
        )
    logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()

