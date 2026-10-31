#!/usr/bin/env python

"""
SmolVLA Real Hardware Inference Runner for SO-101 Robotic Arm.

Connects to the SO-101 follower arm and wrist camera (index 1), loads SmolVLA,
and streams actions to the physical motors with safety checks.

Usage:
    # 1. Dry run (reads camera & joint states, runs SmolVLA, prints actions without moving motors):
    python examples/smolvla/run_smolvla_hardware.py --port COM6 --dry-run-motors

    # 2. Live execution on physical arm:
    python examples/smolvla/run_smolvla_hardware.py --port COM6 --camera-index 1 --task "Pick the object and place it"

    # If COM6 is the leader instead, switch port:
    python examples/smolvla/run_smolvla_hardware.py --port COM7
"""

import argparse
import sys
import time

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.feature_utils import hw_to_dataset_features


def parse_args():
    parser = argparse.ArgumentParser(description="Run SmolVLA on SO-101 Follower Arm.")
    parser.add_argument(
        "--port",
        type=str,
        default="COM8",
        help="Serial COM port for SO-101 Follower Arm (default: COM8).",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="my_follower_arm",
        help="Calibration ID for the follower arm (default: my_follower_arm).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV index for wrist camera (default: 1).",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="camera1",
        help="Camera name key expected by model (default: camera1).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick the object and place it",
        help="Natural language prompt for SmolVLA.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="lerobot/smolvla_base",
        help="Hugging Face model ID or local path to checkpoint.",
    )
    parser.add_argument(
        "--dry-run-motors",
        action="store_true",
        default=False,
        help="If set, reads sensors and runs model inference, but DOES NOT send commands to motors (safe test).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1,
        help="Number of episodes to execute (default: 1).",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=50,
        help="Number of steps per episode (default: 50).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control loop target frequency in Hz (default: 30).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(" SmolVLA SO-101 Hardware Deployment")
    print("=" * 65)
    print(f"Device              : {device}")
    print(f"Follower Port       : {args.port}")
    print(f"Wrist Camera Index  : {args.camera_index} (Name: {args.camera_name})")
    print(f"Task Instruction    : \"{args.task}\"")
    print(f"Model ID            : {args.model_id}")
    print(f"Motor Motion Mode   : {'[DRY-RUN / NO MOTOR MOTION]' if args.dry_run_motors else '[LIVE MOTOR MOTION ACTIVE]'}")
    print("=" * 65)

    # 1. Load SmolVLA Policy
    print(f"\n[1/4] Loading SmolVLA Policy from '{args.model_id}'...")
    try:
        model = SmolVLAPolicy.from_pretrained(args.model_id)
        model.to(device)
        model.eval()
        print(f"SmolVLA loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading model '{args.model_id}': {e}")
        sys.exit(1)

    # 2. Setup Pre/Post Processors
    print("\n[2/4] Setting up data processors...")
    try:
        preprocess, postprocess = make_pre_post_processors(
            model.config,
            args.model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    except Exception as e:
        print(f"Note: Standard pre/post processors could not be loaded ({e}), using direct model passthrough.")
        preprocess = lambda x: x  # noqa: E731
        postprocess = lambda x: x  # noqa: E731

    # 3. Configure Hardware (SO-101 Arm + Wrist Camera)
    expected_cam_keys = [k for k in getattr(model.config, "image_features", {})]
    if expected_cam_keys and f"observation.images.{args.camera_name}" not in expected_cam_keys:
        primary_cam_name = expected_cam_keys[0].replace("observation.images.", "")
        print(f"Auto-adapting camera name: '{args.camera_name}' -> '{primary_cam_name}' (expected by {args.model_id})")
        cam_key_for_robot = primary_cam_name
    else:
        cam_key_for_robot = args.camera_name

    print(f"\n[3/4] Connecting to SO-101 Follower on {args.port} & Wrist Camera {args.camera_index} (as '{cam_key_for_robot}')...")
    camera_config = {
        cam_key_for_robot: OpenCVCameraConfig(
            index_or_path=args.camera_index,
            width=640,
            height=480,
            fps=args.fps,
        )
    }

    robot_cfg = SO100FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras=camera_config,
    )
    robot = SO100Follower(robot_cfg)

    try:
        robot.connect()
        print(f"Successfully connected to SO-101 on {args.port}!")
    except Exception as e:
        print(f"\n[!] Failed to connect to robot on {args.port}: {e}")
        print("Tip: If COM6 is the leader arm, try running with: --port COM7")
        sys.exit(1)

    # 4. Prepare Feature Mappings
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Safety prompt before live motion
    if not args.dry_run_motors:
        print("\n" + "!" * 65)
        print(" WARNING: Live motor control is active!")
        print(" - Keep your hand near the power / emergency disconnect.")
        print(" - Press Ctrl+C at any time to immediately stop and disable torque.")
        print("!" * 65)
        try:
            input("\nPress [Enter] to start execution loop (or Ctrl+C to cancel)...")
        except KeyboardInterrupt:
            print("\nAborted by user.")
            robot.disconnect()
            sys.exit(0)

    # 5. Closed-Loop Inference Loop
    print("\n[4/4] Starting Control Loop...")
    step_dt = 1.0 / args.fps

    try:
        for ep in range(args.max_episodes):
            print(f"\n>>> Starting Episode {ep + 1}/{args.max_episodes} (Task: '{args.task}')")
            model.reset()

            for step in range(args.max_steps_per_episode):
                t_step_start = time.perf_counter()

                # A. Read sensory state (wrist camera + joint positions)
                obs = robot.get_observation()

                # B. Build VLA inference frame
                obs_frame = build_inference_frame(
                    observation=obs,
                    device=device,
                    ds_features=dataset_features,
                    task=args.task,
                    robot_type="so101_follower",
                )

                # C. Preprocess
                processed_obs = preprocess(obs_frame)

                # Ensure image features expected by the model exist in processed_obs
                if hasattr(model.config, "image_features"):
                    avail_imgs = [v for k, v in processed_obs.items() if k.startswith("observation.images.")]
                    if avail_imgs:
                        for exp_key in model.config.image_features:
                            if exp_key not in processed_obs:
                                processed_obs[exp_key] = avail_imgs[0]

                # D. SmolVLA forward inference
                with torch.no_grad():
                    action_tensor = model.select_action(processed_obs)

                # E. Postprocess action
                post_action = postprocess(action_tensor)
                robot_action = make_robot_action(post_action, dataset_features)

                # F. Send to motors (or dry run)
                pos_vals = [v for k, v in obs.items() if k.endswith(".pos")]
                if pos_vals:
                    curr_str = ", ".join(f"{float(x):.1f}°" for x in pos_vals[:6])
                else:
                    curr_str = "N/A"

                act_vals = list(robot_action.values()) if isinstance(robot_action, dict) else list(robot_action)
                act_str = ", ".join(f"{float(x):.1f}°" for x in act_vals[:6])

                if not args.dry_run_motors:
                    robot.send_action(robot_action)
                    status_prefix = "[LIVE]"
                else:
                    status_prefix = "[DRY-RUN]"

                if step % 5 == 0 or step == args.max_steps_per_episode - 1:
                    print(f"{status_prefix} Step {step+1:02d}/{args.max_steps_per_episode} | Q_curr: [{curr_str}] -> Target: [{act_str}]")

                # Rate limiting to target FPS
                elapsed = time.perf_counter() - t_step_start
                sleep_time = step_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f">>> Episode {ep + 1} finished successfully!")

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Halting robot and releasing torque...")
    except Exception as e:
        print(f"\n[!] Runtime error in control loop: {e}")
    finally:
        print("Safely disconnecting robot...")
        robot.disconnect()
        print("Disconnected. Motors are safe.")


if __name__ == "__main__":
    main()
