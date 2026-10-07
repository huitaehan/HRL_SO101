#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True

    # Position-mode PID gains written to Feetech STS3215 motors at connect time.
    position_p_coefficient: int = 16
    position_i_coefficient: int = 0
    position_d_coefficient: int = 32

    # Number of extra attempts when a `sync_read` of the motors fails. Feetech buses can occasionally
    # return a corrupted status packet ("Incorrect status packet!"), especially when several joints move
    # at once, which otherwise aborts the control loop. Retries are immediate (no sleep) and only happen on
    # failure, so the steady-state read cost is unchanged.
    num_read_retries: int = 2

    # Dynamic Perturbation Settings (Mid-episode dynamic shifts)
    # Allows mid-episode programmatic modulation of motor torque, latency, FIFO delays, and gravity offsets.
    perturb_enabled: bool = False  # Set to True to enable dynamic perturbations (default: False)
    perturb_step: int = 100        # Step index T_perturb where the perturbation activates
    perturb_joints: list[str] = field(default_factory=lambda: ["shoulder_lift", "elbow_flex"])  # Target joints

    # Method 1: Hardware Torque Degradation & Gains
    perturb_torque_limit: int = 250  # RAM Torque_Limit (0-1000; default is 1000)
    perturb_p_gain: int | None = None  # Optional target P gain (P_Coefficient)
    perturb_latency_ms: float = 0.0  # Optional sleep delay in ms per step (post-perturbation)

    # Method 2: Action FIFO Buffer Delay (Temporal Latency)
    perturb_fifo_buffer_size: int = 0  # Number of frames to delay actions (e.g., 3 frames = ~100ms at 30Hz)

    # Method 3: Virtual Gravity Sag Offset (Kinematic Bias Offset)
    perturb_virtual_gravity_sag_deg: float = 0.0  # Joint target angle sag offset in degrees


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass


SO100FollowerConfig = SOFollowerRobotConfig
SO101FollowerConfig = SOFollowerRobotConfig
