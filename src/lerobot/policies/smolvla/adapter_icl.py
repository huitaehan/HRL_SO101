#!/usr/bin/env python

# Copyright 2025 LeRobot and Implicit Contraction Layers integration.
# All rights reserved.
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

"""
SmolVLA to Implicit Contraction Layers (ICL) Adapter.

This adapter acts as the bridge between SmolVLA (high-level vision-language-action policy)
and the Implicit Contraction Layer (low-level safety/contraction QP engine).

Contract requirements from ICL pipeline:
1. `act(observation)`:
   - Takes state/observation: 1D (state_dim,) or 2D (batch_size, state_dim).
   - Returns bounded action array: (action_dim,) or (batch_size, action_dim) in [lower, upper].
2. `reset()`:
   - Clears action queues when an episode finishes.
3. Checkpoint compatibility with `best.pt` in ICL runs.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class SmolVLAAdapter:
    """
    Adapter wrapping SmolVLAPolicy to satisfy the ICL teacher / nominal policy interface.
    """

    def __init__(
        self,
        policy: nn.Module,
        action_dim: int,
        action_lower: float | list[float] | np.ndarray = -1.0,
        action_upper: float | list[float] | np.ndarray = 1.0,
        device: str | torch.device = "cpu",
        default_instruction: str = "manipulate robot arm safely",
        default_image_size: tuple[int, int, int] = (3, 480, 640),
        gripper_mode: str = "open",  # "open" (Reach), "fixed_grasp" (Carry), or "dynamic"
    ):
        self.policy = policy.to(device).eval()
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.instruction = default_instruction
        self.default_image_size = default_image_size
        self.gripper_mode = gripper_mode

        # Action bounds
        if isinstance(action_lower, (int, float)):
            self.action_lower = np.full(action_dim, float(action_lower), dtype=np.float32)
        else:
            self.action_lower = np.asarray(action_lower, dtype=np.float32)

        if isinstance(action_upper, (int, float)):
            self.action_upper = np.full(action_dim, float(action_upper), dtype=np.float32)
        else:
            self.action_upper = np.asarray(action_upper, dtype=np.float32)

        self._lower_t = torch.as_tensor(self.action_lower, device=self.device, dtype=torch.float32)
        self._upper_t = torch.as_tensor(self.action_upper, device=self.device, dtype=torch.float32)

        # Cache tokens if policy has processor/tokenizer
        self._cached_tokens: Tensor | None = None
        self._cached_mask: Tensor | None = None
        self._init_language_tokens()

    def _init_language_tokens(self):
        """Prepare cached language tokens if tokenizer is available."""
        processor = getattr(self.policy, "processor", None)
        if processor is None and hasattr(self.policy, "model"):
            vlm = getattr(self.policy.model, "vlm_with_expert", None)
            if vlm is not None:
                processor = getattr(vlm, "processor", None)

        if processor is not None and hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
            encoded = tokenizer(
                self.instruction,
                return_tensors="pt",
                padding="max_length",
                max_length=getattr(getattr(self.policy, "config", None), "tokenizer_max_length", 48),
                truncation=True,
            )
            self._cached_tokens = encoded["input_ids"].to(self.device)
            self._cached_mask = encoded["attention_mask"].to(self.device)

    def reset(self):
        """Reset internal policy queues (e.g., action chunks)."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _build_batch(self, obs_np: np.ndarray) -> dict[str, Tensor]:
        """Convert state numpy array into SmolVLA observation batch."""
        batch_size = obs_np.shape[0]
        state_tensor = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32)

        batch: dict[str, Tensor] = {
            OBS_STATE: state_tensor,
            "observation.state": state_tensor,
        }

        # Language tokens
        if self._cached_tokens is not None and self._cached_mask is not None:
            batch[OBS_LANGUAGE_TOKENS] = self._cached_tokens.expand(batch_size, -1)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = self._cached_mask.expand(batch_size, -1)
        else:
            dummy_len = 16
            batch[OBS_LANGUAGE_TOKENS] = torch.zeros((batch_size, dummy_len), dtype=torch.long, device=self.device)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones((batch_size, dummy_len), dtype=torch.long, device=self.device)

        # Visual features required by SmolVLA config
        config = getattr(self.policy, "config", None)
        if config is not None and hasattr(config, "image_features"):
            for img_key, feat in config.image_features.items():
                shape = (batch_size, *feat.shape)
                batch[img_key] = torch.zeros(shape, dtype=torch.float32, device=self.device)

        return batch

    @torch.no_grad()
    def act(self, observation: np.ndarray, stochastic: bool = False) -> np.ndarray:
        """
        Produce a bounded action given observation/state.

        Args:
            observation: 1D (state_dim,) or 2D (batch_size, state_dim) array.
            stochastic: Whether to sample with noise (ignored for deterministic eval).

        Returns:
            action: 1D or 2D numpy array bounded by [action_lower, action_upper].
        """
        is_single = observation.ndim == 1
        obs_arr = np.atleast_2d(observation).astype(np.float32)

        batch = self._build_batch(obs_arr)

        if hasattr(self.policy, "select_action"):
            raw_action = self.policy.select_action(batch)
        elif hasattr(self.policy, "forward"):
            out = self.policy(batch)
            raw_action = out[0] if isinstance(out, tuple) else out
        else:
            raise AttributeError("Wrapped policy must have `select_action` or `forward` method.")

        if not isinstance(raw_action, torch.Tensor):
            raw_action = torch.as_tensor(raw_action, device=self.device, dtype=torch.float32)

        if raw_action.ndim == 1:
            raw_action = raw_action.unsqueeze(0)

        # Slice to task action dimension
        raw_action = raw_action[:, : self.action_dim]

        # Strictly enforce bounds
        clamped = torch.clamp(raw_action, self._lower_t, self._upper_t)
        action_np = clamped.cpu().numpy()

        return action_np[0] if is_single else action_np

    def checkpoint(self) -> dict[str, Any]:
        """Returns payload for ICL best.pt saving."""
        return {
            "policy_state_dict": self.policy.state_dict(),
            "action_dim": self.action_dim,
            "action_lower": self.action_lower,
            "action_upper": self.action_upper,
            "device": str(self.device),
        }

    def save(self, path: str | Path):
        """Save adapter checkpoint."""
        torch.save(self.checkpoint(), path)

    @classmethod
    def load(cls, path: str | Path, policy_factory, device: str = "cpu") -> "SmolVLAAdapter":
        """Load adapter from checkpoint."""
        data = torch.load(path, map_location=device, weights_only=False)
        policy = policy_factory()
        policy.load_state_dict(data["policy_state_dict"])
        return cls(
            policy=policy,
            action_dim=data["action_dim"],
            action_lower=data["action_lower"],
            action_upper=data["action_upper"],
            device=device,
        )
