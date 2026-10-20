#!/usr/bin/env python
"""
SmolVLA Dry-Run Verification Script for SO-101
Path: examples/smolvla/verify_smolvla.py

Verifies:
1. Environment and dependencies (PyTorch, CUDA, Transformers, LeRobot modules).
2. Model construction and parameter count (VLM Backbone + Action Expert).
3. SO-101 tensor input/output shapes (Images: 3x480x640, State: 6-DoF, Action Chunk: 50x6).
4. Forward loss computation (Flow Matching training step).
5. Action chunk prediction and single-step action selection (Inference step).
"""

import argparse
import sys
import time
from pathlib import Path

import torch


def check_dependencies():
    print("\n" + "=" * 65)
    print(" 1. System & Dependency Check")
    print("=" * 65)
    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"PyTorch Version     : {torch.__version__}")
    print(f"CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total          : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Device              : CPU")

    try:
        import transformers
        print(f"Transformers Version: {transformers.__version__}")
    except ImportError as e:
        print(f"[!] Warning: 'transformers' not found: {e}")
        print("    Install with: pip install transformers or uv sync --extra smolvla")


def build_so101_config(
    load_vlm_weights: bool = False,
    chunk_size: int = 50,
    num_steps: int = 10,
    device: str = "cpu",
):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    input_features = {
        "observation.images.top": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 480, 640),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(6,),
        ),
    }

    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(6,),
        ),
    }

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        num_steps=num_steps,
        load_vlm_weights=load_vlm_weights,
        device=device,
        freeze_vision_encoder=True,
        train_expert_only=True,
        resize_imgs_with_padding=(512, 512),
        tokenizer_max_length=48,
    )
    return config


def run_dry_run(args):
    check_dependencies()

    device_str = "cuda" if (torch.cuda.is_available() and not args.force_cpu) else "cpu"
    device = torch.device(device_str)

    print("\n" + "=" * 65)
    print(" 2. Configuring SmolVLA for SO-101 Arm")
    print("=" * 65)
    print(f"Target Device       : {device}")
    print(f"State Dimension     : 6 (SO-101 joints)")
    print(f"Action Dimension    : 6 (SO-101 motor targets)")
    print(f"Camera Input        : 1x Top Camera (3x480x640)")
    print(f"Action Chunk Size   : {args.chunk_size}")
    print(f"Euler Denoise Steps : {args.num_steps}")
    print(f"Load Full Weights   : {args.load_vlm_weights}")

    config = build_so101_config(
        load_vlm_weights=args.load_vlm_weights,
        chunk_size=args.chunk_size,
        num_steps=args.num_steps,
        device=device_str,
    )

    print("\n" + "=" * 65)
    print(" 3. Instantiating SmolVLAPolicy")
    print("=" * 65)
    t0 = time.perf_counter()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy(config)
    policy.to(device)
    instantiate_time = time.perf_counter() - t0
    print(f"Initialization Time : {instantiate_time:.2f} s")

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    expert_params = sum(p.numel() for p in policy.model.vlm_with_expert.lm_expert.parameters())
    vlm_params = sum(p.numel() for p in policy.model.vlm_with_expert.vlm.parameters())

    print(f"Total Parameters    : {total_params / 1e6:.1f} M")
    print(f"  - VLM Backbone    : {vlm_params / 1e6:.1f} M (Frozen: {config.freeze_vision_encoder})")
    print(f"  - Action Expert   : {expert_params / 1e6:.1f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.1f} M ({trainable_params / total_params * 100:.1f}%)")

    print("\n" + "=" * 65)
    print(" 4. Generating Synthetic SO-101 Observations")
    print("=" * 65)
    batch_size = args.batch_size
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    task_prompt = "Pick the object and place it"
    tokenized = tokenizer(
        task_prompt,
        padding="max_length",
        max_length=config.tokenizer_max_length,
        return_tensors="pt",
    )

    dummy_batch = {
        "observation.images.top": torch.rand(batch_size, 3, 480, 640, device=device, dtype=torch.float32),
        "observation.state": torch.randn(batch_size, 6, device=device, dtype=torch.float32),
        "observation.language.tokens": tokenized["input_ids"].repeat(batch_size, 1).to(device),
        "observation.language.attention_mask": tokenized["attention_mask"].repeat(batch_size, 1).to(device=device, dtype=torch.bool),
        "action": torch.randn(batch_size, config.chunk_size, 6, device=device, dtype=torch.float32),
    }

    print(f"Batch Size          : {batch_size}")
    for k, v in dummy_batch.items():
        print(f"  - {k:<36}: shape {tuple(v.shape)}, dtype {v.dtype}")

    print("\n" + "=" * 65)
    print(" 5. Testing Training Forward Pass (Flow Matching Loss)")
    print("=" * 65)
    policy.train()
    t_start = time.perf_counter()
    loss, loss_dict = policy(dummy_batch)
    train_time = time.perf_counter() - t_start
    print(f"Training Loss       : {loss.item():.4f}")
    print(f"Forward Pass Time   : {train_time * 1000:.1f} ms")
    assert not torch.isnan(loss), "Loss computed is NaN!"

    print("\n" + "=" * 65)
    print(" 6. Testing Inference Pass (predict_action_chunk)")
    print("=" * 65)
    policy.eval()
    with torch.no_grad():
        t_start = time.perf_counter()
        pred_chunk = policy.predict_action_chunk(dummy_batch)
        infer_time = time.perf_counter() - t_start

    print(f"Predicted Chunk     : shape {tuple(pred_chunk.shape)} (Expected: ({batch_size}, {config.chunk_size}, 6))")
    print(f"Inference Time      : {infer_time * 1000:.1f} ms ({infer_time / config.chunk_size * 1000:.1f} ms per step)")
    assert pred_chunk.shape == (batch_size, config.chunk_size, 6), (
        f"Expected shape ({batch_size}, {config.chunk_size}, 6), got {pred_chunk.shape}"
    )

    print("\n" + "=" * 65)
    print(" 7. Testing Sequential select_action (Control Loop Queue)")
    print("=" * 65)
    policy.reset()
    with torch.no_grad():
        single_obs = {
            "observation.images.top": dummy_batch["observation.images.top"][:1],
            "observation.state": dummy_batch["observation.state"][:1],
            "observation.language.tokens": dummy_batch["observation.language.tokens"][:1],
            "observation.language.attention_mask": dummy_batch["observation.language.attention_mask"][:1],
        }
        action_step_1 = policy.select_action(single_obs)
        action_step_2 = policy.select_action(single_obs)

    print(f"Single Step Action 1: shape {tuple(action_step_1.shape)}, values: {action_step_1.cpu().numpy().round(3).tolist()}")
    print(f"Single Step Action 2: shape {tuple(action_step_2.shape)}, values: {action_step_2.cpu().numpy().round(3).tolist()}")
    print(f"Remaining in Queue  : {len(policy._queues['action'])} actions cached")

    print("\n" + "=" * 65)
    print(" [✓] SmolVLA SO-101 Verification Passed Successfully!")
    print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Dry-run SmolVLA verification on SO-101 arm.")
    parser.add_argument(
        "--load-vlm-weights",
        action="store_true",
        default=False,
        help="Download full pretrained SmolVLM weights from Hugging Face Hub (default: False for fast offline test)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        default=False,
        help="Force CPU execution even if CUDA is available",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for verification (default: 2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Action chunk size (default: 50)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of Euler integration denoising steps for quick dry-run (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_dry_run(parse_args())
