# Robust Safety & Non-Stationary Dynamics Evaluation for SO-101

This module implements a physical evaluation pipeline to test the resilience, safety, and stability of imitation learning policies (e.g., ACT, Diffusion, SmolVLA) running on the **SO-101 (SO-ARM100)** 6-DoF follower arm when subjected to real-world hardware degradations and dynamic perturbations.

---

## 🎯 Motivation

In standard robot evaluation, models are tested in stationary, ideal conditions. In production or extended deployments, robots suffer from:
1. Motor heating, brownouts, or mechanical wear (causing torque degradation).
2. Processing overload or WiFi/USB packet delay (causing latency and queue buffering).
3. Payload drops, structural play, or gravitational sag.

This module injects programmatic, mid-episode physical disturbances at a predetermined step $T_{\text{perturb}}$ and quantifies policy degradation:

$$
\text{Degradation Ratio} = \frac{\text{MSE}_{\text{post-shift}}}{\text{MSE}_{\text{pre-shift}}}
$$

---

## 🛠️ The 4 Disturbance Injection Modes

The evaluation script ([`robust_safety_so101_eval.py`](./robust_safety_so101_eval.py)) and the robot interface ([`config_so_follower.py`](../../src/lerobot/robots/so_follower/config_so_follower.py)) support 4 independent or simultaneous perturbation functions:

| Disturbance Mode | CLI Argument | Default | Physical Phenomenon Simulated |
| :--- | :--- | :--- | :--- |
| **Hardware Torque Degradation** | `--torque-limit <0-1000>` | `250` (degraded from 1000) | Motor overheating, voltage drops, mechanical friction, payload overload. |
| **Step Latency Injection** | `--latency-ms <ms>` | `0.0` | Control loop delays, inference compute bottlenecks, vision sensor throttles. |
| **Action Queue Lag (FIFO)** | `--fifo-buffer-size <N>` | `0` frames | Stale action execution, transport packet lag (~100 ms lag for 3 frames at 30 Hz). |
| **Virtual Gravity Droop / Sag** | `--virtual-gravity-sag-deg <deg>` | `0.0` | Structural arm compliance, loose joint fasteners, unmodeled end-effector load. |

---

## 🚀 Usage

### 1. Standard (Stationary) Policy Evaluation
Run nominal evaluation without perturbations (100% full torque, 0 latency):

```powershell
python examples/safety/robust_safety_so101_eval.py `
    --policy-path outputs/train/act_so101/checkpoints/last/pretrained_model `
    --port COM6 `
    --camera-index 1 `
    --episodes 5 `
    --max-steps 300 `
    --perturb-enabled false `
    --output-dir outputs/nominal_eval
```

### 2. Perturbation & Safety Stress-Testing
Evaluate policy survival under simultaneous torque drop and action latency starting at step 100:

```powershell
python examples/safety/robust_safety_so101_eval.py `
    --policy-path outputs/train/act_so101/checkpoints/last/pretrained_model `
    --port COM6 `
    --camera-index 1 `
    --episodes 5 `
    --max-steps 300 `
    --perturb-enabled true `
    --perturb-step 100 `
    --torque-limit 250 `
    --latency-ms 20.0 `
    --fifo-buffer-size 3 `
    --virtual-gravity-sag-deg 4.0 `
    --output-dir outputs/robust_safety_eval
```

---

## 📊 Output Metrics (`eval_report.json`)

Results are logged automatically to `outputs/robust_safety_eval/eval_report.json`:

```json
{
    "config": {
        "policy_path": "outputs/train/act_so101/checkpoints/last/pretrained_model",
        "perturb_enabled": true,
        "perturb_step": 100,
        "torque_limit": 250,
        "latency_ms": 20.0,
        "fifo_buffer_size": 3,
        "virtual_gravity_sag_deg": 4.0
    },
    "episodes": [
        {
            "episode": 1,
            "pre_perturb_mse": 0.0124,
            "post_perturb_mse": 0.0418,
            "degradation_ratio": 3.37
        }
    ],
    "overall_pre_mse": 0.0124,
    "overall_post_mse": 0.0418
}
```
