# HRL_SO101

Repository for training, evaluating, and deploying imitation learning policies on the **SO-101 6-DoF robotic arm** using [LeRobot](https://github.com/huggingface/lerobot), with custom **non-stationary dynamics and robust safety evaluation**.

---

## 🛠️ Hardware Setup

- **Follower Arm**: SO-101 (`so101_follower` with Feetech STS3215 bus)
- **Leader Arm / Teleoperator**: SO-101 Leader (`so101_leader` with Feetech STS3215 bus)
- **Camera**: Top-mounted camera (`opencv`, index `1`, 640x480 @ 30 FPS)
- **Default Ports (Windows)**:
  - Follower: `COM6`
  - Leader: `COM7`

---

## 📦 Dataset

The demonstration dataset is hosted publicly on Hugging Face Hub:
- **Repo ID**: [`huitaehan/so101_demo`](https://huggingface.co/datasets/huitaehan/so101_demo)
- **Episodes**: 48 demonstrations
- **Task**: *"Pick the object and place it"*
- **Sensory Modalities**:
  - `observation.images.top`: 640x480 RGB @ 30 FPS
  - `observation.state`: 6-DoF joint positions
  - `action`: 6-DoF leader joint targets

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/huitaehan/HRL_SO101.git
cd HRL_SO101
```

### 2. Install dependencies
```bash
pip install -e .
# Or with uv:
uv sync --locked --extra feetech
```

---

## 🏋️ Training (On GPU Server)

Train an **ACT (Action Chunking with Transformers)** policy on the dataset. The dataset will be automatically downloaded from Hugging Face:

```bash
lerobot-train \
    --dataset.repo_id=huitaehan/so101_demo \
    --policy.type=act \
    --policy.device=cuda \
    --output_dir=outputs/train/act_so101 \
    --job_name=act_so101 \
    --batch_size=8 \
    --training.steps=20000 \
    --wandb.enable=false
```

Once training finishes, the model checkpoint is saved to:
`outputs/train/act_so101/checkpoints/last/pretrained_model`

---

## 🛡️ Robust Safety & Non-Stationary Dynamics Evaluation

We implemented custom **safety and disturbance injection capabilities** in [`examples/robust_safety_so101_eval.py`](./examples/robust_safety_so101_eval.py) and [`src/lerobot/robots/so_follower/config_so_follower.py`](./src/lerobot/robots/so_follower/config_so_follower.py).

This pipeline evaluates whether a policy can survive sudden, mid-episode physical and system disturbances (triggered at step $T_{\text{perturb}}$):

### 4 Disturbance Functions Implemented:

1. **Hardware Torque Degradation (`--torque-limit`)**:
   - Programmatically reduces Feetech STS3215 RAM `Torque_Limit` (range 0–1000, default degraded to 250) on specified joints (`shoulder_lift`, `elbow_flex`).
   - **Simulates**: Motor overheating, voltage drops, mechanical wear, or heavy payload resistance.

2. **Step Latency Injection (`--latency-ms`)**:
   - Artificially injects execution delays in milliseconds into each control loop step post-perturbation.
   - **Simulates**: Communication latency, inference bottleneck, or sensor bandwidth throttling.

3. **Temporal Action Queue Lag (`--fifo-buffer-size`)**:
   - Enforces an $N$-frame FIFO queue lag on model actions (e.g., 3 frames = ~100 ms delay at 30 Hz).
   - **Simulates**: Transport packet buffering, delayed execution pipelines, and out-of-sync command delivery.

4. **Virtual Gravity Sag / Angular Droop (`--virtual-gravity-sag-deg`)**:
   - Injects a continuous angular offset droop (in degrees) to target joint setpoints.
   - **Simulates**: Structural compliance, loose joint fasteners, or unexpected gravitational droop.

### Running Robust Evaluation:

```powershell
python examples/robust_safety_so101_eval.py `
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
*Outputs JSON metrics containing tracking errors before vs. after perturbation, success rates, and stability logs.*

---

## 🤖 Standard Real-Robot Deployment

To run the trained policy autonomously without perturbations:

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM6 `
    --robot.id=my_follower_arm `
    --robot.cameras="{top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
    --dataset.repo_id=huitaehan/eval_so101 `
    --dataset.single_task="Pick the object and place it" `
    --dataset.num_episodes=10 `
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model `
    --display_data=false `
    --play_sounds=false
```

---

## 🔄 Replay Demonstrations

To verify motor calibration and physically replay a recorded demonstration on the follower arm:

```powershell
lerobot-replay `
    --robot.type=so101_follower `
    --robot.port=COM6 `
    --robot.id=my_follower_arm `
    --dataset.repo_id=huitaehan/so101_demo `
    --dataset.root="$env:USERPROFILE\.cache\huggingface\lerobot\huitaehan\so101_demo" `
    --dataset.episode=0
```

---

## 📹 Collecting More Data

To append additional demonstrations to the dataset:

```powershell
lerobot-record `
    --resume=true `
    --robot.type=so101_follower `
    --robot.port=COM6 `
    --robot.id=my_follower_arm `
    --robot.cameras="{top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=my_leader_arm `
    --dataset.repo_id=huitaehan/so101_demo `
    --dataset.root="$env:USERPROFILE\.cache\huggingface\lerobot\huitaehan\so101_demo" `
    --dataset.single_task="Pick the object and place it" `
    --dataset.num_episodes=20 `
    --dataset.episode_time_s=30 `
    --dataset.reset_time_s=5 `
    --dataset.streaming_encoding=true `
    --dataset.encoder_threads=2 `
    --dataset.push_to_hub=false `
    --display_data=false `
    --play_sounds=false
```
- **Controls**: `n` (finish episode / skip reset), `q` (stop and save), `r` (re-record episode).
