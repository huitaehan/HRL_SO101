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

> [!IMPORTANT]
> **Do not install LeRobot independently.**
> This repository is a self-contained fork of LeRobot containing the full codebase along with custom SO-101 hardware abstraction and robust evaluation pipelines. Installing this repo installs LeRobot. Running `pip install lerobot` from PyPI will overwrite local modifications and cause dependency conflicts.

> [!NOTE]
> **Python Version Requirement**: Python **3.12 or newer** is required (`>=3.12`).

### 1. Clone the repository
```bash
git clone https://github.com/huitaehan/HRL_SO101.git
cd HRL_SO101
```

### 2. Install Dependencies

#### Option A: Recommended (Fast & Conflict-Free with `uv`)
LeRobot uses [`uv`](https://docs.astral.sh/uv/) and a lockfile (`uv.lock`) for reproducible dependency resolution.

```bash
# Install uv if you don't already have it
pip install uv

# Create a clean Python 3.12 virtual environment
uv venv --python 3.12

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Sync locked dependencies (training libs + Feetech servo driver for SO-101)
uv sync --extra training --extra feetech

# (Optional) If training on an NVIDIA GPU, install CUDA-enabled PyTorch:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Option B: Alternative (Using Conda & `pip`)
If you prefer Conda and standard `pip`:

```bash
# 1. Create a Python 3.12 environment
conda create -y -n lerobot python=3.12
conda activate lerobot

# 2. Install PyTorch with CUDA acceleration (for NVIDIA GPU)
# For NVIDIA GPU (CUDA 12.4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Or for CPU-only:
# pip install torch torchvision

# 3. Install this repo in editable mode with training libs + Feetech servo driver
# (Always wrap the extras in quotes to prevent PowerShell parsing issues)
pip install -e ".[training,feetech]"

# 4. Verify installation and CUDA availability
python -c "import torch, torchvision, cv2, serial, feetech_servo_sdk; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 🛠️ Troubleshooting Setup Issues

- **`Package requires Python >= 3.12`**: Check your version with `python --version`. Conda or system defaults often point to Python 3.10 or 3.11, which will fail. Create a fresh Python 3.12+ environment.
- **`The term '.[training,feetech]' is not recognized` (PowerShell)**: PowerShell treats square brackets `[` `]` as wildcard patterns. Always wrap the extra in quotes: `pip install -e ".[training,feetech]"`.
- **`Microsoft Visual C++ 14.0 or greater is required` (Windows)**: Install the [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to build native C/C++ packages like `feetech-servo-sdk`.
- **Infinite dependency resolution with pip**: Use **Option A (`uv sync`)** above, which resolves from `uv.lock` in seconds without backtracking on PyPI.

---

## 🏋️ Training (On GPU Server)

Train an **ACT (Action Chunking with Transformers)** policy on the dataset. The dataset will be automatically downloaded from Hugging Face:

```bash
lerobot-train \
    --dataset.repo_id=huitaehan/so101_demo \
    --policy.type=act \
    --output_dir=outputs/train/act_so101 \
    --job_name=act_so101 \
    --batch_size=8 \
    --steps=20000 \
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
