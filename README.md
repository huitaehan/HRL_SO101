# HRL_SO101

Repository for training, evaluating, and deploying imitation learning policies on the **SO-101 6-DoF robotic arm** using [LeRobot](https://github.com/huggingface/lerobot), with custom **non-stationary dynamics and robust safety evaluation**, and **Neural Contraction Metrics (CCM)**.

---

## 🛠️ Hardware Setup

- **Follower Arm**: SO-101 (`so101_follower` with Feetech STS3215 bus)
- **Leader Arm / Teleoperator**: SO-101 Leader (`so101_leader` with Feetech STS3215 bus)
- **Camera**: Top-mounted camera (`opencv`, index `1`, 640x480 @ 30 FPS)
- **Default Ports (Windows)**:
  - Follower: `COM8`
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

## 📂 Modular Pipelines

This repository is organized into self-contained workflow modules under `examples/`:

| Module | Description | Guide |
|---|---|---|
| **SmolVLA** | Vision-Language-Action foundation model (`smolvla_base`) post-training & SO-101 deployment | [`examples/smolvla/README.md`](examples/smolvla/README.md) |
| **Implicit Contraction Layers (ICL)** | Differentiable QP & Contraction Metrics for robust tracking on SO-101 ([Kim, Han, Sreenath, 2027]) | [Section Below](#-implicit-contraction-layers-icl--smolvla-integration) |
| **Safety Filters** | Robust safety evaluation, perturbations, and Control Barrier Function (CBF) | [`examples/safety/README.md`](examples/safety/README.md) |
| **CCM Pipeline** | Neural Contraction Control Metrics & SO-101 dynamics | [`examples/ccm/README.md`](examples/ccm/README.md) |

---

## 🛡️ Implicit Contraction Layers (ICL) & SmolVLA Integration

Implementation and integration of **"Robust Reinforcement Learning-based Trajectory Tracking through Implicit Contraction Layers"** (*Mintae Kim, Huitae Han, Koushil Sreenath, 2027*).

SmolVLA acts as the high-level multimodal policy (predicting nominal 5-DoF / 6-DoF joint actions), while the **Implicit Contraction Layer (ICL)** serves as the differentiable physical safety & tracking engine:

```text
Camera + Prompt ──► SmolVLA Policy ──► Proposed Action v
                                             │
                                             ▼
Physical SO-101 ◄── Safe Action u* ◄── Differentiable QP Layer (ICL)
Feedback x in R^10                      Enforcing: a(z)^T u <= b(z), u in [-1, 1]^5
                                        Contraction Rate: rho = 0.98, R = I_5
```

### Key Components & File Locations:

| Component | File Path | Description |
| :--- | :--- | :--- |
| **SmolVLA Adapter** | [`src/lerobot/policies/smolvla/adapter_icl.py`](src/lerobot/policies/smolvla/adapter_icl.py) | Formats observations, bounds actions, and manages gripper modes (`open` / `fixed_grasp`). |
| **Differentiable QP** | [`implicit_contraction_layers/src/icl_core/qp.py`](implicit_contraction_layers/src/icl_core/qp.py) | Closed-form analytical active-set QP solver (`solve_box_halfspace`) with exact backward gradients. |
| **Contraction Layer** | [`implicit_contraction_layers/src/icl_core/layer.py`](implicit_contraction_layers/src/icl_core/layer.py) | Projects action proposals to strictly satisfy the contraction inequality $\dot{E} \le -2\lambda E$. |
| **SO-101 5-DoF Task** | [`implicit_contraction_layers/src/icl_core/so101.py`](implicit_contraction_layers/src/icl_core/so101.py) | 5-DoF environment matching paper Appendix B ($dq = [0.25, 0.25, 0.25, 0.25, 0.35]\text{ rad}$, disturbance pulse). |
| **Experiment Config** | [`implicit_contraction_layers/experiments/so101_reach/config.yaml`](implicit_contraction_layers/experiments/so101_reach/config.yaml) | Hyperparameters matching paper Table 2 ($\rho=0.98, R=I_5, \underline{m}=0.05, \bar{m}=15.25$). |
| **Integration Test** | [`examples/smolvla/test_smolvla_icl_integration.py`](examples/smolvla/test_smolvla_icl_integration.py) | Pure CPU sanity test verifying SmolVLAAdapter + ICL QP unification. |
| **Interactive Sim** | [`examples/smolvla/simulate_smolvla_icl.py`](examples/smolvla/simulate_smolvla_icl.py) | Closed-loop simulator with live 3D OpenGL window (`--render`) and MP4 video export (`--save-video`). |
| **Paper Benchmark** | [`examples/smolvla/evaluate_so101_paper_pipeline.py`](examples/smolvla/evaluate_so101_paper_pipeline.py) | Replicates paper Table 1 evaluation (Recovery %, Fallback %, $t_{95}$ latency, peak error) on sim or hardware. |

### Quickstart Commands:

```bash
# 1. Verify SmolVLA + ICL Unification (Pure CPU, no hardware needed):
python examples/smolvla/test_smolvla_icl_integration.py

# 2. Run closed-loop MuJoCo simulation with live 3D window:
python examples/smolvla/simulate_smolvla_icl.py --env reacher --render --steps 200

# 3. Run the paper's exact SO-101 disturbance benchmark (Table 1 reproduction):
python examples/smolvla/evaluate_so101_paper_pipeline.py --task reach --episodes 5

# 4. Evaluate with learned ICL checkpoint (when layer/best.pt is available):
python examples/smolvla/evaluate_so101_paper_pipeline.py --checkpoint path/to/layer/best.pt

# 5. Deploy directly to physical SO-101 robot arm:
python examples/smolvla/evaluate_so101_paper_pipeline.py --mode hardware --port COM3
```

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

## 📂 Repository Modules & Custom Extensions

This repository enhances standard LeRobot with three separated modules:

```
examples/
├── safety/       # 🛡️ Real-robot perturbation & non-stationary dynamics evaluation
├── ccm/          # 🔬 Neural Contraction Metrics (CCM) solver & Riemannian stability
└── smolvla/      # 🤖 SmolVLA multimodal Vision-Language-Action training & deployment
```

- [🛡️ Safety & Disturbance Evaluation](./examples/safety/README.md) — Real-world evaluation under hardware torque loss, latency, and queue lag.
- [🔬 Neural Contraction Metrics (CCM)](./examples/ccm/README.md) — Solves for metric $M(x)$ satisfying the contraction LMI with exponential convergence guarantees.
- [🤖 SmolVLA Policy](./examples/smolvla/README.md) — Compact multimodal VLA policy leveraging SmolVLM and flow-matching action generation.

---

## 🛡️ Robust Safety & Non-Stationary Dynamics Evaluation

We implemented custom **safety and disturbance injection capabilities** in [`examples/safety/robust_safety_so101_eval.py`](./examples/safety/robust_safety_so101_eval.py) and [`src/lerobot/robots/so_follower/config_so_follower.py`](./src/lerobot/robots/so_follower/config_so_follower.py).

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
*Outputs JSON metrics containing tracking errors before vs. after perturbation, success rates, and stability logs.*

---

## 🔬 Contraction Theory & Neural CCM Research

We implemented **Neural Contraction Metrics (CCM)** for the SO-101 6-DoF arm to mathematically guarantee robust trajectory tracking and exponential convergence under non-linear disturbances and model shifts.

- 📖 **Documentation & Full 9-Step Pipeline**: [`examples/ccm/README.md`](./examples/ccm/README.md)
- 🧠 **Training Script**: [`examples/ccm/neural_ccm.py`](./examples/ccm/neural_ccm.py)
- 🔍 **Evaluation & Verification**: [`examples/ccm/eval_ccm.py`](./examples/ccm/eval_ccm.py)
- 📊 **Metric Visualization**: [`examples/ccm/visualize_ccm.py`](./examples/ccm/visualize_ccm.py)

### Quick Commands (Runs on Laptop CPU):
```bash
# Train Neural CCM for 6-DOF SO-101 (12D state space, ~17s on CPU):
python examples/ccm/neural_ccm.py --system so101 --epochs 250 --batch-size 64

# Evaluate positive-definiteness M(x) > 0 and verify contraction R(x) <= 0:
python examples/ccm/eval_ccm.py --checkpoint outputs/ccm/metric_so101.pt

# Visualize 12x12 metric heatmaps and Riemannian convergence curves:
python examples/ccm/visualize_ccm.py --checkpoint outputs/ccm/metric_so101.pt
```

---

## 🤖 SmolVLA: Vision-Language-Action Policy

SmolVLA brings compact multimodal instruction-following policies to the SO-101 arm, pairing a lightweight SmolVLM backbone with a flow-matching action expert:

- 📖 **Documentation & Architecture Guide**: [`examples/smolvla/README.md`](./examples/smolvla/README.md)
- 🏗️ **Policy Code**: [`src/lerobot/policies/smolvla/`](./src/lerobot/policies/smolvla/)

### Training SmolVLA on SO-101:
```bash
lerobot-train \
    --dataset.repo_id=huitaehan/so101_demo \
    --policy.type=smolvla \
    --output_dir=outputs/train/smolvla_so101 \
    --job_name=smolvla_so101 \
    --batch_size=8 \
    --steps=20000 \
    --wandb.enable=false
```

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
