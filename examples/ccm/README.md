# Neural Contraction Metric (CCM) for SO-101 and Robotic Systems

This module implements **Neural Contraction Metrics** to solve for the Riemannian metric $M(x)$ that satisfies the Contraction Linear Matrix Inequality (LMI):

$$\dot{M}(x) + A(x)^T M(x) + M(x) A(x) \preceq -2\lambda M(x)$$

where:
- $x = [q, \dot{q}]^T \in \mathbb{R}^{2n}$ is the system state (12D for the 6-DOF SO-101).
- $A(x) = \frac{\partial f}{\partial x}$ is the Jacobian of the dynamics $\dot{x} = f(x)$.
- $\lambda > 0$ is the guaranteed exponential convergence rate.

---

## Benchmark Comparison: Classical SOS vs. Neural CCM

| Dimension | Classical SOS (Drake + MOSEK) | Neural CCM (`neural_ccm.py`) |
| :--- | :--- | :--- |
| **OS Support** | Linux / macOS only (no native Windows) | Runs natively on **Windows, Linux, macOS** |
| **Hardware** | Pure CPU + High RAM (Gigabytes required) | **Laptop CPU friendly** (runs in seconds) & GPU ready |
| **Scaling** | Combinatorial explosion ($\binom{n+d}{d}$). Chokes on 6-DOF ($n=12$). | Scales smoothly to 12D+ state space with an MLP |
| **Speed** | 10+ minutes to hours or Out Of Memory | **17.5 seconds** for 200 epochs on a laptop CPU |
| **Guarantees** | Exact polynomial proof | Empirical certification over sampled state manifold |

---

## 1. Training

To train the contraction metric model:

### A. 1-DOF Damped Pendulum (Quick Test, ~1 second on CPU)
```bash
python examples/ccm/neural_ccm.py --system pendulum --epochs 150
```
Saved checkpoint: `outputs/ccm/metric_pendulum.pt`

### B. 6-DOF SO-101 Arm (12D State Space, ~17 seconds on CPU)
```bash
python examples/ccm/neural_ccm.py --system so101 --epochs 250 --batch-size 64
```
Saved checkpoint: `outputs/ccm/metric_so101.pt`

---

## 2. Inspecting & Evaluating Results

Use `examples/ccm/eval_ccm.py` to evaluate any trained checkpoint.

```bash
# Evaluate SO-101 Arm Metric:
python examples/ccm/eval_ccm.py --checkpoint outputs/ccm/metric_so101.pt

# Evaluate Pendulum Metric:
python examples/ccm/eval_ccm.py --checkpoint outputs/ccm/metric_pendulum.pt
```

### What `eval_ccm.py` Reports:

1. **Strict Positive Definiteness Check ($M(x) \succ 0$):**
   * Confirms that $\lambda_{\min}(M(x)) > 0$ across thousands of unseen test states.
   * Reports the average and maximum condition number $\kappa(M(x)) = \frac{\lambda_{\max}}{\lambda_{\min}}$ to ensure the metric is well-conditioned.

2. **Contraction Condition Check ($R(x) \preceq 0$):**
   * Calculates the maximum eigenvalue of the contraction matrix $R(x)$.
   * When $\lambda_{\max}(R) \le 0$, trajectories are certified to contract exponentially.

3. **Riemannian Distance Simulation:**
   * Starts two neighboring trajectories $x_1(t)$ and $x_2(t)$.
   * Computes the differential Riemannian distance:
     $$d_M(t) = \sqrt{\delta x(t)^T M(x_1(t)) \delta x(t)}$$
   * Verifies that the Riemannian distance decays towards zero as time progresses ($d_M(t) \le d_M(0) e^{-\lambda t}$).

4. **Matrix Inspection:**
   * Displays the actual numeric values of the learned $12 \times 12$ matrix $M(x)$ at nominal robot configurations.

---

## Architecture Details

1. **Guaranteed Positive Definiteness ($M(x) \succ 0$):**
   The neural network outputs the lower-triangular entries of a Cholesky matrix $L(x)$ with positive diagonals (`softplus`). The metric is computed as:
   $$M(x) = L(x) L(x)^T + \epsilon I$$

2. **Directional Derivative ($\dot{M}(x)$):**
   Calculated using automatic differentiation along the velocity field $f(x)$:
   $$\dot{M}(x) = \sum_{k=1}^{2n} \frac{\partial M(x)}{\partial x_k} f_k(x)$$

3. **Loss Function:**
   $$R(x) = \dot{M}(x) + A(x)^T M(x) + M(x) A(x) + 2\lambda M(x)$$
   $$\mathcal{L} = \frac{1}{B} \sum_{k=1}^B \text{ReLU}\big(\lambda_{\max}(R(x_k))\big)^2 + \beta \text{Tr}(M(x_k))$$
