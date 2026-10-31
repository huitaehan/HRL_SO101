r"""
SmolVLA + Implicit Contraction Layer (ICL) Integration Verification Script.

This test runs purely on CPU and requires ONLY PyTorch and NumPy (no GPU, no MuJoCo needed).
It verifies:
1. `implicit_contraction_layers` is accessible in `c:\Users\Huitae\lerobot\implicit_contraction_layers`.
2. `SmolVLAAdapter` formats observations and enforces action space bounds.
3. The differentiable QP contraction layer (`SafeBoxHalfspaceQP`) successfully projects
   actions to satisfy the contraction halfspace constraint.

Usage:
    conda activate lerobot
    python examples/smolvla/test_smolvla_icl_integration.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Ensure workspace and implicit_contraction_layers/src are in Python path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ICL_SRC = WORKSPACE_ROOT / "implicit_contraction_layers" / "src"

if str(WORKSPACE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
if str(ICL_SRC) not in sys.path:
    sys.path.insert(0, str(ICL_SRC))

def main():
    print("=" * 70)
    print(" SmolVLA + Implicit Contraction Layers (ICL) Unification Test")
    print(" (Pure CPU / PyTorch - No GPU or MuJoCo Required)")
    print("=" * 70)

    # 1. Test Imports
    print("\n[1/3] Testing Module Imports...")
    try:
        from lerobot.policies.smolvla.adapter_icl import SmolVLAAdapter
        print("  [OK] Successfully imported `SmolVLAAdapter` from `lerobot.policies.smolvla.adapter_icl`")
    except Exception as e:
        print(f"  [FAIL] Failed to import SmolVLAAdapter: {e}")
        return

    try:
        # Import the pure math QP projection from ICL
        from icl_core.qp import solve_box_halfspace, solution_jacobian
        print("  [OK] Successfully imported `solve_box_halfspace` & `solution_jacobian` from `icl_core.qp`")
    except Exception as e:
        print(f"  [FAIL] Failed to import icl_core.qp: {e}")
        return

    # 2. Test Adapter Behavior
    print("\n[2/3] Testing SmolVLAAdapter with Mock Policy...")
    class MockPolicy(nn.Module):
        """Mock policy simulating raw unconstrained action proposals."""
        def select_action(self, batch):
            state = batch["observation.state"]
            batch_size = state.shape[0]
            # Proposes an extreme action: [2.5, -2.5]
            return torch.full((batch_size, 2), 2.5)

    adapter = SmolVLAAdapter(
        policy=MockPolicy(),
        action_dim=2,
        action_lower=[-1.0, -1.0],
        action_upper=[1.0, 1.0],
        device="cpu",
    )

    test_state = np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float32)
    bounded_proposal = adapter.act(test_state)
    print(f"  Input State       : {test_state}")
    print(f"  Adapter Output    : {bounded_proposal} (Clamped to action limits [-1.0, 1.0])")
    assert np.allclose(bounded_proposal, [1.0, 1.0]), "Adapter clamping failed!"
    print("  [OK] Adapter clamped proposal correctly within bounds.")

    # 3. Test QP Projection Layer (ICL Engine)
    print("\n[3/3] Testing Differentiable QP Projection Layer...")
    # SafeBoxHalfspaceQP definition using icl_core.qp
    class SafeBoxHalfspaceQP(nn.Module):
        def __init__(self, lower, upper, weights, tolerance=1e-5):
            super().__init__()
            self.lower = torch.as_tensor(lower, dtype=torch.float32)
            self.upper = torch.as_tensor(upper, dtype=torch.float32)
            self.weights = torch.as_tensor(weights, dtype=torch.float32)
            self.tolerance = tolerance

        def forward(self, proposal, normal, bound):
            action, feasible, _ = solve_box_halfspace(
                proposal, normal, bound, self.lower, self.upper, self.weights, self.tolerance
            )
            return action, feasible

    qp_layer = SafeBoxHalfspaceQP(
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        weights=[1.0, 1.0],
    )

    # Let proposal = [1.0, 1.0].
    # Contraction constraint: normal^T u <= bound => 1.0 * u[0] <= 0.2
    proposal_t = torch.tensor(bounded_proposal, dtype=torch.float32)
    normal_t = torch.tensor([1.0, 0.0], dtype=torch.float32)
    bound_t = torch.tensor(0.2, dtype=torch.float32)

    projected_action, feasible = qp_layer(proposal_t, normal_t, bound_t)
    projected_np = projected_action.numpy()
    print(f"  Pre-QP Proposal   : {proposal_t.numpy()}")
    print(f"  Contraction Bound : normal^T u <= {bound_t.item()}")
    print(f"  Post-QP Action    : {projected_np} (u[0] projected to 0.2, u[1] remains 1.0)")

    assert feasible, "QP should be feasible!"
    assert projected_np[0] <= 0.2 + 1e-4, f"QP constraint violated: {projected_np[0]}"
    assert np.isclose(projected_np[1], 1.0), f"Unconstrained dimension altered: {projected_np[1]}"
    print("  [OK] QP Layer successfully projected action to satisfy contraction constraint!")

    print("\n" + "=" * 70)
    print(" ALL TESTS PASSED: SmolVLAAdapter and ICL QP Engine are Unified!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
