"""
Point-wise MLP baseline for CFD surface-field prediction.

For each surface point:

    Input:
        [x, y, z, velocity]

    Output:
        [HTC, wall_shear]

The MLP treats every surface point independently.

Therefore, unlike DGCNN, this model does not explicitly use
neighboring-point or point-cloud geometry information.

This model is intended as a simple baseline for comparison
with the geometry-aware DGCNN model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# =====================================================================
# Model
# =====================================================================

class PointwiseMLP(nn.Module):
    """
    Point-wise multilayer perceptron.

    Default architecture:

        4
        ↓
        128
        ↓
        128
        ↓
        64
        ↓
        2

    Input features:
        0 -> x
        1 -> y
        2 -> z
        3 -> inlet velocity

    Output targets:
        0 -> HTC
        1 -> wall shear

    The model operates on the final tensor dimension.

    Therefore both of these are valid:

        (N, 4)
        (B, N, 4)

    and produce:

        (N, 2)
        (B, N, 2)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: tuple[int, ...] = (
            128,
            128,
            64,
        ),
        output_dim: int = 2,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be positive."
            )

        if not hidden_dims:
            raise ValueError(
                "hidden_dims must contain at least one layer."
            )

        if any(
            dim <= 0
            for dim in hidden_dims
        ):
            raise ValueError(
                "All hidden dimensions must be positive."
            )

        # -------------------------------------------------------------
        # Build network
        # -------------------------------------------------------------

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.ReLU()
            )

            previous_dim = hidden_dim

        # Final regression layer.
        #
        # No activation is applied because HTC and wall shear
        # are continuous regression targets.
        layers.append(
            nn.Linear(
                previous_dim,
                output_dim,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Tensor whose final dimension is 4:

                [x, y, z, velocity]

        Returns
        -------
        Tensor whose final dimension is 2:

                [HTC, wall_shear]
        """

        if x.ndim < 2:
            raise ValueError(
                f"Expected at least 2D input, "
                f"found shape {tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected final input dimension "
                f"{self.input_dim}, "
                f"found shape {tuple(x.shape)}."
            )

        return self.network(x)


# =====================================================================
# Utilities
# =====================================================================

def count_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =====================================================================
# Standalone self-test
# =====================================================================

def main():

    print("=" * 72)
    print("POINT-WISE MLP SELF-TEST")
    print("=" * 72)

    model = PointwiseMLP()

    print()
    print(model)

    print()
    print(
        f"Trainable parameters : "
        f"{count_parameters(model):,}"
    )

    # -------------------------------------------------------------
    # Test 1:
    # ordinary point-wise input
    # -------------------------------------------------------------

    x = torch.randn(
        32,
        4,
        dtype=torch.float32,
    )

    y = model(x)

    print()
    print("[POINT BATCH]")
    print(
        f"Input shape  : {tuple(x.shape)}"
    )
    print(
        f"Output shape : {tuple(y.shape)}"
    )

    if y.shape != (
        32,
        2,
    ):
        raise RuntimeError(
            "Point-batch output shape test failed."
        )

    # -------------------------------------------------------------
    # Test 2:
    # grouped point-cloud-shaped input
    #
    # Linear layers operate on the last dimension,
    # so the same model can also process this shape.
    # -------------------------------------------------------------

    x_grouped = torch.randn(
        2,
        7000,
        4,
        dtype=torch.float32,
    )

    y_grouped = model(
        x_grouped
    )

    print()
    print("[GROUPED POINTS]")
    print(
        f"Input shape  : "
        f"{tuple(x_grouped.shape)}"
    )
    print(
        f"Output shape : "
        f"{tuple(y_grouped.shape)}"
    )

    if y_grouped.shape != (
        2,
        7000,
        2,
    ):
        raise RuntimeError(
            "Grouped-input output shape test failed."
        )

    # -------------------------------------------------------------
    # Numerical check
    # -------------------------------------------------------------

    if not torch.isfinite(
        y
    ).all():
        raise RuntimeError(
            "Model output contains NaN or Inf."
        )

    if not torch.isfinite(
        y_grouped
    ).all():
        raise RuntimeError(
            "Grouped model output contains NaN or Inf."
        )

    print()
    print("Point input test   : PASS")
    print("Grouped input test : PASS")
    print("Finite output test : PASS")

    print()
    print("=" * 72)
    print("MLP MODEL SELF-TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()