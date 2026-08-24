"""
Common evaluation metrics for MLP and DGCNN CFD models.

Target order:
    [HTC, wall_shear, pressure]

Metrics:
    MAE
    RMSE
    R^2

Important:
    Metrics should normally be calculated AFTER inverse normalization,
    so that errors are reported in the original physical units.

Examples:
    HTC        -> W/(m^2 K)
    wall shear -> Pa
    pressure   -> Pa

This module accepts either:
    NumPy arrays
    PyTorch tensors

Expected final target dimension:
    3 -> [HTC, wall_shear, pressure]

Examples:
    MLP:
        (N, 3)

    DGCNN:
        (B, 7000, 3)
"""

from __future__ import annotations

import numpy as np


# =====================================================================
# Constants
# =====================================================================

TARGET_NAMES = (
    "HTC",
    "wall_shear",
    "pressure",
)

NUM_TARGETS = 3


# =====================================================================
# Array conversion / validation
# =====================================================================

def _to_numpy(
    array,
) -> np.ndarray:
    """
    Convert NumPy array or PyTorch tensor to NumPy float64.

    PyTorch is intentionally not imported here so that this common
    metrics module depends only on NumPy.
    """

    # PyTorch tensor
    if hasattr(array, "detach"):
        array = (
            array
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        array,
        dtype=np.float64,
    )


def _validate_targets(
    y_true,
    y_pred,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate prediction and target arrays.

    Final dimension must be:
        3 -> [HTC, wall_shear, pressure]

    Leading dimensions may be arbitrary.

    Examples:
        (N, 3)
        (B, 7000, 3)
    """

    y_true = _to_numpy(
        y_true
    )

    y_pred = _to_numpy(
        y_pred
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"y_true = {y_true.shape}\n"
            f"y_pred = {y_pred.shape}"
        )

    if y_true.ndim < 2:
        raise ValueError(
            f"Targets must have at least 2 dimensions, "
            f"found shape {y_true.shape}."
        )

    if y_true.shape[-1] != NUM_TARGETS:
        raise ValueError(
            f"Expected final dimension {NUM_TARGETS} "
            f"[HTC, wall_shear, pressure], "
            f"found shape {y_true.shape}."
        )

    if not np.all(
        np.isfinite(y_true)
    ):
        raise ValueError(
            "y_true contains NaN or Inf."
        )

    if not np.all(
        np.isfinite(y_pred)
    ):
        raise ValueError(
            "y_pred contains NaN or Inf."
        )

    # Flatten all sample / point dimensions.
    y_true = y_true.reshape(
        -1,
        NUM_TARGETS,
    )

    y_pred = y_pred.reshape(
        -1,
        NUM_TARGETS,
    )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot calculate metrics on empty arrays."
        )

    return y_true, y_pred


# =====================================================================
# Individual metrics
# =====================================================================

def mae(
    y_true,
    y_pred,
) -> np.ndarray:
    """
    Mean Absolute Error for each target.

    Returns
    -------
    ndarray, shape (3,)

        [HTC_MAE, wall_shear_MAE, pressure_MAE]
    """

    y_true, y_pred = (
        _validate_targets(
            y_true,
            y_pred,
        )
    )

    return np.mean(
        np.abs(
            y_pred - y_true
        ),
        axis=0,
    )


def rmse(
    y_true,
    y_pred,
) -> np.ndarray:
    """
    Root Mean Squared Error for each target.

    Returns
    -------
    ndarray, shape (3,)

        [HTC_RMSE, wall_shear_RMSE, pressure_RMSE]
    """

    y_true, y_pred = (
        _validate_targets(
            y_true,
            y_pred,
        )
    )

    mse = np.mean(
        (
            y_pred
            - y_true
        ) ** 2,
        axis=0,
    )

    return np.sqrt(
        mse
    )


def r2_score(
    y_true,
    y_pred,
) -> np.ndarray:
    """
    R^2 score for each target.

    R^2 = 1 - SS_res / SS_tot

    Returns
    -------
    ndarray, shape (3,)

        [HTC_R2, wall_shear_R2, pressure_R2]

    Notes
    -----
    If a target has zero variance in y_true, R^2 is undefined.
    In that case NaN is returned for that target.
    """

    y_true, y_pred = (
        _validate_targets(
            y_true,
            y_pred,
        )
    )

    residual = (
        y_true
        - y_pred
    )

    ss_res = np.sum(
        residual ** 2,
        axis=0,
    )

    mean_true = np.mean(
        y_true,
        axis=0,
    )

    ss_tot = np.sum(
        (
            y_true
            - mean_true
        ) ** 2,
        axis=0,
    )

    result = np.full(
        NUM_TARGETS,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        ss_tot > 0
    )

    result[valid] = (
        1.0
        - ss_res[valid]
        / ss_tot[valid]
    )

    return result


# =====================================================================
# Combined evaluation
# =====================================================================

def calculate_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate all model evaluation metrics.

    Returns
    -------
    dict

    Example:
        {
            "HTC": {
                "MAE": ...,
                "RMSE": ...,
                "R2": ...
            },
            "wall_shear": {
                "MAE": ...,
                "RMSE": ...,
                "R2": ...
            },
            "pressure": {
                "MAE": ...,
                "RMSE": ...,
                "R2": ...
            }
        }
    """

    y_true, y_pred = (
        _validate_targets(
            y_true,
            y_pred,
        )
    )

    mae_values = np.mean(
        np.abs(
            y_pred - y_true
        ),
        axis=0,
    )

    rmse_values = np.sqrt(
        np.mean(
            (
                y_pred
                - y_true
            ) ** 2,
            axis=0,
        )
    )

    residual = (
        y_true - y_pred
    )

    ss_res = np.sum(
        residual ** 2,
        axis=0,
    )

    mean_true = np.mean(
        y_true,
        axis=0,
    )

    ss_tot = np.sum(
        (
            y_true
            - mean_true
        ) ** 2,
        axis=0,
    )

    r2_values = np.full(
        NUM_TARGETS,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        ss_tot > 0
    )

    r2_values[valid] = (
        1.0
        - ss_res[valid]
        / ss_tot[valid]
    )

    results = {}

    for i, target_name in enumerate(
        TARGET_NAMES
    ):

        results[target_name] = {
            "MAE": float(
                mae_values[i]
            ),
            "RMSE": float(
                rmse_values[i]
            ),
            "R2": float(
                r2_values[i]
            ),
        }

    return results


# =====================================================================
# Display
# =====================================================================

def print_metrics(
    metrics: dict,
) -> None:
    """
    Print metric results in a compact table.
    """

    print("=" * 72)
    print("MODEL EVALUATION METRICS")
    print("=" * 72)

    print(
        f"{'Target':<16}"
        f"{'MAE':>16}"
        f"{'RMSE':>16}"
        f"{'R^2':>16}"
    )

    print("-" * 72)

    for target_name in TARGET_NAMES:

        values = metrics[
            target_name
        ]

        print(
            f"{target_name:<16}"
            f"{values['MAE']:>16.6f}"
            f"{values['RMSE']:>16.6f}"
            f"{values['R2']:>16.6f}"
        )

    print("=" * 72)


# =====================================================================
# Save results
# =====================================================================

def save_metrics(
    metrics: dict,
    path,
) -> None:
    """
    Save metrics as a human-readable text file.
    """

    from pathlib import Path

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = []

    lines.append(
        "CFD MODEL EVALUATION METRICS"
    )

    lines.append(
        "=" * 64
    )

    lines.append(
        f"{'Target':<16}"
        f"{'MAE':>16}"
        f"{'RMSE':>16}"
        f"{'R^2':>16}"
    )

    lines.append(
        "-" * 64
    )

    for target_name in TARGET_NAMES:

        values = metrics[
            target_name
        ]

        lines.append(
            f"{target_name:<16}"
            f"{values['MAE']:>16.6f}"
            f"{values['RMSE']:>16.6f}"
            f"{values['R2']:>16.6f}"
        )

    lines.append(
        "=" * 64
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =====================================================================
# Standalone self-test
# =====================================================================

def main():
    """
    Internal metric calculation test.

    This uses synthetic data only.
    """

    print("=" * 72)
    print("CFD METRICS SELF-TEST")
    print("=" * 72)

    y_true = np.array(
        [
            [30.0, 0.10, 12.0],
            [50.0, 0.20, 18.0],
            [70.0, 0.30, 24.0],
            [90.0, 0.40, 30.0],
        ],
        dtype=np.float64,
    )

    y_pred = np.array(
        [
            [32.0, 0.11, 13.0],
            [48.0, 0.18, 16.0],
            [74.0, 0.33, 27.0],
            [87.0, 0.39, 29.0],
        ],
        dtype=np.float64,
    )

    results = calculate_metrics(
        y_true,
        y_pred,
    )

    print()
    print_metrics(
        results
    )

    # -------------------------------------------------------------
    # Independent manual checks
    # -------------------------------------------------------------

    expected_htc_mae = (
        2.0
        + 2.0
        + 4.0
        + 3.0
    ) / 4.0

    expected_shear_mae = (
        0.01
        + 0.02
        + 0.03
        + 0.01
    ) / 4.0

    if not np.isclose(
        results["HTC"]["MAE"],
        expected_htc_mae,
    ):
        raise RuntimeError(
            "HTC MAE self-test failed."
        )

    if not np.isclose(
        results["wall_shear"]["MAE"],
        expected_shear_mae,
    ):
        raise RuntimeError(
            "Wall-shear MAE self-test failed."
        )

    expected_pressure_mae = (
        1.0
        + 2.0
        + 3.0
        + 1.0
    ) / 4.0

    if not np.isclose(
        results["pressure"]["MAE"],
        expected_pressure_mae,
    ):
        raise RuntimeError(
            "Pressure MAE self-test failed."
        )

    # Perfect prediction test
    perfect = calculate_metrics(
        y_true,
        y_true,
    )

    if not np.isclose(
        perfect["HTC"]["MAE"],
        0.0,
    ):
        raise RuntimeError(
            "Perfect-prediction MAE test failed."
        )

    if not np.isclose(
        perfect["HTC"]["RMSE"],
        0.0,
    ):
        raise RuntimeError(
            "Perfect-prediction RMSE test failed."
        )

    if not np.isclose(
        perfect["HTC"]["R2"],
        1.0,
    ):
        raise RuntimeError(
            "Perfect-prediction R2 test failed."
        )

    if not np.isclose(
        perfect["wall_shear"]["R2"],
        1.0,
    ):
        raise RuntimeError(
            "Perfect wall-shear R2 test failed."
        )

    if not np.isclose(
        perfect["pressure"]["R2"],
        1.0,
    ):
        raise RuntimeError(
            "Perfect pressure R2 test failed."
        )

    print()
    print("MAE calculation     : PASS")
    print("RMSE calculation    : PASS")
    print("R^2 calculation     : PASS")
    print("Perfect prediction  : PASS")

    print()
    print("=" * 72)
    print("METRICS SELF-TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()