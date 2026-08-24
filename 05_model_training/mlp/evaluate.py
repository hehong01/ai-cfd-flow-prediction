"""
Evaluate a trained point-wise MLP on the held-out CFD test set.

Input per CFD wall node:
    [x, y, z, velocity]

Target per CFD wall node:
    [HTC, wall_shear, pressure]

Test split:
    face_0091 ~ face_0100
    10 faces x 3 velocities = 30 CSV files

Workflow:
    1. Load best_model.pt
    2. Load training scaler
    3. Load TEST CSV files only
    4. Apply saved TRAIN normalization
    5. Run MLP inference
    6. Inverse-transform predictions
    7. Calculate MAE / RMSE / R^2 in physical units

No training or weight updates are performed here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


# =====================================================================
# Project paths
# =====================================================================

THIS_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = THIS_DIR.parent
GITHUB_ROOT = TRAINING_ROOT.parent
COMMON_DIR = TRAINING_ROOT / "common"

for path in (
    GITHUB_ROOT,
    TRAINING_ROOT,
    COMMON_DIR,
    THIS_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


from project_paths import DATA_ROOT

from model import (
    PointwiseMLP,
    count_parameters,
)

from scalers import CFDScaler

from metrics import (
    calculate_metrics,
    print_metrics,
)

from train import (
    load_split,
    build_loader,
    predict_loader,
    get_device,
)


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_BATCH_SIZE = 8192

WEIGHT_DIR = (
    TRAINING_ROOT
    / "weights"
    / "mlp"
)

DEFAULT_MODEL_PATH = (
    WEIGHT_DIR
    / "best_model.pt"
)

DEFAULT_SCALER_PATH = (
    WEIGHT_DIR
    / "scalers.npz"
)


# =====================================================================
# Load trained model
# =====================================================================

def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[PointwiseMLP, dict]:
    """
    Load a saved PointwiseMLP checkpoint.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    required = {
        "model_name",
        "input_dim",
        "hidden_dims",
        "output_dim",
        "epoch",
        "val_loss",
        "model_state_dict",
    }

    missing = (
        required
        - set(checkpoint.keys())
    )

    if missing:
        raise ValueError(
            "Model checkpoint is missing fields: "
            f"{sorted(missing)}"
        )

    if checkpoint["model_name"] != "PointwiseMLP":
        raise ValueError(
            "Unexpected model type in checkpoint: "
            f"{checkpoint['model_name']}"
        )

    if int(checkpoint["output_dim"]) != 3:
        raise ValueError(
            "Legacy or incompatible MLP checkpoint: "
            f"expected output_dim=3 "
            f"[HTC, wall_shear, pressure], "
            f"found output_dim={checkpoint['output_dim']}."
        )

    model = PointwiseMLP(
        input_dim=int(
            checkpoint["input_dim"]
        ),
        hidden_dims=tuple(
            checkpoint["hidden_dims"]
        ),
        output_dim=int(
            checkpoint["output_dim"]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model, checkpoint


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(args) -> None:

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------

    data_root = (
        Path(args.data_root)
        if args.data_root is not None
        else Path(DATA_ROOT)
    ).resolve()

    csv_dir = (
        data_root
        / "05_cfd_csv"
    )

    model_path = Path(
        args.model_path
    ).resolve()

    scaler_path = Path(
        args.scaler_path
    ).resolve()

    if not csv_dir.exists():
        raise FileNotFoundError(
            f"CFD CSV directory not found:\n"
            f"{csv_dir}"
        )

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = get_device()

    print("=" * 78)
    print("POINT-WISE MLP TEST EVALUATION")
    print("=" * 78)

    print(
        f"Data root      : {data_root}"
    )

    print(
        f"CSV directory  : {csv_dir}"
    )

    print(
        f"Model          : {model_path}"
    )

    print(
        f"Scaler         : {scaler_path}"
    )

    print(
        f"Device         : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"PyTorch        : {torch.__version__}"
    )

    print(
        f"Batch size     : {args.batch_size}"
    )

    print("=" * 78)

    # =================================================================
    # Load scaler
    # =================================================================

    print()
    print("[LOAD SCALER]")

    scaler = CFDScaler.load(
        scaler_path
    )

    scaler.print_statistics()

    # =================================================================
    # Load trained model
    # =================================================================

    print()
    print("[LOAD MODEL]")

    model, checkpoint = load_model(
        model_path,
        device,
    )

    print(model)

    print(
        f"Trainable parameters : "
        f"{count_parameters(model):,}"
    )

    print(
        f"Saved best epoch     : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Saved val loss       : "
        f"{checkpoint['val_loss']:.8f}"
    )

    # =================================================================
    # Load TEST data
    # =================================================================

    print()
    print("=" * 78)
    print("LOAD HELD-OUT TEST SET")
    print("=" * 78)

    x_test, y_test = load_split(
        csv_dir,
        "test",
    )

    print()
    print(
        f"Test points : {len(x_test):,}"
    )

    # =================================================================
    # Apply TRAIN scaler
    # =================================================================

    print()
    print("[NORMALIZE TEST DATA]")

    x_test_norm = (
        scaler
        .transform_input(
            x_test
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    y_test_norm = (
        scaler
        .transform_target(
            y_test
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    print(
        "Test normalization complete."
    )

    # Original arrays are no longer needed.
    del x_test
    del y_test

    # =================================================================
    # DataLoader
    # =================================================================

    test_loader = build_loader(
        x_test_norm,
        y_test_norm,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )

    del x_test_norm
    del y_test_norm

    # =================================================================
    # Inference
    # =================================================================

    print()
    print("=" * 78)
    print("TEST INFERENCE")
    print("=" * 78)

    start = time.perf_counter()

    y_true_norm, y_pred_norm = (
        predict_loader(
            model,
            test_loader,
            device,
        )
    )

    inference_time = (
        time.perf_counter()
        - start
    )

    if y_true_norm.ndim != 2 or y_true_norm.shape[1] != 3:
        raise RuntimeError(
            "Unexpected normalized target shape: "
            f"{y_true_norm.shape}. "
            "Expected (N, 3) for "
            "[HTC, wall_shear, pressure]."
        )

    if y_pred_norm.shape != y_true_norm.shape:
        raise RuntimeError(
            "Prediction/target shape mismatch: "
            f"prediction={y_pred_norm.shape}, "
            f"target={y_true_norm.shape}."
        )

    print(
        f"Inference complete : "
        f"{len(y_pred_norm):,} points"
    )

    print(
        f"Inference runtime  : "
        f"{inference_time:.2f} s"
    )

    # =================================================================
    # Restore physical units
    # =================================================================

    y_true = scaler.inverse_target(
        y_true_norm
    )

    y_pred = scaler.inverse_target(
        y_pred_norm
    )

    # =================================================================
    # Final TEST metrics
    # =================================================================

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print()
    print("=" * 78)
    print("HELD-OUT TEST RESULTS")
    print("=" * 78)

    print_metrics(
        metrics
    )

    # =================================================================
    # Summary
    # =================================================================

    print()
    print("=" * 78)
    print("MLP TEST EVALUATION COMPLETE")
    print("=" * 78)

    print(
        "Test faces       : "
        "face_0091 ~ face_0100"
    )

    print(
        "Test CSV files   : 30"
    )

    print(
        f"Test points      : "
        f"{len(y_pred):,}"
    )

    print(
        f"Checkpoint epoch : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation loss  : "
        f"{checkpoint['val_loss']:.8f}"
    )

    print("=" * 78)


# =====================================================================
# CLI
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained PointwiseMLP "
            "on the held-out CFD test set."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional ai-cfd-data root. "
            "Default: project_paths.DATA_ROOT"
        ),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=DEFAULT_SCALER_PATH,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    return parser.parse_args()


# =====================================================================
# Entry point
# =====================================================================

def main():

    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    evaluate(
        args
    )


if __name__ == "__main__":
    main()