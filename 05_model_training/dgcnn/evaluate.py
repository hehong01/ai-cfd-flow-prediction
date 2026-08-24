"""
Evaluate a trained DGCNN CFD regression model.

This script does NOT train the model.

It loads:

    weights/dgcnn/best_model.pt
    weights/dgcnn/scalers.npz

and evaluates the model only on the held-out TEST split:

    face_0091 ~ face_0100
    10 faces x 3 velocities
    =
    30 samples

Input per point:
    [x, y, z, velocity]

Target per point:
    [HTC, wall_shear, pressure]

Evaluation metrics are calculated after inverse normalization,
therefore the reported errors are in physical units.
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
    THIS_DIR,
    COMMON_DIR,
    TRAINING_ROOT,
    GITHUB_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


from project_paths import DATA_ROOT

from model import (
    DGCNNRegressor,
    count_parameters,
)

from scalers import CFDScaler

from metrics import (
    calculate_metrics,
    print_metrics,
)

# Reuse the already validated training-side data utilities.
from train import (
    load_dataset_module,
    create_base_dataset,
    extract_xy,
    NormalizedDGCNNDataset,
    build_loader,
    predict_loader,
)


# =====================================================================
# Defaults
# =====================================================================

EXPECTED_TEST_SAMPLES = 30

DEFAULT_BATCH_SIZE = 1

DEFAULT_MODEL_PATH = (
    TRAINING_ROOT
    / "weights"
    / "dgcnn"
    / "best_model.pt"
)

DEFAULT_SCALER_PATH = (
    TRAINING_ROOT
    / "weights"
    / "dgcnn"
    / "scalers.npz"
)


# =====================================================================
# Device
# =====================================================================

def get_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# =====================================================================
# Main evaluation
# =====================================================================

def evaluate(
    args,
) -> None:

    # -----------------------------------------------------------------
    # Resolve paths
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
            "CFD CSV directory not found:\n"
            f"{csv_dir}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            "DGCNN checkpoint not found:\n"
            f"{model_path}\n\n"
            "Run dgcnn/train.py first."
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            "DGCNN scaler not found:\n"
            f"{scaler_path}\n\n"
            "Run dgcnn/train.py first."
        )

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = get_device()

    print("=" * 78)
    print("DGCNN CFD TEST EVALUATION")
    print("=" * 78)

    print(
        f"Data root       : "
        f"{data_root}"
    )

    print(
        f"CSV directory   : "
        f"{csv_dir}"
    )

    print(
        f"Model           : "
        f"{model_path}"
    )

    print(
        f"Scaler          : "
        f"{scaler_path}"
    )

    print(
        f"Device          : "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"PyTorch         : "
        f"{torch.__version__}"
    )

    print(
        f"Batch size      : "
        f"{args.batch_size}"
    )

    if args.point_limit is None:

        print(
            "Point limit     : "
            "None (full FPS point cloud)"
        )

    else:

        print(
            f"Point limit     : "
            f"{args.point_limit} "
            "(DEBUG ONLY)"
        )

    print("=" * 78)

    # =================================================================
    # Load checkpoint
    # =================================================================

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = (
        "model_state_dict",
        "input_dim",
        "output_dim",
        "k",
        "epoch",
        "val_loss",
        "first_knn_space",
    )

    for key in required_keys:

        if key not in checkpoint:
            raise KeyError(
                "Checkpoint is missing required key: "
                f"{key}"
            )

    checkpoint_input_dim = int(
        checkpoint[
            "input_dim"
        ]
    )

    checkpoint_output_dim = int(
        checkpoint[
            "output_dim"
        ]
    )

    if checkpoint_output_dim != 3:
        raise RuntimeError(
            "Legacy or incompatible DGCNN checkpoint: "
            f"expected output_dim=3 "
            f"[HTC, wall_shear, pressure], "
            f"found output_dim={checkpoint_output_dim}."
        )

    checkpoint_k = int(
        checkpoint[
            "k"
        ]
    )

    checkpoint_chunk_size = int(
        checkpoint.get(
            "knn_chunk_size",
            1024,
        )
    )

    checkpoint_point_count = (
        checkpoint.get(
            "point_count",
            None,
        )
    )

    checkpoint_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    checkpoint_val_loss = float(
        checkpoint[
            "val_loss"
        ]
    )

    checkpoint_first_knn_space = str(
        checkpoint[
            "first_knn_space"
        ]
    )

    if checkpoint_first_knn_space != "raw_xyz":
        raise RuntimeError(
            "Unsupported first k-NN graph space in checkpoint: "
            f"{checkpoint_first_knn_space!r}. "
            "Expected 'raw_xyz'."
        )

    # -----------------------------------------------------------------
    # kNN chunk size does not alter learned weights or architecture.
    #
    # Therefore evaluation may override it for the current hardware.
    # -----------------------------------------------------------------

    if args.knn_chunk_size is None:

        evaluation_chunk_size = (
            checkpoint_chunk_size
        )

    else:

        evaluation_chunk_size = (
            args.knn_chunk_size
        )

    print()
    print("=" * 78)
    print("CHECKPOINT")
    print("=" * 78)

    print(
        f"Model name      : "
        f"{checkpoint.get('model_name', 'unknown')}"
    )

    print(
        f"Best epoch      : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Validation loss : "
        f"{checkpoint_val_loss:.8f}"
    )

    print(
        f"Input dim       : "
        f"{checkpoint_input_dim}"
    )

    print(
        f"Output dim      : "
        f"{checkpoint_output_dim}"
    )

    print(
        f"k               : "
        f"{checkpoint_k}"
    )

    print(
        f"First kNN graph : "
        f"{checkpoint_first_knn_space}"
    )

    print(
        f"Train kNN chunk : "
        f"{checkpoint_chunk_size}"
    )

    print(
        f"Eval kNN chunk  : "
        f"{evaluation_chunk_size}"
    )

    if checkpoint_point_count is not None:

        print(
            f"Train points    : "
            f"{checkpoint_point_count}"
        )

    # =================================================================
    # Load scaler
    # =================================================================

    scaler = CFDScaler.load(
        scaler_path
    )

    print()
    print("=" * 78)
    print("DGCNN SCALER")
    print("=" * 78)

    scaler.print_statistics()

    # =================================================================
    # Load test dataset
    # =================================================================

    dataset_module = load_dataset_module(
        data_root
    )

    num_points = int(
        getattr(
            dataset_module,
            "NUM_POINTS",
            7000,
        )
    )

    test_base = create_base_dataset(
        dataset_module,
        "test",
    )

    print()
    print("=" * 78)
    print("TEST DATASET")
    print("=" * 78)

    print(
        f"Test samples   : "
        f"{len(test_base)}"
    )

    print(
        f"FPS points     : "
        f"{num_points}"
    )

    if (
        len(test_base)
        != EXPECTED_TEST_SAMPLES
    ):
        raise RuntimeError(
            "Unexpected number of test samples. "
            f"Expected {EXPECTED_TEST_SAMPLES}, "
            f"found {len(test_base)}."
        )

    # -----------------------------------------------------------------
    # Verify one real test sample.
    # -----------------------------------------------------------------

    x_check, y_check = extract_xy(
        test_base[0]
    )

    print(
        f"Sample X shape : "
        f"{tuple(x_check.shape)}"
    )

    print(
        f"Sample Y shape : "
        f"{tuple(y_check.shape)}"
    )

    if (
        x_check.shape[0]
        != num_points
    ):
        raise RuntimeError(
            "Unexpected test point count:\n"
            f"Expected {num_points}, "
            f"found {x_check.shape[0]}."
        )

    # =================================================================
    # Determine actual evaluation point count
    # =================================================================

    actual_point_count = (
        args.point_limit
        if args.point_limit is not None
        else num_points
    )

    if checkpoint_k >= actual_point_count:
        raise ValueError(
            f"k={checkpoint_k} must be smaller than "
            f"evaluation point count={actual_point_count}."
        )

    # -----------------------------------------------------------------
    # A checkpoint trained with a debug point limit should be evaluated
    # with the same point count.
    #
    # This prevents accidentally treating a 512-point smoke-test model
    # as a real 7000-point model.
    # -----------------------------------------------------------------

    if checkpoint_point_count is not None:

        checkpoint_point_count = int(
            checkpoint_point_count
        )

        if (
            checkpoint_point_count
            != actual_point_count
        ):

            raise RuntimeError(
                "\nCheckpoint point-count mismatch.\n\n"
                f"Checkpoint was trained with : "
                f"{checkpoint_point_count} points/sample\n"
                f"Evaluation requested         : "
                f"{actual_point_count} points/sample\n\n"
                "For the current 512-point debug checkpoint, "
                "run evaluate.py with:\n"
                "    --point-limit 512\n\n"
                "For the final 7000-point checkpoint, "
                "omit --point-limit."
            )

    # =================================================================
    # Normalized test dataset
    # =================================================================

    test_dataset = (
        NormalizedDGCNNDataset(
            test_base,
            scaler,
            point_limit=(
                args.point_limit
            ),
        )
    )

    test_loader = build_loader(
        test_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        device=device,
        seed=42,
    )

    # =================================================================
    # Reconstruct model
    # =================================================================

    model = DGCNNRegressor(
        input_dim=(
            checkpoint_input_dim
        ),
        k=(
            checkpoint_k
        ),
        knn_chunk_size=(
            evaluation_chunk_size
        ),
    ).to(
        device
    )

    if int(model.output_dim) != 3:
        raise RuntimeError(
            "Reconstructed DGCNN model has an unexpected "
            f"output_dim={model.output_dim}. "
            "Expected 3 targets: "
            "[HTC, wall_shear, pressure]."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print()
    print("=" * 78)
    print("MODEL")
    print("=" * 78)

    print(model)

    print()

    print(
        f"Trainable parameters : "
        f"{count_parameters(model):,}"
    )

    print(
        f"Evaluation points    : "
        f"{actual_point_count:,}"
    )

    # =================================================================
    # Inference
    # =================================================================

    print()
    print("=" * 78)
    print("TEST INFERENCE")
    print("=" * 78)

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()

    inference_start = (
        time.perf_counter()
    )

    y_true_norm, y_pred_norm = (
        predict_loader(
            model,
            test_loader,
            device,
            max_batches=(
                args.max_test_batches
            ),
        )
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # -----------------------------------------------------------------
    # Validate prediction
    # -----------------------------------------------------------------

    if (
        y_true_norm.shape
        != y_pred_norm.shape
    ):
        raise RuntimeError(
            "Prediction / target shape mismatch:\n"
            f"Target     : {y_true_norm.shape}\n"
            f"Prediction : {y_pred_norm.shape}"
        )

    if (
        y_true_norm.ndim != 2
        or y_true_norm.shape[1] != 3
    ):
        raise RuntimeError(
            "Unexpected evaluation output shape: "
            f"{y_true_norm.shape}. "
            "Expected (N, 3) for "
            "[HTC, wall_shear, pressure]."
        )

    if not np.isfinite(
        y_pred_norm
    ).all():
        raise RuntimeError(
            "Prediction contains NaN or Inf."
        )

    evaluated_points = (
        y_true_norm.shape[0]
    )

    normalized_mse = float(
        np.mean(
            (
                y_pred_norm
                - y_true_norm
            )
            ** 2
        )
    )

    print(
        f"Evaluated points : "
        f"{evaluated_points:,}"
    )

    print(
        f"Inference runtime: "
        f"{inference_time:.2f} s"
    )

    print(
        f"Normalized MSE   : "
        f"{normalized_mse:.8f}"
    )

    if device.type == "cuda":

        peak_memory_gb = (
            torch.cuda
            .max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"GPU peak memory  : "
            f"{peak_memory_gb:.2f} GB"
        )

    # =================================================================
    # Return predictions to physical units
    # =================================================================

    y_true_physical = (
        scaler.inverse_target(
            y_true_norm
        )
    )

    y_pred_physical = (
        scaler.inverse_target(
            y_pred_norm
        )
    )

    # =================================================================
    # Physical-unit metrics
    # =================================================================

    metrics = calculate_metrics(
        y_true_physical,
        y_pred_physical,
    )

    print()
    print("=" * 78)
    print("HELD-OUT TEST METRICS")
    print("=" * 78)

    print_metrics(
        metrics
    )

    # =================================================================
    # Final summary
    # =================================================================

    print()
    print("=" * 78)
    print("DGCNN TEST EVALUATION COMPLETE")
    print("=" * 78)

    print(
        f"Checkpoint epoch : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Checkpoint val   : "
        f"{checkpoint_val_loss:.8f}"
    )

    print(
        f"Test samples     : "
        f"{len(test_base)}"
    )

    if args.max_test_batches is not None:

        print(
            f"Test batch limit : "
            f"{args.max_test_batches} "
            "(DEBUG ONLY)"
        )

    else:

        print(
            "Test batch limit : "
            "None (full test set)"
        )

    print(
        f"Points/sample    : "
        f"{actual_point_count:,}"
    )

    print(
        f"Evaluated points : "
        f"{evaluated_points:,}"
    )

    print(
        f"Inference runtime: "
        f"{inference_time:.2f} s"
    )

    print("=" * 78)


# =====================================================================
# CLI
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained DGCNN on the held-out "
            "CFD test split."
        )
    )

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional ai-cfd-data root. "
            "Default: project_paths.DATA_ROOT"
        ),
    )

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    # -----------------------------------------------------------------
    # Checkpoint / scaler
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # k-NN
    #
    # This can be changed at evaluation time because chunk size only
    # changes memory/computation scheduling, not the learned model.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--knn-chunk-size",
        type=int,
        default=None,
        help=(
            "Optional evaluation-time kNN chunk size. "
            "Default: use value stored in checkpoint."
        ),
    )

    # -----------------------------------------------------------------
    # Debug options
    # -----------------------------------------------------------------

    parser.add_argument(
        "--point-limit",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY. Must match the checkpoint point count. "
            "Final 7000-point evaluation should omit this option."
        ),
    )

    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY: evaluate only this many test batches."
        ),
    )

    return parser.parse_args()


# =====================================================================
# Argument validation
# =====================================================================

def validate_args(
    args,
) -> None:

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if (
        args.knn_chunk_size is not None
        and args.knn_chunk_size <= 0
    ):
        raise ValueError(
            "--knn-chunk-size must be positive."
        )

    if (
        args.point_limit is not None
        and args.point_limit <= 0
    ):
        raise ValueError(
            "--point-limit must be positive."
        )

    if (
        args.max_test_batches is not None
        and args.max_test_batches <= 0
    ):
        raise ValueError(
            "--max-test-batches must be positive."
        )


# =====================================================================
# Entry point
# =====================================================================

def main():

    args = parse_args()

    validate_args(
        args
    )

    evaluate(
        args
    )


if __name__ == "__main__":
    main()