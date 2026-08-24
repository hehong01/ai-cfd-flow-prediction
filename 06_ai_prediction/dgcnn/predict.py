"""
Run the trained DGCNN on preprocessed prediction point clouds.

Input:
    ai-cfd-data/07_predictions/dgcnn/input_csv/<name>.csv

Expected columns:
    x,y,z

The input must contain the same point count used by the trained checkpoint
(final model: 7000 FPS-selected points).

A single inlet velocity is appended to every point:
    [x, y, z] + velocity -> [x, y, z, velocity]

Inference follows the validated training/evaluation pipeline:
    raw physical XYZ -> first k-NN graph
    normalized [x,y,z,velocity] -> DGCNN edge values/features
    best_model.pt -> normalized [HTC, wall_shear]
    saved TRAIN scaler -> physical [HTC, wall_shear]

Output:
    ai-cfd-data/07_predictions/dgcnn/prediction_csv/<name>_vel<speed>.csv

Columns:
    x,y,z,velocity,predicted_htc,predicted_wall_shear
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
PREDICTION_CODE_ROOT = THIS_DIR.parent
GITHUB_ROOT = PREDICTION_CODE_ROOT.parent

TRAINING_ROOT = GITHUB_ROOT / "05_model_training"
TRAINING_DGCNN_DIR = TRAINING_ROOT / "dgcnn"
TRAINING_COMMON_DIR = TRAINING_ROOT / "common"

for path in (
    TRAINING_DGCNN_DIR,
    TRAINING_COMMON_DIR,
    TRAINING_ROOT,
    GITHUB_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from project_paths import DATA_ROOT
from model import DGCNNRegressor, count_parameters
from scalers import CFDScaler


PREDICTION_ROOT = DATA_ROOT / "07_predictions"
INPUT_CSV_DIR = PREDICTION_ROOT / "dgcnn" / "input_csv"
OUTPUT_CSV_DIR = PREDICTION_ROOT / "dgcnn" / "prediction_csv"

WEIGHT_DIR = TRAINING_ROOT / "weights" / "dgcnn"
DEFAULT_MODEL_PATH = WEIGHT_DIR / "best_model.pt"
DEFAULT_SCALER_PATH = WEIGHT_DIR / "scalers.npz"

INPUT_HEADER = "x,y,z"
OUTPUT_HEADER = (
    "x,y,z,velocity,"
    "predicted_htc,predicted_wall_shear"
)


def load_xyz_csv(csv_path: Path) -> np.ndarray:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found:\n{csv_path}")

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:
        header = file.readline().strip()

    if header != INPUT_HEADER:
        raise ValueError(
            f"{csv_path.name}: unexpected header '{header}'. "
            f"Expected '{INPUT_HEADER}'."
        )

    xyz = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )

    if xyz.ndim == 1:
        xyz = xyz.reshape(1, -1)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"{csv_path.name}: invalid XYZ shape {xyz.shape}."
        )

    if len(xyz) == 0:
        raise ValueError(
            f"{csv_path.name}: CSV contains no points."
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            f"{csv_path.name}: XYZ contains NaN or Inf."
        )

    return xyz


def collect_input_csvs(input_value: str | None) -> list[Path]:
    if input_value:
        csv_path = Path(input_value)

        if not csv_path.is_absolute():
            csv_path = INPUT_CSV_DIR / csv_path

        csv_path = csv_path.resolve()

        if csv_path.suffix.lower() != ".csv":
            raise ValueError(
                f"Input must be a CSV file:\n{csv_path}"
            )

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Input CSV not found:\n{csv_path}"
            )

        return [csv_path]

    if not INPUT_CSV_DIR.exists():
        return []

    return sorted(
        path
        for path in INPUT_CSV_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
    )


def velocity_tag(velocity: float) -> str:
    text = f"{velocity:g}"
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text


def build_output_path(
    input_csv: Path,
    velocity: float,
) -> Path:
    return (
        OUTPUT_CSV_DIR
        / f"{input_csv.stem}_vel{velocity_tag(velocity)}.csv"
    )


def output_needs_rebuild(
    output_path: Path,
    input_csv: Path,
    model_path: Path,
    scaler_path: Path,
    overwrite: bool,
) -> tuple[bool, str]:
    if overwrite:
        return True, "--overwrite"

    if not output_path.exists():
        return True, "missing"

    output_time = output_path.stat().st_mtime_ns

    for label, path in (
        ("input CSV", input_csv),
        ("model checkpoint", model_path),
        ("scaler", scaler_path),
    ):
        if path.stat().st_mtime_ns > output_time:
            return True, f"{label} is newer"

    return False, "up to date"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint_and_model(
    model_path: Path,
    device: torch.device,
    knn_chunk_size_override: int | None,
) -> tuple[DGCNNRegressor, dict, int]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"DGCNN checkpoint not found:\n{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    required = {
        "model_state_dict",
        "input_dim",
        "k",
        "epoch",
        "val_loss",
        "first_knn_space",
    }

    missing = required - set(checkpoint.keys())

    if missing:
        raise ValueError(
            "DGCNN checkpoint is missing fields: "
            f"{sorted(missing)}"
        )

    first_knn_space = str(
        checkpoint["first_knn_space"]
    )

    if first_knn_space != "raw_xyz":
        raise RuntimeError(
            "Unsupported first k-NN graph space: "
            f"{first_knn_space!r}. Expected 'raw_xyz'."
        )

    input_dim = int(
        checkpoint["input_dim"]
    )

    k = int(
        checkpoint["k"]
    )

    checkpoint_chunk_size = int(
        checkpoint.get(
            "knn_chunk_size",
            1024,
        )
    )

    if knn_chunk_size_override is None:
        inference_chunk_size = checkpoint_chunk_size
    else:
        inference_chunk_size = int(
            knn_chunk_size_override
        )

    if inference_chunk_size <= 0:
        raise ValueError(
            "--knn-chunk-size must be positive."
        )

    model = DGCNNRegressor(
        input_dim=input_dim,
        k=k,
        knn_chunk_size=inference_chunk_size,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint, inference_chunk_size


def build_model_input(
    xyz: np.ndarray,
    velocity: float,
) -> np.ndarray:
    velocity_column = np.full(
        (len(xyz), 1),
        velocity,
        dtype=np.float64,
    )

    x = np.concatenate(
        (xyz, velocity_column),
        axis=1,
    )

    if x.shape != (len(xyz), 4):
        raise RuntimeError(
            f"Unexpected model-input shape: {x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Model input contains NaN or Inf."
        )

    return x


def run_inference(
    model: DGCNNRegressor,
    x_normalized: np.ndarray,
    raw_xyz: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if x_normalized.ndim != 2 or x_normalized.shape[1] != 4:
        raise ValueError(
            f"Expected normalized X shape (N,4), found {x_normalized.shape}."
        )

    if raw_xyz.shape != (
        len(x_normalized),
        3,
    ):
        raise ValueError(
            f"Expected raw XYZ shape ({len(x_normalized)},3), "
            f"found {raw_xyz.shape}."
        )

    x_tensor = torch.from_numpy(
        x_normalized
    ).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    raw_xyz_tensor = torch.from_numpy(
        raw_xyz
    ).to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        y_tensor = model(
            x_tensor,
            raw_xyz_tensor,
        )

    expected_shape = (
        1,
        len(x_normalized),
        2,
    )

    if tuple(y_tensor.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected DGCNN output shape: "
            f"{tuple(y_tensor.shape)}, expected {expected_shape}."
        )

    y_normalized = (
        y_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    if not np.all(np.isfinite(y_normalized)):
        raise RuntimeError(
            "Normalized DGCNN prediction contains NaN or Inf."
        )

    return y_normalized


def save_prediction_csv(
    output_path: Path,
    xyz: np.ndarray,
    velocity: float,
    predictions: np.ndarray,
) -> None:
    if predictions.shape != (len(xyz), 2):
        raise ValueError(
            f"Invalid prediction shape: {predictions.shape}"
        )

    velocity_column = np.full(
        (len(xyz), 1),
        velocity,
        dtype=np.float64,
    )

    output = np.concatenate(
        (
            xyz,
            velocity_column,
            predictions,
        ),
        axis=1,
    )

    if output.shape != (len(xyz), 6):
        raise RuntimeError(
            f"Unexpected output shape: {output.shape}"
        )

    if not np.all(np.isfinite(output)):
        raise ValueError(
            "Prediction output contains NaN or Inf."
        )

    OUTPUT_CSV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        f"{output_path.stem}.tmp.csv"
    )

    if temp_path.exists():
        temp_path.unlink()

    try:
        np.savetxt(
            temp_path,
            output,
            delimiter=",",
            header=OUTPUT_HEADER,
            comments="",
            fmt="%.10e",
        )

        temp_path.replace(
            output_path
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def print_prediction_summary(
    predictions: np.ndarray,
) -> None:
    htc = predictions[:, 0]
    wall_shear = predictions[:, 1]

    print(
        "  HTC [W/(m^2 K)] : "
        f"min={htc.min():.6f}, "
        f"mean={htc.mean():.6f}, "
        f"max={htc.max():.6f}"
    )

    print(
        "  wall shear [Pa] : "
        f"min={wall_shear.min():.6f}, "
        f"mean={wall_shear.mean():.6f}, "
        f"max={wall_shear.max():.6f}"
    )


def predict_one_csv(
    input_csv: Path,
    velocity: float,
    model: DGCNNRegressor,
    scaler: CFDScaler,
    device: torch.device,
    checkpoint: dict,
    model_path: Path,
    scaler_path: Path,
    overwrite: bool,
) -> None:
    output_path = build_output_path(
        input_csv,
        velocity,
    )

    rebuild, reason = output_needs_rebuild(
        output_path=output_path,
        input_csv=input_csv,
        model_path=model_path,
        scaler_path=scaler_path,
        overwrite=overwrite,
    )

    print()
    print("=" * 78)
    print(f"INPUT: {input_csv.name}")
    print("=" * 78)

    if not rebuild:
        print(
            f"[PREDICTION] SKIP ({reason})"
        )
        print(
            f"Output: {output_path}"
        )
        return

    print(
        f"[PREDICTION] GENERATE ({reason})"
    )

    xyz = load_xyz_csv(
        input_csv
    )

    checkpoint_point_count = checkpoint.get(
        "point_count",
        None,
    )

    if checkpoint_point_count is not None:
        expected_points = int(
            checkpoint_point_count
        )

        if len(xyz) != expected_points:
            raise ValueError(
                "Point-count mismatch:\n"
                f"  checkpoint : {expected_points:,}\n"
                f"  input CSV  : {len(xyz):,}"
            )

    k = int(
        checkpoint["k"]
    )

    if k >= len(xyz):
        raise ValueError(
            f"k={k} must be smaller than point count={len(xyz)}."
        )

    print(
        f"Points       : {len(xyz):,}"
    )
    print(
        f"Velocity     : {velocity:g} m/s"
    )

    # Preserve raw physical XYZ BEFORE standardization.
    # EdgeConv1 uses these coordinates for exact k-NN.
    raw_xyz = xyz.astype(
        np.float32,
        copy=True,
    )

    x = build_model_input(
        xyz,
        velocity,
    )

    # Saved DGCNN TRAIN scaler.
    x_normalized = (
        scaler
        .transform_input(x)
        .astype(
            np.float32,
            copy=False,
        )
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    y_normalized = run_inference(
        model=model,
        x_normalized=x_normalized,
        raw_xyz=raw_xyz,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter()
        - start
    )

    predictions = scaler.inverse_target(
        y_normalized
    )

    if predictions.shape != (len(xyz), 2):
        raise RuntimeError(
            f"Unexpected physical prediction shape: {predictions.shape}"
        )

    print(
        f"Inference    : {inference_time:.4f} s"
    )

    print_prediction_summary(
        predictions
    )

    save_prediction_csv(
        output_path=output_path,
        xyz=xyz,
        velocity=velocity,
        predictions=predictions,
    )

    print(
        f"[PREDICTION] SAVED: {output_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict HTC and wall shear from a 7000-point FPS cloud "
            "using the trained DGCNN."
        )
    )

    parser.add_argument(
        "--velocity",
        type=float,
        required=True,
        help=(
            "Inlet velocity in m/s. "
            "The training velocities were 5, 8, and 10 m/s."
        ),
    )

    parser.add_argument(
        "--input",
        help=(
            "Optional single DGCNN input CSV. "
            "A bare filename is resolved inside "
            "ai-cfd-data/07_predictions/dgcnn/input_csv. "
            "If omitted, all CSV files in that folder are scanned."
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
        "--knn-chunk-size",
        type=int,
        default=None,
        help=(
            "Optional exact-kNN query chunk-size override. "
            "If omitted, use the checkpoint value."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate prediction CSV even if the existing output "
            "is already up to date."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not np.isfinite(args.velocity):
        raise ValueError(
            "--velocity must be finite."
        )

    if args.velocity <= 0.0:
        raise ValueError(
            "--velocity must be positive."
        )

    if (
        args.knn_chunk_size is not None
        and args.knn_chunk_size <= 0
    ):
        raise ValueError(
            "--knn-chunk-size must be positive."
        )

    model_path = Path(
        args.model_path
    ).resolve()

    scaler_path = Path(
        args.scaler_path
    ).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"DGCNN checkpoint not found:\n{model_path}"
        )

    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"DGCNN scaler not found:\n{scaler_path}"
        )

    input_csvs = collect_input_csvs(
        args.input
    )

    OUTPUT_CSV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("=" * 78)
    print("AI-CFD DGCNN PREDICTION")
    print("=" * 78)
    print(
        f"Input folder    : {INPUT_CSV_DIR}"
    )
    print(
        f"Output folder   : {OUTPUT_CSV_DIR}"
    )
    print(
        f"Model           : {model_path}"
    )
    print(
        f"Scaler          : {scaler_path}"
    )
    print(
        f"Velocity        : {args.velocity:g} m/s"
    )
    print(
        f"Device          : {device}"
    )
    print(
        f"CSV discovered  : {len(input_csvs)}"
    )

    if args.velocity not in (5.0, 8.0, 10.0):
        if 5.0 <= args.velocity <= 10.0:
            print(
                "Velocity note   : interpolation between "
                "training velocities 5/8/10 m/s"
            )
        else:
            print(
                "Velocity warning: outside the training velocity "
                "range 5-10 m/s; prediction is extrapolative."
            )

    print("=" * 78)

    if not input_csvs:
        print(
            "No DGCNN input CSV files found."
        )
        return 0

    print()
    print("[LOAD SCALER]")
    scaler = CFDScaler.load(
        scaler_path
    )

    print("[LOAD MODEL]")
    (
        model,
        checkpoint,
        inference_chunk_size,
    ) = load_checkpoint_and_model(
        model_path=model_path,
        device=device,
        knn_chunk_size_override=args.knn_chunk_size,
    )

    checkpoint_point_count = checkpoint.get(
        "point_count",
        "unknown",
    )

    print(
        f"Model name      : "
        f"{checkpoint.get('model_name', 'unknown')}"
    )
    print(
        f"Parameters      : {count_parameters(model):,}"
    )
    print(
        f"Best epoch      : {int(checkpoint['epoch'])}"
    )
    print(
        f"Validation loss : {float(checkpoint['val_loss']):.8f}"
    )
    print(
        f"k               : {int(checkpoint['k'])}"
    )
    print(
        f"First kNN graph : {checkpoint['first_knn_space']}"
    )
    print(
        f"Train points    : {checkpoint_point_count}"
    )
    print(
        f"kNN chunk       : {inference_chunk_size}"
    )

    failures: list[tuple[str, str]] = []

    for input_csv in input_csvs:
        try:
            predict_one_csv(
                input_csv=input_csv,
                velocity=args.velocity,
                model=model,
                scaler=scaler,
                device=device,
                checkpoint=checkpoint,
                model_path=model_path,
                scaler_path=scaler_path,
                overwrite=args.overwrite,
            )

        except Exception as exc:
            message = (
                f"{type(exc).__name__}: {exc}"
            )

            failures.append(
                (
                    input_csv.name,
                    message,
                )
            )

            print(
                f"[FAILED] {input_csv.name}: {message}"
            )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"Input CSVs : {len(input_csvs)}"
    )
    print(
        f"Failed     : {len(failures)}"
    )

    if failures:
        for name, message in failures:
            print(
                f"  - {name}: {message}"
            )

        return 1

    print(
        "DGCNN prediction completed successfully."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
