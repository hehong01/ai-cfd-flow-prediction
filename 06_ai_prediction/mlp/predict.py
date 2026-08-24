"""
Run the trained point-wise MLP on preprocessed prediction geometry.

Input
-----
ai-cfd-data/07_predictions/mlp/input_csv/<name>.csv

Expected columns:
    x,y,z

A single inlet velocity is supplied from the command line and appended to
every surface point:

    [x, y, z] + velocity -> [x, y, z, velocity]

The saved TRAIN scaler and trained best_model.pt from 05_model_training
are then used for inference.

Output
------
ai-cfd-data/07_predictions/mlp/prediction_csv/<name>_vel<speed>.csv

Columns:
    x,y,z,velocity,predicted_htc,predicted_wall_shear

Examples
--------
From github/06_ai_prediction/mlp:

    python predict.py --velocity 8
    python predict.py --velocity 8 --input test_face.csv
    python predict.py --velocity 5 --input test_face.csv --overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# Project paths
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
PREDICTION_CODE_ROOT = THIS_DIR.parent
GITHUB_ROOT = PREDICTION_CODE_ROOT.parent

TRAINING_ROOT = GITHUB_ROOT / "05_model_training"
TRAINING_MLP_DIR = TRAINING_ROOT / "mlp"
TRAINING_COMMON_DIR = TRAINING_ROOT / "common"

for path in (
    GITHUB_ROOT,
    TRAINING_ROOT,
    TRAINING_MLP_DIR,
    TRAINING_COMMON_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from project_paths import DATA_ROOT
from model import PointwiseMLP, count_parameters
from scalers import CFDScaler


PREDICTION_ROOT = DATA_ROOT / "07_predictions"

INPUT_CSV_DIR = (
    PREDICTION_ROOT
    / "mlp"
    / "input_csv"
)

OUTPUT_CSV_DIR = (
    PREDICTION_ROOT
    / "mlp"
    / "prediction_csv"
)

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

DEFAULT_BATCH_SIZE = 8192

INPUT_HEADER = "x,y,z"

OUTPUT_HEADER = (
    "x,y,z,velocity,"
    "predicted_htc,predicted_wall_shear"
)


# =============================================================================
# Input discovery / validation
# =============================================================================

def load_xyz_csv(
    csv_path: Path,
) -> np.ndarray:
    """Load one preprocessed MLP XYZ CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found:\n{csv_path}"
        )

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


def collect_input_csvs(
    input_value: str | None,
) -> list[Path]:
    """
    Return one requested MLP input CSV or every CSV in mlp/input_csv.
    """
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


# =============================================================================
# Output naming / incremental rebuild
# =============================================================================

def velocity_tag(
    velocity: float,
) -> str:
    """
    Convert a velocity to a filename-safe compact tag.

    Examples:
        8.0  -> 8
        8.5  -> 8p5
        10.0 -> 10
    """
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
    """
    Regenerate if output is missing or any dependency is newer.
    """
    if overwrite:
        return True, "--overwrite"

    if not output_path.exists():
        return True, "missing"

    output_time = output_path.stat().st_mtime_ns

    dependencies = (
        ("input CSV", input_csv),
        ("model checkpoint", model_path),
        ("scaler", scaler_path),
    )

    for label, path in dependencies:
        if path.stat().st_mtime_ns > output_time:
            return True, f"{label} is newer"

    return False, "up to date"


# =============================================================================
# Device / trained model
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[PointwiseMLP, dict]:
    """
    Load the final saved PointwiseMLP checkpoint using the same checkpoint
    structure as 05_model_training/mlp/evaluate.py.
    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
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

    missing = required - set(checkpoint.keys())

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
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# =============================================================================
# MLP inference
# =============================================================================

def build_model_input(
    xyz: np.ndarray,
    velocity: float,
) -> np.ndarray:
    """
    Construct [x, y, z, velocity] for every input point.
    """
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
    model: PointwiseMLP,
    x_normalized: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Run point-wise inference in batches and return normalized predictions.
    """
    predictions: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(x_normalized),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(x_normalized),
            )

            x_batch = torch.from_numpy(
                x_normalized[start:end]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            y_batch = model(
                x_batch
            )

            if y_batch.ndim != 2 or y_batch.shape[1] != 2:
                raise RuntimeError(
                    "Unexpected model output shape: "
                    f"{tuple(y_batch.shape)}"
                )

            predictions.append(
                y_batch
                .detach()
                .cpu()
                .numpy()
            )

    y_normalized = np.concatenate(
        predictions,
        axis=0,
    )

    if y_normalized.shape != (
        len(x_normalized),
        2,
    ):
        raise RuntimeError(
            "Unexpected concatenated prediction shape: "
            f"{y_normalized.shape}"
        )

    if not np.all(np.isfinite(y_normalized)):
        raise RuntimeError(
            "Normalized prediction contains NaN or Inf."
        )

    return y_normalized


# =============================================================================
# Output
# =============================================================================

def save_prediction_csv(
    output_path: Path,
    xyz: np.ndarray,
    velocity: float,
    predictions: np.ndarray,
) -> None:
    """
    Save:
        x,y,z,velocity,predicted_htc,predicted_wall_shear
    """
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


# =============================================================================
# One CSV
# =============================================================================

def predict_one_csv(
    input_csv: Path,
    velocity: float,
    model: PointwiseMLP,
    scaler: CFDScaler,
    device: torch.device,
    batch_size: int,
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

    print(
        f"Points       : {len(xyz):,}"
    )
    print(
        f"Velocity     : {velocity:g} m/s"
    )

    x = build_model_input(
        xyz,
        velocity,
    )

    # Saved TRAIN-data normalization.
    x_normalized = (
        scaler
        .transform_input(x)
        .astype(
            np.float32,
            copy=False,
        )
    )

    start = time.perf_counter()

    y_normalized = run_inference(
        model=model,
        x_normalized=x_normalized,
        device=device,
        batch_size=batch_size,
    )

    inference_time = (
        time.perf_counter()
        - start
    )

    # Restore physical units:
    # [HTC, wall_shear]
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


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict HTC and wall shear from preprocessed XYZ points "
            "using the trained point-wise MLP."
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
            "Optional single MLP input CSV. "
            "A bare filename is resolved inside "
            "ai-cfd-data/07_predictions/mlp/input_csv. "
            "If omitted, all CSV files in that folder are scanned."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Inference batch size (default: {DEFAULT_BATCH_SIZE})."
        ),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Optional trained MLP checkpoint path."
        ),
    )

    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=DEFAULT_SCALER_PATH,
        help=(
            "Optional MLP scaler path."
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

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    model_path = Path(
        args.model_path
    ).resolve()

    scaler_path = Path(
        args.scaler_path
    ).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Scaler not found:\n{scaler_path}"
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
    print("AI-CFD MLP PREDICTION")
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
        f"Batch size      : {args.batch_size:,}"
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
            "No MLP input CSV files found."
        )
        return 0

    # Load once and reuse for every input geometry.
    print()
    print("[LOAD SCALER]")
    scaler = CFDScaler.load(
        scaler_path
    )

    print("[LOAD MODEL]")
    model, checkpoint = load_model(
        model_path,
        device,
    )

    print(
        f"Architecture     : "
        f"{checkpoint['input_dim']} -> "
        f"{' -> '.join(str(v) for v in checkpoint['hidden_dims'])} -> "
        f"{checkpoint['output_dim']}"
    )
    print(
        f"Parameters       : {count_parameters(model):,}"
    )
    print(
        f"Best epoch       : {checkpoint['epoch']}"
    )
    print(
        f"Validation loss  : {checkpoint['val_loss']:.8f}"
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
                batch_size=args.batch_size,
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
        "MLP prediction completed successfully."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
