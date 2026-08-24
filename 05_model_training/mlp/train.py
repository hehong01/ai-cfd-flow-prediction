"""
Train the point-wise MLP baseline for CFD surface-field prediction.

Input per CFD wall node:
    [x, y, z, velocity]

Target per CFD wall node:
    [HTC, wall_shear, pressure]

Dataset split:
    train : face_0001 ~ face_0080
    val   : face_0081 ~ face_0090
    test  : face_0091 ~ face_0100

Important:
    - MLP uses ALL original CFD wall nodes.
    - DGCNN FPS preprocessing is NOT used here.
    - Scaler statistics are fitted using TRAIN data only.
    - Validation data is never used to fit normalization statistics.
    - The latest completed training state is saved after every epoch
      so interrupted Colab training can resume.

Outputs:
    weights/mlp/best_model.pt
    weights/mlp/scalers.npz
    weights/mlp/last_checkpoint.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =====================================================================
# Project paths
# =====================================================================

THIS_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = THIS_DIR.parent
GITHUB_ROOT = TRAINING_ROOT.parent
COMMON_DIR = TRAINING_ROOT / "common"

if str(GITHUB_ROOT) not in sys.path:
    sys.path.insert(0, str(GITHUB_ROOT))

if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from project_paths import DATA_ROOT
from model import PointwiseMLP, count_parameters
from scalers import CFDScaler
from metrics import calculate_metrics, print_metrics


# =====================================================================
# Constants
# =====================================================================

SPEEDS = (
    ("05mps", 5.0),
    ("08mps", 8.0),
    ("10mps", 10.0),
)

DEFAULT_BATCH_SIZE = 8192
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_PATIENCE = 15
DEFAULT_SEED = 42
DEFAULT_LOG_EVERY = 50

WEIGHT_DIR = TRAINING_ROOT / "weights" / "mlp"

DEFAULT_MODEL_PATH = WEIGHT_DIR / "best_model.pt"
DEFAULT_SCALER_PATH = WEIGHT_DIR / "scalers.npz"
DEFAULT_CHECKPOINT_PATH = WEIGHT_DIR / "last_checkpoint.pt"


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed: int) -> None:
    """Set Python / NumPy / PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Device
# =====================================================================

def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =====================================================================
# Dataset paths
# =====================================================================

def get_face_range(split: str) -> range:
    """Return the face-number range for one split."""

    if split == "train":
        return range(1, 81)

    if split == "val":
        return range(81, 91)

    if split == "test":
        return range(91, 101)

    raise ValueError(f"Invalid split: {split}")


def build_csv_list(csv_dir: Path, split: str) -> list[tuple[Path, float]]:
    """Build the CSV list for one split."""

    samples = []

    for face_num in get_face_range(split):
        face_id = f"face_{face_num:04d}"

        for speed_name, velocity in SPEEDS:
            csv_path = csv_dir / f"{face_id}_{speed_name}.csv"

            if not csv_path.exists():
                raise FileNotFoundError(f"Missing CFD CSV:\n{csv_path}")

            samples.append((csv_path, velocity))

    return samples


# =====================================================================
# Raw CSV loading
# =====================================================================

def load_one_csv(csv_path: Path, velocity: float) -> tuple[np.ndarray, np.ndarray]:
    """Load one CFD CSV as X=[x,y,z,velocity], Y=[HTC,wall_shear]."""

    data = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3, 4, 7, 9),
        dtype=np.float32,
    )

    if data.ndim == 1:
        data = data.reshape(1, 6)

    if data.shape[1] != 6:
        raise ValueError(
            f"{csv_path.name}: unexpected loaded shape {data.shape}"
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(f"{csv_path.name}: NaN or Inf found.")

    num_points = len(data)

    velocity_column = np.full(
        (num_points, 1),
        velocity,
        dtype=np.float32,
    )

    x = np.concatenate(
        (data[:, 0:3], velocity_column),
        axis=1,
    )

    y = np.stack(
        (
            data[:, 5],  # HTC
            data[:, 4],  # wall shear
            data[:, 3],  # pressure
        ),
        axis=1,
    )

    return x, y


def load_split(csv_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load every CFD wall node belonging to one split."""

    csv_samples = build_csv_list(csv_dir, split)

    print()
    print(f"[LOAD {split.upper()}]")
    print(f"CSV files : {len(csv_samples)}")

    x_parts = []
    y_parts = []
    total_rows = 0
    start = time.perf_counter()

    for index, (csv_path, velocity) in enumerate(csv_samples, start=1):
        x, y = load_one_csv(csv_path, velocity)

        x_parts.append(x)
        y_parts.append(y)
        total_rows += len(x)

        if index == 1 or index % 25 == 0 or index == len(csv_samples):
            print(
                f"  {index:3d}/{len(csv_samples)}  "
                f"{csv_path.name:<24} "
                f"rows={len(x):5d}  "
                f"total={total_rows:,}",
                flush=True,
            )

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    elapsed = time.perf_counter() - start

    if x_all.shape[1] != 4:
        raise RuntimeError(f"{split}: invalid X shape {x_all.shape}")

    if y_all.shape[1] != 3:
        raise RuntimeError(f"{split}: invalid Y shape {y_all.shape}")

    print(f"{split.upper()} loaded:")
    print(f"  X shape : {x_all.shape}")
    print(f"  Y shape : {y_all.shape}")
    print(f"  runtime : {elapsed:.1f} s")

    return x_all, y_all


# =====================================================================
# DataLoader
# =====================================================================

def build_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Convert normalized arrays to a PyTorch DataLoader."""

    x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))

    dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


# =====================================================================
# One training epoch
# =====================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int,
    max_batches: int | None = None,
) -> float:
    """Run one training epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0
    total_batches = len(loader)
    effective_batches = (
        total_batches
        if max_batches is None
        else min(total_batches, max_batches)
    )

    start = time.perf_counter()

    for batch_index, (x_batch, y_batch) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)

        loss.backward()
        optimizer.step()

        current_batch_size = len(x_batch)
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

        if (
            batch_index == 1
            or batch_index % log_every == 0
            or batch_index == effective_batches
        ):
            elapsed = time.perf_counter() - start
            print(
                f"    batch {batch_index:4d}/{effective_batches:<4d} | "
                f"loss={loss.item():.6f} | {elapsed:.1f}s",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError("No training batches were processed.")

    return total_loss / total_samples


# =====================================================================
# Validation
# =====================================================================

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Calculate mean validation loss."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch_index, (x_batch, y_batch) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)

        current_batch_size = len(x_batch)
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

    if total_samples == 0:
        raise RuntimeError("No validation batches were processed.")

    return total_loss / total_samples


# =====================================================================
# Full prediction
# =====================================================================

@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a DataLoader and return normalized y_true / y_pred."""

    model.eval()

    true_parts = []
    pred_parts = []

    for batch_index, (x_batch, y_batch) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        x_batch = x_batch.to(device, non_blocking=True)
        prediction = model(x_batch)

        true_parts.append(y_batch.numpy())
        pred_parts.append(prediction.detach().cpu().numpy())

    if not true_parts:
        raise RuntimeError("No prediction batches were processed.")

    return (
        np.concatenate(true_parts, axis=0),
        np.concatenate(pred_parts, axis=0),
    )


# =====================================================================
# Checkpoint helpers
# =====================================================================

def atomic_torch_save(obj, path: Path) -> None:
    """Save via a temporary file and atomically replace the destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(path.name + ".tmp")

    torch.save(obj, temp_path)
    temp_path.replace(path)


def save_best_checkpoint(
    path: Path,
    model: PointwiseMLP,
    epoch: int,
    val_loss: float,
) -> None:
    """Save the current best MLP model."""

    checkpoint = {
        "model_name": "PointwiseMLP",
        "input_dim": model.input_dim,
        "hidden_dims": model.hidden_dims,
        "output_dim": model.output_dim,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "model_state_dict": model.state_dict(),
    }

    atomic_torch_save(checkpoint, path)


def save_training_checkpoint(
    path: Path,
    model: PointwiseMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    best_epoch: int,
    epochs_without_improvement: int,
    batch_size: int,
    train_rows: int,
    val_rows: int,
) -> None:
    """Save the latest completed MLP training state for resume."""

    checkpoint = {
        "checkpoint_type": "training_resume",
        "model_name": "PointwiseMLP",
        "input_dim": model.input_dim,
        "hidden_dims": model.hidden_dims,
        "output_dim": model.output_dim,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "batch_size": int(batch_size),
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    atomic_torch_save(checkpoint, path)


def load_training_checkpoint(
    path: Path,
    model: PointwiseMLP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_batch_size: int,
    expected_train_rows: int,
    expected_val_rows: int,
):
    """Restore model, optimizer, and early-stopping state."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    required_keys = (
        "checkpoint_type",
        "model_name",
        "input_dim",
        "hidden_dims",
        "output_dim",
        "epoch",
        "best_val_loss",
        "best_epoch",
        "epochs_without_improvement",
        "batch_size",
        "train_rows",
        "val_rows",
        "model_state_dict",
        "optimizer_state_dict",
    )

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"Resume checkpoint is missing required key: {key}"
            )

    if checkpoint["checkpoint_type"] != "training_resume":
        raise RuntimeError(
            f"Unsupported checkpoint type: {checkpoint['checkpoint_type']!r}"
        )

    if checkpoint["model_name"] != "PointwiseMLP":
        raise RuntimeError(
            f"Resume checkpoint model mismatch: {checkpoint['model_name']!r}"
        )

    if int(checkpoint["input_dim"]) != int(model.input_dim):
        raise RuntimeError("Resume checkpoint input dimension mismatch.")

    if tuple(checkpoint["hidden_dims"]) != tuple(model.hidden_dims):
        raise RuntimeError(
            "Resume checkpoint hidden-layer configuration mismatch."
        )

    if int(checkpoint["output_dim"]) != int(model.output_dim):
        raise RuntimeError("Resume checkpoint output dimension mismatch.")

    if int(checkpoint["batch_size"]) != int(expected_batch_size):
        raise RuntimeError(
            "Resume checkpoint batch-size mismatch: "
            f"checkpoint={checkpoint['batch_size']}, "
            f"requested={expected_batch_size}"
        )

    if int(checkpoint["train_rows"]) != int(expected_train_rows):
        raise RuntimeError(
            "Resume checkpoint train-row mismatch: "
            f"checkpoint={checkpoint['train_rows']}, "
            f"current={expected_train_rows}"
        )

    if int(checkpoint["val_rows"]) != int(expected_val_rows):
        raise RuntimeError(
            "Resume checkpoint val-row mismatch: "
            f"checkpoint={checkpoint['val_rows']}, "
            f"current={expected_val_rows}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    completed_epoch = int(checkpoint["epoch"])

    return (
        completed_epoch + 1,
        float(checkpoint["best_val_loss"]),
        int(checkpoint["best_epoch"]),
        int(checkpoint["epochs_without_improvement"]),
        checkpoint,
    )


# =====================================================================
# Training
# =====================================================================

def train(args) -> None:
    set_seed(args.seed)

    data_root = Path(
        args.data_root if args.data_root is not None else DATA_ROOT
    ).resolve()

    csv_dir = data_root / "05_cfd_csv"
    model_path = Path(args.model_path).resolve()
    scaler_path = Path(args.scaler_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()

    if not csv_dir.exists():
        raise FileNotFoundError(f"CFD CSV directory not found:\n{csv_dir}")

    if args.resume:
        for required_path, name in (
            (checkpoint_path, "Resume checkpoint"),
            (scaler_path, "Scaler"),
            (model_path, "Best-model checkpoint"),
        ):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"{name} required for resume was not found:\n{required_path}"
                )

    elif not args.overwrite:
        existing = [
            path
            for path in (model_path, scaler_path, checkpoint_path)
            if path.exists()
        ]

        if existing:
            existing_text = "\n".join(str(path) for path in existing)
            raise FileExistsError(
                "Training output already exists:\n"
                f"{existing_text}\n\n"
                "Use --overwrite for a fresh run or --resume to continue."
            )

    device = get_device()

    print("=" * 78)
    print("POINT-WISE MLP TRAINING")
    print("=" * 78)
    print(f"Data root        : {data_root}")
    print(f"CSV directory    : {csv_dir}")
    print(f"Device           : {device}")

    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")

    print(f"PyTorch          : {torch.__version__}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Epochs           : {args.epochs}")
    print(f"Learning rate    : {args.learning_rate}")
    print(f"Weight decay     : {args.weight_decay}")
    print(f"Patience         : {args.patience}")
    print(f"Seed             : {args.seed}")
    print(f"Best model path  : {model_path}")
    print(f"Scaler path      : {scaler_path}")
    print(f"Resume checkpoint: {checkpoint_path}")
    print(f"Resume mode      : {args.resume}")

    if args.max_train_batches is not None:
        print(
            f"Max train batches: {args.max_train_batches} (DEBUG ONLY)"
        )

    if args.max_val_batches is not None:
        print(f"Max val batches  : {args.max_val_batches} (DEBUG ONLY)")

    print("=" * 78)

    # =================================================================
    # Load raw data
    # =================================================================

    x_train, y_train = load_split(csv_dir, "train")
    x_val, y_val = load_split(csv_dir, "val")

    train_rows = int(x_train.shape[0])
    val_rows = int(x_val.shape[0])

    print()
    print("=" * 78)
    print("DATASET")
    print("=" * 78)
    print(f"Train rows : {train_rows:,}")
    print(f"Val rows   : {val_rows:,}")
    print("=" * 78)

    # =================================================================
    # Scaler
    # =================================================================

    if args.resume:
        print()
        print("=" * 78)
        print("LOAD EXISTING MLP SCALER FOR RESUME")
        print("=" * 78)

        scaler = CFDScaler.load(scaler_path)
        scaler.print_statistics()
        print(f"Scaler loaded : {scaler_path}")

    else:
        print()
        print("=" * 78)
        print("FIT MLP SCALER FROM TRAIN DATA")
        print("=" * 78)

        scaler = CFDScaler().fit(x_train, y_train)
        scaler.print_statistics()
        scaler.save(scaler_path)
        print(f"Scaler saved : {scaler_path}")

    # =================================================================
    # Normalize
    # =================================================================

    print()
    print("[NORMALIZE]")

    x_train = scaler.transform_input(x_train).astype(
        np.float32, copy=False
    )
    y_train = scaler.transform_target(y_train).astype(
        np.float32, copy=False
    )
    x_val = scaler.transform_input(x_val).astype(
        np.float32, copy=False
    )
    y_val = scaler.transform_target(y_val).astype(
        np.float32, copy=False
    )

    print("Train / validation normalization complete.")

    # =================================================================
    # DataLoaders
    # =================================================================

    train_loader = build_loader(
        x_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
    )

    val_loader = build_loader(
        x_val,
        y_val,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )

    del x_train
    del y_train
    del x_val
    del y_val

    # =================================================================
    # Model / optimizer
    # =================================================================

    model = PointwiseMLP(
        input_dim=4,
        hidden_dims=(256, 256, 256, 256),
        output_dim=3,
    ).to(device)

    print()
    print("=" * 78)
    print("MODEL")
    print("=" * 78)
    print(model)
    print()
    print(f"Trainable parameters : {count_parameters(model):,}")

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # =================================================================
    # Fresh or resume state
    # =================================================================

    if args.resume:
        (
            start_epoch,
            best_val_loss,
            best_epoch,
            epochs_without_improvement,
            resume_checkpoint,
        ) = load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            device,
            expected_batch_size=args.batch_size,
            expected_train_rows=train_rows,
            expected_val_rows=val_rows,
        )

        print()
        print("=" * 78)
        print("RESUME TRAINING STATE")
        print("=" * 78)
        print(f"Completed epoch   : {resume_checkpoint['epoch']}")
        print(f"Resume from epoch : {start_epoch}")
        print(f"Best epoch        : {best_epoch}")
        print(f"Best val loss     : {best_val_loss:.8f}")
        print(f"No-improve count  : {epochs_without_improvement}")
        print(f"Optimizer LR      : {optimizer.param_groups[0]['lr']}")
        print(f"Checkpoint        : {checkpoint_path}")
        print("=" * 78)

    else:
        start_epoch = 1
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

    # =================================================================
    # Train
    # =================================================================

    print()
    print("=" * 78)
    print("TRAINING")
    print("=" * 78)

    training_start = time.perf_counter()

    if epochs_without_improvement >= args.patience:
        print()
        print(
            "[EARLY STOP ALREADY REACHED] "
            "Checkpoint already satisfies the requested patience."
        )

    elif start_epoch > args.epochs:
        print()
        print(
            "[NO TRAINING NEEDED] "
            f"Checkpoint already completed epoch {start_epoch - 1}, "
            f"while --epochs={args.epochs}."
        )

    else:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.perf_counter()

            print()
            print(f"[EPOCH {epoch}/{args.epochs}]", flush=True)

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                log_every=args.log_every,
                max_batches=args.max_train_batches,
            )

            val_loss = evaluate_loss(
                model,
                val_loader,
                criterion,
                device,
                max_batches=args.max_val_batches,
            )

            epoch_time = time.perf_counter() - epoch_start
            improved = val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                save_best_checkpoint(
                    model_path,
                    model,
                    epoch,
                    val_loss,
                )

                marker = "  *BEST*"

            else:
                epochs_without_improvement += 1
                marker = ""

            print()
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f} | "
                f"{epoch_time:.1f}s"
                f"{marker}",
                flush=True,
            )

            if device.type == "cuda":
                allocated = (
                    torch.cuda.max_memory_allocated()
                    / (1024 ** 3)
                )

                print(
                    f"GPU peak allocated: {allocated:.2f} GB",
                    flush=True,
                )

                torch.cuda.reset_peak_memory_stats()

            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                batch_size=args.batch_size,
                train_rows=train_rows,
                val_rows=val_rows,
            )

            print(
                f"Resume checkpoint saved: {checkpoint_path}",
                flush=True,
            )

            if epochs_without_improvement >= args.patience:
                print()
                print(
                    "[EARLY STOP] "
                    f"No validation improvement for {args.patience} epochs."
                )
                break

    training_time = time.perf_counter() - training_start

    # =================================================================
    # Reload best checkpoint
    # =================================================================

    if not model_path.exists():
        raise FileNotFoundError(
            f"Best-model checkpoint was not created:\n{model_path}"
        )

    best_checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(best_checkpoint["model_state_dict"])

    best_epoch = int(best_checkpoint["epoch"])
    best_val_loss = float(best_checkpoint["val_loss"])

    # =================================================================
    # Physical-unit validation metrics
    # =================================================================

    print()
    print("=" * 78)
    print("BEST MODEL VALIDATION")
    print("=" * 78)

    y_val_norm, y_pred_norm = predict_loader(
        model,
        val_loader,
        device,
        max_batches=args.max_val_batches,
    )

    y_val_physical = scaler.inverse_target(y_val_norm)
    y_pred_physical = scaler.inverse_target(y_pred_norm)

    metrics = calculate_metrics(
        y_val_physical,
        y_pred_physical,
    )

    print_metrics(metrics)

    # =================================================================
    # Summary
    # =================================================================

    print()
    print("=" * 78)
    print("MLP TRAINING COMPLETE")
    print("=" * 78)
    print(f"Best epoch       : {best_epoch}")
    print(f"Best val loss    : {best_val_loss:.8f}")
    print(f"Training runtime : {training_time / 60.0:.1f} min")
    print(f"Train rows       : {train_rows:,}")
    print(f"Val rows         : {val_rows:,}")
    print(f"Best model       : {model_path}")
    print(f"Scaler           : {scaler_path}")
    print(f"Last checkpoint  : {checkpoint_path}")
    print("=" * 78)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the point-wise MLP baseline on CFD wall-field CSV data."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional ai-cfd-data root. Default: project_paths.DATA_ROOT"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
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
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Latest MLP training-state checkpoint. "
            "Saved after every completed epoch."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume model / optimizer / early-stopping state from "
            "--checkpoint-path. Existing scaler and best_model.pt are loaded."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Start a fresh run and allow existing output files to be replaced."
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Print training-batch progress every N batches.",
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY: stop each training epoch after this many batches."
        ),
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY: stop validation and validation-metric prediction "
            "after this many batches."
        ),
    )

    return parser.parse_args()


# =====================================================================
# Argument validation
# =====================================================================

def validate_args(args) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")

    if args.patience <= 0:
        raise ValueError("--patience must be positive.")

    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")

    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be positive.")

    if args.max_val_batches is not None and args.max_val_batches <= 0:
        raise ValueError("--max-val-batches must be positive.")

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")


# =====================================================================
# Entry point
# =====================================================================

def main():
    args = parse_args()
    validate_args(args)
    train(args)


if __name__ == "__main__":
    main()
