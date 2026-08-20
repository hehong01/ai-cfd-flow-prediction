"""
Train the point-wise MLP baseline for CFD surface-field prediction.

Input per CFD wall node:
    [x, y, z, velocity]

Target per CFD wall node:
    [HTC, wall_shear]

Dataset split:
    train : face_0001 ~ face_0080
    val   : face_0081 ~ face_0090
    test  : face_0091 ~ face_0100

Important:
    - MLP uses ALL original CFD wall nodes.
    - DGCNN FPS preprocessing is NOT used here.
    - Scaler statistics are fitted using TRAIN data only.
    - Validation data is never used to fit normalization statistics.

Outputs:
    weights/mlp/best_model.pt
    weights/mlp/scalers.npz
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

WEIGHT_DIR = TRAINING_ROOT / "weights" / "mlp"

DEFAULT_MODEL_PATH = WEIGHT_DIR / "best_model.pt"
DEFAULT_SCALER_PATH = WEIGHT_DIR / "scalers.npz"


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

    raise ValueError(
        f"Invalid split: {split}"
    )


def build_csv_list(
    csv_dir: Path,
    split: str,
) -> list[tuple[Path, float]]:
    """
    Build the CSV list for one split.

    Returns:
        [
            (csv_path, velocity),
            ...
        ]
    """

    samples = []

    for face_num in get_face_range(split):

        face_id = f"face_{face_num:04d}"

        for speed_name, velocity in SPEEDS:

            csv_path = (
                csv_dir
                / f"{face_id}_{speed_name}.csv"
            )

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Missing CFD CSV:\n{csv_path}"
                )

            samples.append(
                (
                    csv_path,
                    velocity,
                )
            )

    return samples


# =====================================================================
# Raw CSV loading
# =====================================================================

def load_one_csv(
    csv_path: Path,
    velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one CFD CSV.

    Original Fluent columns:

        0 nodenumber
        1 x-coordinate
        2 y-coordinate
        3 z-coordinate
        4 pressure
        5 temperature
        6 y-plus
        7 wall-shear
        8 heat-flux
        9 heat-transfer-coef

    Returns:

        X : (N, 4)
            [x, y, z, velocity]

        Y : (N, 2)
            [HTC, wall_shear]
    """

    # Read only:
    #
    # x, y, z, wall-shear, HTC
    data = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3, 7, 9),
        dtype=np.float32,
    )

    if data.ndim == 1:
        data = data.reshape(
            1,
            5,
        )

    if data.shape[1] != 5:
        raise ValueError(
            f"{csv_path.name}: "
            f"unexpected loaded shape {data.shape}"
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"{csv_path.name}: "
            f"NaN or Inf found."
        )

    num_points = len(data)

    # -------------------------------------------------------------
    # Input
    #
    # [x, y, z, velocity]
    # -------------------------------------------------------------

    velocity_column = np.full(
        (
            num_points,
            1,
        ),
        velocity,
        dtype=np.float32,
    )

    x = np.concatenate(
        (
            data[:, 0:3],
            velocity_column,
        ),
        axis=1,
    )

    # -------------------------------------------------------------
    # Target
    #
    # loaded:
    # data[:, 3] -> wall shear
    # data[:, 4] -> HTC
    #
    # desired:
    # [HTC, wall_shear]
    # -------------------------------------------------------------

    y = np.stack(
        (
            data[:, 4],
            data[:, 3],
        ),
        axis=1,
    )

    return x, y


def load_split(
    csv_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load every CFD node belonging to one split.

    MLP treats every wall node as an independent sample.
    """

    csv_samples = build_csv_list(
        csv_dir,
        split,
    )

    print()
    print(
        f"[LOAD {split.upper()}]"
    )

    print(
        f"CSV files : {len(csv_samples)}"
    )

    x_parts = []
    y_parts = []

    total_rows = 0

    start = time.perf_counter()

    for index, (
        csv_path,
        velocity,
    ) in enumerate(
        csv_samples,
        start=1,
    ):

        x, y = load_one_csv(
            csv_path,
            velocity,
        )

        x_parts.append(x)
        y_parts.append(y)

        total_rows += len(x)

        if (
            index == 1
            or index % 25 == 0
            or index == len(csv_samples)
        ):
            print(
                f"  {index:3d}/{len(csv_samples)}  "
                f"{csv_path.name:<24} "
                f"rows={len(x):5d}  "
                f"total={total_rows:,}"
            )

    x_all = np.concatenate(
        x_parts,
        axis=0,
    )

    y_all = np.concatenate(
        y_parts,
        axis=0,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if x_all.shape[1] != 4:
        raise RuntimeError(
            f"{split}: invalid X shape "
            f"{x_all.shape}"
        )

    if y_all.shape[1] != 2:
        raise RuntimeError(
            f"{split}: invalid Y shape "
            f"{y_all.shape}"
        )

    print(
        f"{split.upper()} loaded:"
    )

    print(
        f"  X shape : {x_all.shape}"
    )

    print(
        f"  Y shape : {y_all.shape}"
    )

    print(
        f"  runtime : {elapsed:.1f} s"
    )

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

    x_tensor = torch.from_numpy(
        np.asarray(
            x,
            dtype=np.float32,
        )
    )

    y_tensor = torch.from_numpy(
        np.asarray(
            y,
            dtype=np.float32,
        )
    )

    dataset = TensorDataset(
        x_tensor,
        y_tensor,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    return loader


# =====================================================================
# One training epoch
# =====================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:

        x_batch = x_batch.to(
            device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            x_batch
        )

        loss = criterion(
            prediction,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        batch_size = len(x_batch)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# =====================================================================
# Validation
# =====================================================================

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Calculate mean validation loss."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:

        x_batch = x_batch.to(
            device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )

        prediction = model(
            x_batch
        )

        loss = criterion(
            prediction,
            y_batch,
        )

        batch_size = len(x_batch)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# =====================================================================
# Full prediction
# =====================================================================

@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict an entire DataLoader.

    Returns normalized:
        y_true
        y_pred
    """

    model.eval()

    true_parts = []
    pred_parts = []

    for x_batch, y_batch in loader:

        x_batch = x_batch.to(
            device,
            non_blocking=True,
        )

        prediction = model(
            x_batch
        )

        true_parts.append(
            y_batch.numpy()
        )

        pred_parts.append(
            prediction
            .detach()
            .cpu()
            .numpy()
        )

    y_true = np.concatenate(
        true_parts,
        axis=0,
    )

    y_pred = np.concatenate(
        pred_parts,
        axis=0,
    )

    return y_true, y_pred


# =====================================================================
# Checkpoint
# =====================================================================

def save_checkpoint(
    path: Path,
    model: PointwiseMLP,
    epoch: int,
    val_loss: float,
) -> None:
    """Save best MLP model."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_name": "PointwiseMLP",
        "input_dim": model.input_dim,
        "hidden_dims": model.hidden_dims,
        "output_dim": model.output_dim,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": (
            model.state_dict()
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


# =====================================================================
# Training
# =====================================================================

def train(args) -> None:

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    set_seed(
        args.seed
    )

    # -------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------

    data_root = (
        args.data_root
        if args.data_root is not None
        else DATA_ROOT
    )

    data_root = Path(
        data_root
    ).resolve()

    csv_dir = (
        data_root
        / "05_cfd_csv"
    )

    model_path = Path(
        args.model_path
    )

    scaler_path = Path(
        args.scaler_path
    )

    if not csv_dir.exists():
        raise FileNotFoundError(
            f"CFD CSV directory not found:\n"
            f"{csv_dir}"
        )

    if not args.overwrite:

        existing = [
            path
            for path in (
                model_path,
                scaler_path,
            )
            if path.exists()
        ]

        if existing:
            text = "\n".join(
                str(path)
                for path in existing
            )

            raise FileExistsError(
                "Training output already exists:\n"
                f"{text}\n\n"
                "Use --overwrite to replace."
            )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = get_device()

    print("=" * 78)
    print("POINT-WISE MLP TRAINING")
    print("=" * 78)

    print(
        f"Data root      : {data_root}"
    )

    print(
        f"CSV directory  : {csv_dir}"
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

    print(
        f"Epochs         : {args.epochs}"
    )

    print(
        f"Learning rate  : {args.learning_rate}"
    )

    print(
        f"Weight decay   : {args.weight_decay}"
    )

    print(
        f"Patience       : {args.patience}"
    )

    print(
        f"Seed           : {args.seed}"
    )

    print("=" * 78)

    # =================================================================
    # Load raw data
    # =================================================================

    x_train, y_train = load_split(
        csv_dir,
        "train",
    )

    x_val, y_val = load_split(
        csv_dir,
        "val",
    )

    # =================================================================
    # Fit TRAIN scaler only
    # =================================================================

    print()
    print("=" * 78)
    print("FIT MLP SCALER FROM TRAIN DATA")
    print("=" * 78)

    scaler = CFDScaler().fit(
        x_train,
        y_train,
    )

    scaler.print_statistics()

    scaler.save(
        scaler_path
    )

    print(
        f"Scaler saved : {scaler_path}"
    )

    # =================================================================
    # Normalize
    # =================================================================

    print()
    print("[NORMALIZE]")

    x_train = scaler.transform_input(
        x_train
    ).astype(
        np.float32,
        copy=False,
    )

    y_train = scaler.transform_target(
        y_train
    ).astype(
        np.float32,
        copy=False,
    )

    x_val = scaler.transform_input(
        x_val
    ).astype(
        np.float32,
        copy=False,
    )

    y_val = scaler.transform_target(
        y_val
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        "Train / validation normalization complete."
    )

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

    # We no longer need the NumPy X arrays after DataLoader creation.
    del x_train
    del y_train
    del x_val
    del y_val

    # =================================================================
    # Model
    # =================================================================

    model = PointwiseMLP(
        input_dim=4,
        hidden_dims=(
            128,
            128,
            64,
        ),
        output_dim=2,
    ).to(
        device
    )

    print()
    print("=" * 78)
    print("MODEL")
    print("=" * 78)

    print(model)

    print(
        f"Trainable parameters : "
        f"{count_parameters(model):,}"
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # =================================================================
    # Train
    # =================================================================

    print()
    print("=" * 78)
    print("TRAINING")
    print("=" * 78)

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0
    epochs_without_improvement = 0

    training_start = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        improved = (
            val_loss
            < best_val_loss
        )

        if improved:

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                model_path,
                model,
                epoch,
                val_loss,
            )

            marker = "  *BEST*"

        else:

            epochs_without_improvement += 1

            marker = ""

        print(
            f"Epoch "
            f"{epoch:3d}/{args.epochs} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"{epoch_time:.1f}s"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print()
            print(
                f"[EARLY STOP] "
                f"No validation improvement for "
                f"{args.patience} epochs."
            )

            break

    training_time = (
        time.perf_counter()
        - training_start
    )

    # =================================================================
    # Load best checkpoint
    # =================================================================

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # =================================================================
    # Physical-unit validation metrics
    # =================================================================

    print()
    print("=" * 78)
    print("BEST MODEL VALIDATION")
    print("=" * 78)

    y_val_norm, y_pred_norm = (
        predict_loader(
            model,
            val_loader,
            device,
        )
    )

    y_val_physical = (
        scaler.inverse_target(
            y_val_norm
        )
    )

    y_pred_physical = (
        scaler.inverse_target(
            y_pred_norm
        )
    )

    metrics = calculate_metrics(
        y_val_physical,
        y_pred_physical,
    )

    print_metrics(
        metrics
    )

    # =================================================================
    # Summary
    # =================================================================

    print()
    print("=" * 78)
    print("MLP TRAINING COMPLETE")
    print("=" * 78)

    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Best val loss    : "
        f"{best_val_loss:.8f}"
    )

    print(
        f"Training runtime : "
        f"{training_time / 60.0:.1f} min"
    )

    print(
        f"Best model       : "
        f"{model_path}"
    )

    print(
        f"Scaler           : "
        f"{scaler_path}"
    )

    print("=" * 78)


# =====================================================================
# CLI
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train the point-wise MLP baseline "
            "on CFD wall-field CSV data."
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
        "--overwrite",
        action="store_true",
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

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be positive."
        )

    if args.patience <= 0:
        raise ValueError(
            "--patience must be positive."
        )

    train(
        args
    )


if __name__ == "__main__":
    main()