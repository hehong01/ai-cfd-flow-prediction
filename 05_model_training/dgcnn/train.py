"""
Train the DGCNN CFD regression model.

Input per point:
    [x, y, z, velocity]

Target per point:
    [HTC, wall_shear]

Dataset split:
    train : face_0001 ~ face_0080
            80 faces x 3 velocities = 240 samples

    val   : face_0081 ~ face_0090
            10 faces x 3 velocities = 30 samples

    test  : face_0091 ~ face_0100
            NOT used during training

DGCNN input:
    one CSV = one point-cloud sample

    X:
        (7000, 4)

    Y:
        (7000, 2)

The deterministic FPS-selected 7000-point samples are loaded through:

    04_cfd_dataset/dataset.py

Normalization:
    DGCNN-specific scaler statistics are fitted from TRAIN data only.

Outputs:
    weights/dgcnn/best_model.pt
    weights/dgcnn/scalers.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)


# =====================================================================
# Project paths
# =====================================================================

THIS_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = THIS_DIR.parent
GITHUB_ROOT = TRAINING_ROOT.parent
COMMON_DIR = TRAINING_ROOT / "common"

DATASET_MODULE_PATH = (
    GITHUB_ROOT
    / "04_cfd_dataset"
    / "dataset.py"
)

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


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0

DEFAULT_PATIENCE = 15
DEFAULT_SEED = 42

DEFAULT_K = 20
DEFAULT_KNN_CHUNK_SIZE = 1024

DEFAULT_LOG_EVERY = 10

EXPECTED_TRAIN_SAMPLES = 240
EXPECTED_VAL_SAMPLES = 30

WEIGHT_DIR = (
    TRAINING_ROOT
    / "weights"
    / "dgcnn"
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
# Reproducibility
# =====================================================================

def set_seed(
    seed: int,
) -> None:
    """Set Python / NumPy / PyTorch random seeds."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
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
# Existing 04_cfd_dataset loader
# =====================================================================

def load_dataset_module(
    data_root: Path,
):
    """
    Dynamically import:

        04_cfd_dataset/dataset.py

    The folder starts with a number, so importing it as a normal
    Python package is inconvenient. Loading by file path avoids that.

    If --data-root is supplied, CSV_DIR inside the imported module
    is redirected to:

        <data_root>/05_cfd_csv

    This allows the same training code to run locally or on Colab.
    """

    if not DATASET_MODULE_PATH.exists():
        raise FileNotFoundError(
            "DGCNN dataset module not found:\n"
            f"{DATASET_MODULE_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "cfd_dataset_module",
        DATASET_MODULE_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not create import specification for:\n"
            f"{DATASET_MODULE_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    # -------------------------------------------------------------
    # Allow local / Colab data-root switching without hardcoding.
    # -------------------------------------------------------------

    csv_dir = (
        data_root
        / "05_cfd_csv"
    )

    if hasattr(
        module,
        "DATA_ROOT",
    ):
        module.DATA_ROOT = (
            data_root
        )

    if hasattr(
        module,
        "CSV_DIR",
    ):
        module.CSV_DIR = (
            csv_dir
        )

    return module


def create_base_dataset(
    dataset_module,
    split: str,
):
    """
    Create DGCNNCFDDataset from the existing dataset.py.

    The current project loader is expected to provide:

        DGCNNCFDDataset

    This helper inspects the constructor so minor optional-argument
    differences do not require duplicating the dataset implementation.
    """

    if not hasattr(
        dataset_module,
        "DGCNNCFDDataset",
    ):
        raise AttributeError(
            "04_cfd_dataset/dataset.py does not contain "
            "DGCNNCFDDataset."
        )

    dataset_class = (
        dataset_module
        .DGCNNCFDDataset
    )

    signature = inspect.signature(
        dataset_class
    )

    parameters = (
        signature.parameters
    )

    kwargs = {}

    # -------------------------------------------------------------
    # Split
    # -------------------------------------------------------------

    if "split" in parameters:

        kwargs[
            "split"
        ] = split

    elif "split_name" in parameters:

        kwargs[
            "split_name"
        ] = split

    else:

        raise TypeError(
            "DGCNNCFDDataset constructor does not expose "
            "'split' or 'split_name'."
        )

    # -------------------------------------------------------------
    # We do not need sample IDs during training.
    # -------------------------------------------------------------

    if (
        "return_sample_id"
        in parameters
    ):
        kwargs[
            "return_sample_id"
        ] = False

    # -------------------------------------------------------------
    # Use the existing fixed DGCNN point count.
    # -------------------------------------------------------------

    if "num_points" in parameters:

        num_points = getattr(
            dataset_module,
            "NUM_POINTS",
            7000,
        )

        kwargs[
            "num_points"
        ] = int(
            num_points
        )

    dataset = dataset_class(
        **kwargs
    )

    return dataset


# =====================================================================
# Dataset item validation
# =====================================================================

def extract_xy(
    item,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract X and Y from one DGCNNCFDDataset item.

    Expected normal form:

        (X, Y)

    The loader may optionally return:

        (X, Y, sample_id)

    in which case sample_id is ignored here.
    """

    if isinstance(
        item,
        (tuple, list),
    ):

        if len(item) < 2:
            raise ValueError(
                "Dataset item must contain at least X and Y."
            )

        x = item[0]
        y = item[1]

    elif isinstance(
        item,
        dict,
    ):

        # Defensive compatibility.
        if "x" in item:
            x = item["x"]
        elif "X" in item:
            x = item["X"]
        else:
            raise KeyError(
                "Dataset dictionary does not contain X."
            )

        if "y" in item:
            y = item["y"]
        elif "Y" in item:
            y = item["Y"]
        else:
            raise KeyError(
                "Dataset dictionary does not contain Y."
            )

    else:

        raise TypeError(
            "Unsupported DGCNN dataset item type: "
            f"{type(item)}"
        )

    if not torch.is_tensor(
        x
    ):
        x = torch.as_tensor(
            x
        )

    if not torch.is_tensor(
        y
    ):
        y = torch.as_tensor(
            y
        )

    x = x.float()
    y = y.float()

    if x.ndim != 2:
        raise ValueError(
            f"Expected X shape (N,4), "
            f"found {tuple(x.shape)}."
        )

    if y.ndim != 2:
        raise ValueError(
            f"Expected Y shape (N,2), "
            f"found {tuple(y.shape)}."
        )

    if x.shape[-1] != 4:
        raise ValueError(
            f"Expected X feature dimension 4, "
            f"found {x.shape[-1]}."
        )

    if y.shape[-1] != 2:
        raise ValueError(
            f"Expected Y target dimension 2, "
            f"found {y.shape[-1]}."
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "X and Y point counts differ:\n"
            f"X = {tuple(x.shape)}\n"
            f"Y = {tuple(y.shape)}"
        )

    if not torch.isfinite(
        x
    ).all():
        raise ValueError(
            "X contains NaN or Inf."
        )

    if not torch.isfinite(
        y
    ).all():
        raise ValueError(
            "Y contains NaN or Inf."
        )

    return x, y


# =====================================================================
# Fit DGCNN-specific scaler
# =====================================================================

def fit_dgcnn_scaler(
    train_dataset,
) -> CFDScaler:
    """
    Fit normalization statistics using the exact TRAIN points
    consumed by the DGCNN dataset.

    For the current dataset:

        240 train samples
        x
        7000 FPS-selected points

        =
        1,680,000 points

    These statistics are independent of the MLP scaler because the
    MLP uses all original CFD wall nodes while DGCNN uses FPS points.
    """

    print()
    print("=" * 78)
    print("FIT DGCNN SCALER FROM TRAIN DATA")
    print("=" * 78)

    x_parts = []
    y_parts = []

    total_points = 0

    start = time.perf_counter()

    num_samples = len(
        train_dataset
    )

    for index in range(
        num_samples
    ):

        x, y = extract_xy(
            train_dataset[
                index
            ]
        )

        x_np = (
            x
            .detach()
            .cpu()
            .numpy()
        )

        y_np = (
            y
            .detach()
            .cpu()
            .numpy()
        )

        x_parts.append(
            x_np
        )

        y_parts.append(
            y_np
        )

        total_points += (
            x.shape[0]
        )

        current = (
            index + 1
        )

        if (
            current == 1
            or current % 25 == 0
            or current == num_samples
        ):

            print(
                f"  {current:3d}/{num_samples}  "
                f"points={x.shape[0]:5d}  "
                f"total={total_points:,}"
            )

    x_train = np.concatenate(
        x_parts,
        axis=0,
    )

    y_train = np.concatenate(
        y_parts,
        axis=0,
    )

    del x_parts
    del y_parts

    scaler = CFDScaler().fit(
        x_train,
        y_train,
    )

    del x_train
    del y_train

    elapsed = (
        time.perf_counter()
        - start
    )

    print()

    print(
        f"Samples used : "
        f"{num_samples}"
    )

    print(
        f"Points used  : "
        f"{total_points:,}"
    )

    print(
        f"Runtime      : "
        f"{elapsed:.1f} s"
    )

    print()

    scaler.print_statistics()

    return scaler


# =====================================================================
# Normalized dataset wrapper
# =====================================================================

class NormalizedDGCNNDataset(
    Dataset
):
    """
    Apply the saved TRAIN scaler to samples returned by the existing
    DGCNNCFDDataset.

    Normalization is performed on-the-fly.

    Optional point_limit exists only for quick smoke tests.

    Final training should use:
        point_limit = None

    so that all 7000 FPS points are used.
    """

    def __init__(
        self,
        base_dataset,
        scaler: CFDScaler,
        point_limit: int | None = None,
    ):
        self.base_dataset = (
            base_dataset
        )

        self.point_limit = (
            point_limit
        )

        self.input_mean = torch.tensor(
            scaler.input_mean,
            dtype=torch.float32,
        )

        self.input_std = torch.tensor(
            scaler.input_std,
            dtype=torch.float32,
        )

        self.target_mean = torch.tensor(
            scaler.target_mean,
            dtype=torch.float32,
        )

        self.target_std = torch.tensor(
            scaler.target_std,
            dtype=torch.float32,
        )

    # -----------------------------------------------------------------

    def __len__(
        self,
    ):
        return len(
            self.base_dataset
        )

    # -----------------------------------------------------------------

    def __getitem__(
        self,
        index,
    ):

        x, y = extract_xy(
            self.base_dataset[
                index
            ]
        )

        # ---------------------------------------------------------
        # Debug / smoke test only.
        #
        # Final training uses all 7000 points.
        # ---------------------------------------------------------

        if (
            self.point_limit
            is not None
        ):

            if (
                self.point_limit
                > x.shape[0]
            ):
                raise ValueError(
                    f"point_limit={self.point_limit} "
                    f"is greater than sample point count "
                    f"{x.shape[0]}."
                )

            x = x[
                :self.point_limit
            ]

            y = y[
                :self.point_limit
            ]

        # ---------------------------------------------------------
        # Preserve raw physical xyz BEFORE feature standardization.
        #
        # The first DGCNN graph must use physical Euclidean geometry.
        # Standardizing x/y/z independently would distort that metric.
        # ---------------------------------------------------------

        raw_xyz = (
            x[
                :,
                :3,
            ]
            .clone()
        )

        # ---------------------------------------------------------
        # Standardization
        #
        # Edge values still use normalized [x,y,z,velocity].
        # ---------------------------------------------------------

        x = (
            x
            - self.input_mean
        ) / self.input_std

        y = (
            y
            - self.target_mean
        ) / self.target_std

        return x, y, raw_xyz


# =====================================================================
# DataLoader
# =====================================================================

def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    """Build PyTorch DataLoader."""

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
        drop_last=False,
        generator=generator,
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

    num_batches = len(
        loader
    )

    start = time.perf_counter()

    for batch_index, (
        x_batch,
        y_batch,
        raw_xyz_batch,
    ) in enumerate(
        loader,
        start=1,
    ):

        # ---------------------------------------------------------
        # Optional smoke-test limit.
        # ---------------------------------------------------------

        if (
            max_batches
            is not None
            and batch_index
            > max_batches
        ):
            break

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

        raw_xyz_batch = raw_xyz_batch.to(
            device,
            non_blocking=True,
        )

        prediction = model(
            x_batch,
            first_knn_xyz=raw_xyz_batch,
        )

        loss = criterion(
            prediction,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            x_batch.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        # ---------------------------------------------------------
        # Progress output.
        # ---------------------------------------------------------

        if (
            batch_index == 1
            or batch_index % log_every == 0
            or batch_index == num_batches
            or (
                max_batches is not None
                and batch_index
                == max_batches
            )
        ):

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"    batch "
                f"{batch_index:4d}/"
                f"{num_batches} | "
                f"loss={loss.item():.6f} | "
                f"{elapsed:.1f}s"
            )

    if total_samples == 0:
        raise RuntimeError(
            "No training batches were processed."
        )

    return (
        total_loss
        / total_samples
    )


# =====================================================================
# Validation loss
# =====================================================================

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Calculate validation loss."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch_index, (
        x_batch,
        y_batch,
        raw_xyz_batch,
    ) in enumerate(
        loader,
        start=1,
    ):

        if (
            max_batches
            is not None
            and batch_index
            > max_batches
        ):
            break

        x_batch = x_batch.to(
            device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )

        raw_xyz_batch = raw_xyz_batch.to(
            device,
            non_blocking=True,
        )

        prediction = model(
            x_batch,
            first_knn_xyz=raw_xyz_batch,
        )

        loss = criterion(
            prediction,
            y_batch,
        )

        batch_size = (
            x_batch.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    if total_samples == 0:
        raise RuntimeError(
            "No validation batches were processed."
        )

    return (
        total_loss
        / total_samples
    )


# =====================================================================
# Full validation prediction
# =====================================================================

@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict normalized targets.

    Returns:
        y_true_norm
        y_pred_norm

    Shapes after concatenation:
        (number_of_points, 2)
    """

    model.eval()

    true_parts = []
    pred_parts = []

    for batch_index, (
        x_batch,
        y_batch,
        raw_xyz_batch,
    ) in enumerate(
        loader,
        start=1,
    ):

        if (
            max_batches
            is not None
            and batch_index
            > max_batches
        ):
            break

        x_batch = x_batch.to(
            device,
            non_blocking=True,
        )

        raw_xyz_batch = raw_xyz_batch.to(
            device,
            non_blocking=True,
        )

        prediction = model(
            x_batch,
            first_knn_xyz=raw_xyz_batch,
        )

        y_true_np = (
            y_batch
            .detach()
            .cpu()
            .numpy()
            .reshape(
                -1,
                2,
            )
        )

        y_pred_np = (
            prediction
            .detach()
            .cpu()
            .numpy()
            .reshape(
                -1,
                2,
            )
        )

        true_parts.append(
            y_true_np
        )

        pred_parts.append(
            y_pred_np
        )

    if not true_parts:
        raise RuntimeError(
            "No prediction batches were processed."
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
    model: DGCNNRegressor,
    epoch: int,
    val_loss: float,
    point_count: int,
) -> None:
    """Save the current best DGCNN checkpoint."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_name": (
            "DGCNNRegressor"
        ),
        "input_dim": (
            model.input_dim
        ),
        "output_dim": (
            model.output_dim
        ),
        "k": (
            model.k
        ),
        "knn_chunk_size": (
            model.knn_chunk_size
        ),
        "point_count": (
            point_count
        ),
        "first_knn_space": (
            "raw_xyz"
        ),
        "epoch": (
            epoch
        ),
        "val_loss": (
            val_loss
        ),
        "model_state_dict": (
            model.state_dict()
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


# =====================================================================
# Main training
# =====================================================================

def train(
    args,
) -> None:

    # -----------------------------------------------------------------
    # Seed
    # -----------------------------------------------------------------

    set_seed(
        args.seed
    )

    # -----------------------------------------------------------------
    # Data root
    # -----------------------------------------------------------------

    data_root = (
        Path(args.data_root)
        if args.data_root
        is not None
        else Path(DATA_ROOT)
    ).resolve()

    csv_dir = (
        data_root
        / "05_cfd_csv"
    )

    if not csv_dir.exists():
        raise FileNotFoundError(
            "CFD CSV directory not found:\n"
            f"{csv_dir}"
        )

    # -----------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------

    model_path = Path(
        args.model_path
    ).resolve()

    scaler_path = Path(
        args.scaler_path
    ).resolve()

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

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = get_device()

    print("=" * 78)
    print("DGCNN CFD TRAINING")
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
        f"Dataset module  : "
        f"{DATASET_MODULE_PATH}"
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

    print(
        f"Epochs          : "
        f"{args.epochs}"
    )

    print(
        f"Learning rate   : "
        f"{args.learning_rate}"
    )

    print(
        f"Weight decay    : "
        f"{args.weight_decay}"
    )

    print(
        f"k               : "
        f"{args.k}"
    )

    print(
        f"kNN chunk size  : "
        f"{args.knn_chunk_size}"
    )

    print(
        f"Patience        : "
        f"{args.patience}"
    )

    print(
        f"Seed            : "
        f"{args.seed}"
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
    # Load existing dataset implementation
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

    # -----------------------------------------------------------------
    # Base train / validation datasets
    # -----------------------------------------------------------------

    train_base = create_base_dataset(
        dataset_module,
        "train",
    )

    val_base = create_base_dataset(
        dataset_module,
        "val",
    )

    print()
    print("=" * 78)
    print("DATASET")
    print("=" * 78)

    print(
        f"Train samples : "
        f"{len(train_base)}"
    )

    print(
        f"Val samples   : "
        f"{len(val_base)}"
    )

    print(
        f"FPS points    : "
        f"{num_points}"
    )

    if (
        len(train_base)
        != EXPECTED_TRAIN_SAMPLES
    ):
        raise RuntimeError(
            "Unexpected number of train samples. "
            f"Expected {EXPECTED_TRAIN_SAMPLES}, "
            f"found {len(train_base)}."
        )

    if (
        len(val_base)
        != EXPECTED_VAL_SAMPLES
    ):
        raise RuntimeError(
            "Unexpected number of validation samples. "
            f"Expected {EXPECTED_VAL_SAMPLES}, "
            f"found {len(val_base)}."
        )

    # -----------------------------------------------------------------
    # Validate one actual sample before doing anything expensive.
    # -----------------------------------------------------------------

    x_check, y_check = extract_xy(
        train_base[0]
    )

    print(
        f"Sample X shape: "
        f"{tuple(x_check.shape)}"
    )

    print(
        f"Sample Y shape: "
        f"{tuple(y_check.shape)}"
    )

    if (
        x_check.shape[0]
        != num_points
    ):
        raise RuntimeError(
            "Unexpected point count in train sample:\n"
            f"Expected {num_points}, "
            f"found {x_check.shape[0]}."
        )

    # =================================================================
    # Fit DGCNN scaler
    # =================================================================

    scaler = fit_dgcnn_scaler(
        train_base
    )

    scaler.save(
        scaler_path
    )

    print(
        f"Scaler saved : "
        f"{scaler_path}"
    )

    # =================================================================
    # Normalized datasets
    # =================================================================

    train_dataset = (
        NormalizedDGCNNDataset(
            train_base,
            scaler,
            point_limit=(
                args.point_limit
            ),
        )
    )

    val_dataset = (
        NormalizedDGCNNDataset(
            val_base,
            scaler,
            point_limit=(
                args.point_limit
            ),
        )
    )

    actual_point_count = (
        args.point_limit
        if args.point_limit
        is not None
        else num_points
    )

    if args.k >= actual_point_count:
        raise ValueError(
            f"k={args.k} must be smaller than "
            f"point count={actual_point_count}."
        )

    # =================================================================
    # DataLoaders
    # =================================================================

    train_loader = build_loader(
        train_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        device=device,
        seed=args.seed,
    )

    val_loader = build_loader(
        val_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        device=device,
        seed=args.seed,
    )

    # =================================================================
    # Model
    # =================================================================

    model = DGCNNRegressor(
        input_dim=4,
        k=args.k,
        knn_chunk_size=(
            args.knn_chunk_size
        ),
    ).to(
        device
    )

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
        f"Points/sample        : "
        f"{actual_point_count:,}"
    )

    # =================================================================
    # Loss / optimizer
    # =================================================================

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=(
            args.weight_decay
        ),
    )

    # =================================================================
    # Training
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

        print()
        print(
            f"[EPOCH "
            f"{epoch}/{args.epochs}]"
        )

        # ---------------------------------------------------------
        # Train
        # ---------------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            log_every=(
                args.log_every
            ),
            max_batches=(
                args.max_train_batches
            ),
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
            max_batches=(
                args.max_val_batches
            ),
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

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model_path,
                model,
                epoch,
                val_loss,
                point_count=(
                    actual_point_count
                ),
            )

            marker = (
                "  *BEST*"
            )

        else:

            epochs_without_improvement += 1

            marker = ""

        print()
        print(
            f"Epoch "
            f"{epoch:3d}/{args.epochs} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"{epoch_time:.1f}s"
            f"{marker}"
        )

        if device.type == "cuda":

            allocated = (
                torch.cuda
                .max_memory_allocated()
                / (1024 ** 3)
            )

            print(
                f"GPU peak allocated: "
                f"{allocated:.2f} GB"
            )

            torch.cuda.reset_peak_memory_stats()

        # ---------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print()
            print(
                "[EARLY STOP] "
                f"No validation improvement "
                f"for {args.patience} epochs."
            )

            break

    training_time = (
        time.perf_counter()
        - training_start
    )

    # =================================================================
    # Reload best checkpoint
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
    # Best-model validation metrics in physical units
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
            max_batches=(
                args.max_val_batches
            ),
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
    print("DGCNN TRAINING COMPLETE")
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
        f"Points/sample    : "
        f"{actual_point_count:,}"
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
            "Train DGCNN on deterministic "
            "FPS-selected CFD point clouds."
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
    # Training
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # DGCNN
    # -----------------------------------------------------------------

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
    )

    parser.add_argument(
        "--knn-chunk-size",
        type=int,
        default=(
            DEFAULT_KNN_CHUNK_SIZE
        ),
    )

    # -----------------------------------------------------------------
    # Output
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

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
    )

    # -----------------------------------------------------------------
    # Debug / smoke-test options
    #
    # These must NOT be used for final training.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--point-limit",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY: use only the first N FPS points "
            "for model execution. "
            "Final training should omit this option."
        ),
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY: stop each training epoch "
            "after this many batches."
        ),
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY: stop validation after "
            "this many batches."
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

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be positive."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay cannot be negative."
        )

    if args.patience <= 0:
        raise ValueError(
            "--patience must be positive."
        )

    if args.k <= 0:
        raise ValueError(
            "--k must be positive."
        )

    if args.knn_chunk_size <= 0:
        raise ValueError(
            "--knn-chunk-size must be positive."
        )

    if args.log_every <= 0:
        raise ValueError(
            "--log-every must be positive."
        )

    if (
        args.point_limit
        is not None
        and args.point_limit <= 0
    ):
        raise ValueError(
            "--point-limit must be positive."
        )

    if (
        args.max_train_batches
        is not None
        and args.max_train_batches <= 0
    ):
        raise ValueError(
            "--max-train-batches must be positive."
        )

    if (
        args.max_val_batches
        is not None
        and args.max_val_batches <= 0
    ):
        raise ValueError(
            "--max-val-batches must be positive."
        )


# =====================================================================
# Entry point
# =====================================================================

def main():

    args = parse_args()

    validate_args(
        args
    )

    train(
        args
    )


if __name__ == "__main__":
    main()