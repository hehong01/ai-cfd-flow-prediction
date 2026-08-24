"""
DGCNN CFD dataset loader.

Each CFD CSV represents one surface point-cloud sample.

Input per point:
    [x, y, z, velocity]

Target per point:
    [HTC, wall_shear, pressure]

FPS preprocessing:
    preprocessing_dgcnn.py generates one fixed 7000-point row-index
    mask for each CFD CSV.

Dataset split:
    face_0001 ~ face_0080 -> train
    face_0081 ~ face_0090 -> val
    face_0091 ~ face_0100 -> test

Expected sample counts:
    train : 80 faces x 3 velocities = 240
    val   : 10 faces x 3 velocities = 30
    test  : 10 faces x 3 velocities = 30

Running this file directly performs a full validation of all 300 samples.

Importing this file from model-training code does NOT run the validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# =====================================================================
# Project paths
# =====================================================================

THIS_DIR = Path(__file__).resolve().parent
GITHUB_ROOT = THIS_DIR.parent

if str(GITHUB_ROOT) not in sys.path:
    sys.path.insert(0, str(GITHUB_ROOT))

from project_paths import DATA_ROOT


CSV_DIR = DATA_ROOT / "05_cfd_csv"

FPS_INDEX_PATH = THIS_DIR / "fps_indices_7000.npz"

NUM_POINTS = 7000

SPEEDS = (
    ("05mps", 5.0),
    ("08mps", 8.0),
    ("10mps", 10.0),
)


# =====================================================================
# Dataset split
# =====================================================================

def get_split(face_num: int) -> str:
    """
    Return train / val / test split from face number.

    0001 ~ 0080 -> train
    0081 ~ 0090 -> val
    0091 ~ 0100 -> test
    """

    if 1 <= face_num <= 80:
        return "train"

    if 81 <= face_num <= 90:
        return "val"

    if 91 <= face_num <= 100:
        return "test"

    raise ValueError(
        f"Invalid face number: {face_num}"
    )


# =====================================================================
# Sample list
# =====================================================================

def build_sample_list(split: str) -> list[dict]:
    """
    Build the list of CFD samples belonging to one split.

    One sample:
        one face
        at one inlet velocity

    Example:
        face_0001_05mps
    """

    if split not in {
        "train",
        "val",
        "test",
    }:
        raise ValueError(
            f"Invalid split: {split}"
        )

    samples: list[dict] = []

    for face_num in range(1, 101):

        if get_split(face_num) != split:
            continue

        face_id = f"face_{face_num:04d}"

        for speed_name, velocity in SPEEDS:

            sample_id = (
                f"{face_id}_{speed_name}"
            )

            csv_path = (
                CSV_DIR
                / f"{sample_id}.csv"
            )

            samples.append(
                {
                    "sample_id": sample_id,
                    "face_id": face_id,
                    "face_num": face_num,
                    "speed_name": speed_name,
                    "velocity": velocity,
                    "csv_path": csv_path,
                }
            )

    return samples


# =====================================================================
# DGCNN Dataset
# =====================================================================

class DGCNNCFDDataset(Dataset):
    """
    PyTorch Dataset for DGCNN CFD field prediction.

    Each item returns:

        X : torch.float32, shape (7000, 4)

            columns:
                0 -> x
                1 -> y
                2 -> z
                3 -> inlet velocity

        Y : torch.float32, shape (7000, 3)

            columns:
                0 -> HTC
                1 -> wall shear
                2 -> pressure

    If return_sample_id=True:

        return X, Y, sample_id

    Otherwise:

        return X, Y
    """

    def __init__(
        self,
        split: str,
        return_sample_id: bool = False,
    ):

        self.split = split
        self.return_sample_id = return_sample_id

        # -------------------------------------------------------------
        # Required data
        # -------------------------------------------------------------

        if not CSV_DIR.exists():
            raise FileNotFoundError(
                f"CFD CSV directory not found:\n"
                f"{CSV_DIR}"
            )

        if not FPS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FPS index file not found:\n"
                f"{FPS_INDEX_PATH}\n\n"
                f"Run preprocessing_dgcnn.py first."
            )

        # -------------------------------------------------------------
        # Build split sample list
        # -------------------------------------------------------------

        self.samples = build_sample_list(
            split
        )

        # -------------------------------------------------------------
        # Load FPS masks once
        #
        # Expected keys:
        #
        # face_0001_05mps
        # face_0001_08mps
        # ...
        # face_0100_10mps
        # -------------------------------------------------------------

        with np.load(FPS_INDEX_PATH) as fps_npz:

            self.fps_indices = {
                key: fps_npz[key].astype(
                    np.int64,
                    copy=False,
                )
                for key in fps_npz.files
            }

        # -------------------------------------------------------------
        # Validate required files and masks
        # -------------------------------------------------------------

        for sample in self.samples:

            sample_id = sample["sample_id"]
            csv_path = sample["csv_path"]

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Missing CFD CSV:\n"
                    f"{csv_path}"
                )

            if sample_id not in self.fps_indices:
                raise KeyError(
                    f"FPS indices missing for "
                    f"{sample_id}"
                )

            indices = self.fps_indices[
                sample_id
            ]

            if indices.shape != (
                NUM_POINTS,
            ):
                raise ValueError(
                    f"{sample_id}: expected "
                    f"{NUM_POINTS} FPS indices, "
                    f"found shape {indices.shape}"
                )

            if len(np.unique(indices)) != NUM_POINTS:
                raise ValueError(
                    f"{sample_id}: FPS indices "
                    f"contain duplicates."
                )

            if np.min(indices) < 0:
                raise ValueError(
                    f"{sample_id}: negative "
                    f"FPS index found."
                )

    # -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    # -----------------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ):

        sample = self.samples[index]

        sample_id = sample["sample_id"]
        csv_path = sample["csv_path"]
        velocity = sample["velocity"]

        # -------------------------------------------------------------
        # Read only the CFD columns actually required by the model.
        #
        # Original Fluent CSV:
        #
        # 0  nodenumber
        # 1  x-coordinate
        # 2  y-coordinate
        # 3  z-coordinate
        # 4  pressure
        # 5  temperature
        # 6  y-plus
        # 7  wall-shear
        # 8  heat-flux
        # 9  heat-transfer-coef
        #
        # Loaded array:
        #
        # 0 -> x
        # 1 -> y
        # 2 -> z
        # 3 -> pressure
        # 4 -> wall shear
        # 5 -> HTC
        # -------------------------------------------------------------

        data = np.loadtxt(
            csv_path,
            delimiter=",",
            skiprows=1,
            usecols=(1, 2, 3, 4, 7, 9),
            dtype=np.float64,
        )

        if data.ndim == 1:
            data = data.reshape(
                1,
                6,
            )

        if data.shape[1] != 6:
            raise ValueError(
                f"{sample_id}: invalid loaded "
                f"data shape {data.shape}"
            )

        # -------------------------------------------------------------
        # Apply the FPS row-index mask belonging specifically
        # to this CSV.
        # -------------------------------------------------------------

        indices = self.fps_indices[
            sample_id
        ]

        max_index = int(
            np.max(indices)
        )

        if max_index >= len(data):
            raise IndexError(
                f"{sample_id}: FPS index "
                f"{max_index} exceeds CSV "
                f"row count {len(data)}"
            )

        sampled = data[
            indices
        ]

        # -------------------------------------------------------------
        # Input
        #
        # X = [x, y, z, velocity]
        #
        # shape:
        #     (7000, 4)
        # -------------------------------------------------------------

        coords = sampled[
            :,
            0:3,
        ]

        velocity_column = np.full(
            (
                NUM_POINTS,
                1,
            ),
            velocity,
            dtype=np.float64,
        )

        x = np.concatenate(
            (
                coords,
                velocity_column,
            ),
            axis=1,
        )

        # -------------------------------------------------------------
        # Target
        #
        # Loaded columns:
        #
        # sampled[:, 3] = pressure
        # sampled[:, 4] = wall shear
        # sampled[:, 5] = HTC
        #
        # Desired target order:
        #
        # Y = [HTC, wall shear, pressure]
        #
        # shape:
        #     (7000, 3)
        # -------------------------------------------------------------

        htc = sampled[
            :,
            5:6,
        ]

        wall_shear = sampled[
            :,
            4:5,
        ]

        pressure = sampled[
            :,
            3:4,
        ]

        y = np.concatenate(
            (
                htc,
                wall_shear,
                pressure,
            ),
            axis=1,
        )

        # -------------------------------------------------------------
        # Final sample validation
        # -------------------------------------------------------------

        if x.shape != (
            NUM_POINTS,
            4,
        ):
            raise RuntimeError(
                f"{sample_id}: invalid "
                f"X shape {x.shape}"
            )

        if y.shape != (
            NUM_POINTS,
            3,
        ):
            raise RuntimeError(
                f"{sample_id}: invalid "
                f"Y shape {y.shape}"
            )

        if not np.all(
            np.isfinite(x)
        ):
            raise ValueError(
                f"{sample_id}: X contains "
                f"NaN or Inf."
            )

        if not np.all(
            np.isfinite(y)
        ):
            raise ValueError(
                f"{sample_id}: Y contains "
                f"NaN or Inf."
            )

        # -------------------------------------------------------------
        # Convert to PyTorch float32 tensors.
        # -------------------------------------------------------------

        x = torch.from_numpy(
            x.astype(
                np.float32,
                copy=False,
            )
        )

        y = torch.from_numpy(
            y.astype(
                np.float32,
                copy=False,
            )
        )

        if self.return_sample_id:
            return (
                x,
                y,
                sample_id,
            )

        return x, y


# =====================================================================
# Full standalone validation
# =====================================================================

def main():

    print("=" * 72)
    print("DGCNN DATASET FULL TEST")
    print("=" * 72)

    total_checked = 0

    expected_counts = {
        "train": 240,
        "val": 30,
        "test": 30,
    }

    for split in (
        "train",
        "val",
        "test",
    ):

        dataset = DGCNNCFDDataset(
            split=split,
            return_sample_id=True,
        )

        expected = expected_counts[
            split
        ]

        if len(dataset) != expected:
            raise RuntimeError(
                f"{split}: expected "
                f"{expected} samples, "
                f"found {len(dataset)}"
            )

        print()
        print(
            f"[{split.upper()}]"
        )

        print(
            f"samples : {len(dataset)}"
        )

        # -------------------------------------------------------------
        # Actually load every sample.
        # -------------------------------------------------------------

        for i in range(
            len(dataset)
        ):

            x, y, sample_id = (
                dataset[i]
            )

            # ---------------------------------------------------------
            # Shape
            # ---------------------------------------------------------

            if tuple(x.shape) != (
                NUM_POINTS,
                4,
            ):
                raise RuntimeError(
                    f"{sample_id}: "
                    f"invalid X shape "
                    f"{tuple(x.shape)}"
                )

            if tuple(y.shape) != (
                NUM_POINTS,
                3,
            ):
                raise RuntimeError(
                    f"{sample_id}: "
                    f"invalid Y shape "
                    f"{tuple(y.shape)}"
                )

            # ---------------------------------------------------------
            # dtype
            # ---------------------------------------------------------

            if x.dtype != torch.float32:
                raise RuntimeError(
                    f"{sample_id}: "
                    f"invalid X dtype "
                    f"{x.dtype}"
                )

            if y.dtype != torch.float32:
                raise RuntimeError(
                    f"{sample_id}: "
                    f"invalid Y dtype "
                    f"{y.dtype}"
                )

            # ---------------------------------------------------------
            # NaN / Inf
            # ---------------------------------------------------------

            if not bool(
                torch.isfinite(
                    x
                ).all()
            ):
                raise RuntimeError(
                    f"{sample_id}: "
                    f"X contains NaN or Inf."
                )

            if not bool(
                torch.isfinite(
                    y
                ).all()
            ):
                raise RuntimeError(
                    f"{sample_id}: "
                    f"Y contains NaN or Inf."
                )

            total_checked += 1

            # ---------------------------------------------------------
            # Progress output
            # ---------------------------------------------------------

            if (
                (i + 1) % 25 == 0
                or
                (i + 1) == len(dataset)
            ):
                print(
                    f"  "
                    f"{i + 1:3d}/"
                    f"{len(dataset)}  "
                    f"{sample_id} OK"
                )

        print(
            f"{split.upper()} PASSED: "
            f"{len(dataset)} samples"
        )

    # =================================================================
    # Final result
    # =================================================================

    if total_checked != 300:
        raise RuntimeError(
            f"Expected 300 total samples, "
            f"checked {total_checked}."
        )

    print()
    print("=" * 72)
    print("FULL DATASET TEST PASSED")
    print("=" * 72)

    print(
        f"Total samples checked : "
        f"{total_checked}"
    )

    print(
        f"Points / sample       : "
        f"{NUM_POINTS}"
    )

    print(
        "Input                : "
        "[x, y, z, velocity]"
    )

    print(
        "Input shape          : "
        "(7000, 4)"
    )

    print(
        "Target               : "
        "[HTC, wall_shear, pressure]"
    )

    print(
        "Target shape         : "
        "(7000, 3)"
    )

    print(
        "Tensor dtype         : "
        "torch.float32"
    )

    print("=" * 72)


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    main()