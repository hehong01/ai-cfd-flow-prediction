"""
DGCNN preprocessing using Farthest Point Sampling (FPS).

Each CFD CSV is treated as an independent point cloud.

For every CSV:
    1. Read x, y, z coordinates.
    2. Perform Farthest Point Sampling (FPS).
    3. Select exactly 7000 surface points.
    4. Save the selected 0-based CSV row indices.

The raw CFD CSV files are NOT modified.

Output:
    github/04_cfd_dataset/fps_indices_7000.npz

The saved NPZ contains one index array for each CFD CSV.

Example:
    face_0001_05mps -> 7000 indices
    face_0001_08mps -> 7000 indices
    face_0001_10mps -> 7000 indices
    ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# =====================================================================
# Project paths
# =====================================================================

THIS_DIR = Path(__file__).resolve().parent
GITHUB_ROOT = THIS_DIR.parent

if str(GITHUB_ROOT) not in sys.path:
    sys.path.insert(0, str(GITHUB_ROOT))

from project_paths import DATA_ROOT


CSV_DIR = DATA_ROOT / "05_cfd_csv"

DEFAULT_NUM_POINTS = 7000

EXPECTED_HEADER = [
    "nodenumber",
    "x-coordinate",
    "y-coordinate",
    "z-coordinate",
    "pressure",
    "temperature",
    "y-plus",
    "wall-shear",
    "heat-flux",
    "heat-transfer-coef",
]


# =====================================================================
# CSV loading
# =====================================================================

def check_header(csv_path: Path) -> None:
    """Check that the CSV header matches the expected Fluent export."""

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        first_line = f.readline().strip()

    header = [item.strip() for item in first_line.split(",")]

    if header != EXPECTED_HEADER:
        raise ValueError(
            f"{csv_path.name}: unexpected CSV header.\n"
            f"Expected:\n{EXPECTED_HEADER}\n"
            f"Found:\n{header}"
        )


def load_coordinates(csv_path: Path) -> np.ndarray:
    """
    Load only x, y, z coordinates from one CFD CSV.

    Returns
    -------
    coords : ndarray, shape (N, 3)
        Surface point coordinates in meters.
    """

    check_header(csv_path)

    coords = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3),
        dtype=np.float64,
    )

    if coords.ndim == 1:
        coords = coords.reshape(1, 3)

    if coords.shape[1] != 3:
        raise ValueError(
            f"{csv_path.name}: invalid coordinate shape {coords.shape}"
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError(
            f"{csv_path.name}: coordinates contain NaN or Inf."
        )

    return coords


# =====================================================================
# Farthest Point Sampling
# =====================================================================

def farthest_point_sampling(
    coords: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """
    Deterministic Farthest Point Sampling.

    Parameters
    ----------
    coords:
        Point coordinates, shape (N, 3).

    num_samples:
        Number of points to retain.

    Returns
    -------
    selected:
        0-based row indices into the original CSV,
        shape (num_samples,).

    Method
    ------
    1. Start from the point farthest from the point-cloud centroid.
    2. For every unselected point, track the distance to its nearest
       selected point.
    3. Repeatedly choose the point with the largest nearest distance.

    This generates a spatially distributed subset of the original
    surface point cloud.
    """

    coords = np.asarray(coords, dtype=np.float64)

    n_points = coords.shape[0]

    if num_samples > n_points:
        raise ValueError(
            f"Requested {num_samples} points, "
            f"but point cloud has only {n_points}."
        )

    if num_samples == n_points:
        return np.arange(
            n_points,
            dtype=np.int32,
        )

    selected = np.empty(
        num_samples,
        dtype=np.int32,
    )

    # -------------------------------------------------------------
    # Deterministic starting point:
    # point farthest from the point-cloud centroid
    # -------------------------------------------------------------

    centroid = np.mean(
        coords,
        axis=0,
    )

    delta = coords - centroid

    dist_centroid_sq = np.einsum(
        "ij,ij->i",
        delta,
        delta,
    )

    current = int(
        np.argmax(dist_centroid_sq)
    )

    # Distance from each point to its nearest selected point.
    min_dist_sq = np.full(
        n_points,
        np.inf,
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # FPS loop
    # -------------------------------------------------------------

    for i in range(num_samples):

        selected[i] = current

        delta = coords - coords[current]

        dist_sq = np.einsum(
            "ij,ij->i",
            delta,
            delta,
        )

        np.minimum(
            min_dist_sq,
            dist_sq,
            out=min_dist_sq,
        )

        # Prevent already-selected points from being selected again.
        min_dist_sq[current] = -1.0

        current = int(
            np.argmax(min_dist_sq)
        )

    return selected


# =====================================================================
# Dataset discovery
# =====================================================================

def find_cfd_csv_files() -> list[Path]:
    """
    Return all expected CFD CSV files in face/speed order.

    Expected:
        100 faces
        x 3 velocities
        = 300 CSV files
    """

    files: list[Path] = []

    for face_num in range(1, 101):

        face_id = f"face_{face_num:04d}"

        for speed in ("05mps", "08mps", "10mps"):

            csv_path = CSV_DIR / f"{face_id}_{speed}.csv"

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Missing CFD CSV:\n{csv_path}"
                )

            files.append(csv_path)

    return files


# =====================================================================
# Main preprocessing
# =====================================================================

def preprocess(
    num_points: int,
    output_path: Path,
    overwrite: bool,
) -> None:

    if not CSV_DIR.exists():
        raise FileNotFoundError(
            f"CSV directory does not exist:\n{CSV_DIR}"
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists:\n{output_path}\n\n"
            f"Use --overwrite to replace it."
        )

    csv_files = find_cfd_csv_files()

    print("=" * 78)
    print("DGCNN FPS PREPROCESSING")
    print("=" * 78)
    print(f"CSV directory : {CSV_DIR}")
    print(f"CSV files     : {len(csv_files)}")
    print(f"Points / CSV  : {num_points}")
    print("Sampling      : Farthest Point Sampling")
    print(f"Output        : {output_path}")
    print("=" * 78)

    fps_indices: dict[str, np.ndarray] = {}

    original_counts: list[int] = []

    total_start = time.perf_counter()

    # -------------------------------------------------------------
    # Each CSV is processed independently.
    #
    # No assumption is made that:
    #   - different faces have matching coordinates
    #   - different velocities have matching coordinates
    #   - CSV row order matches between samples
    # -------------------------------------------------------------

    for file_num, csv_path in enumerate(
        csv_files,
        start=1,
    ):

        sample_id = csv_path.stem

        coords = load_coordinates(csv_path)

        n_original = coords.shape[0]

        original_counts.append(n_original)

        if n_original < num_points:
            raise ValueError(
                f"{csv_path.name}: "
                f"only {n_original} points, "
                f"cannot sample {num_points}."
            )

        print(
            f"[{file_num:3d}/{len(csv_files)}] "
            f"{sample_id}: "
            f"{n_original:5d} -> {num_points} ... ",
            end="",
            flush=True,
        )

        start = time.perf_counter()

        indices = farthest_point_sampling(
            coords,
            num_points,
        )

        elapsed = time.perf_counter() - start

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        if indices.shape != (num_points,):
            raise RuntimeError(
                f"{sample_id}: invalid FPS output shape "
                f"{indices.shape}"
            )

        if len(np.unique(indices)) != num_points:
            raise RuntimeError(
                f"{sample_id}: FPS produced duplicate indices."
            )

        if np.min(indices) < 0:
            raise RuntimeError(
                f"{sample_id}: negative row index found."
            )

        if np.max(indices) >= n_original:
            raise RuntimeError(
                f"{sample_id}: FPS index exceeds CSV row count."
            )

        fps_indices[sample_id] = indices.astype(
            np.int32,
            copy=False,
        )

        print(f"OK ({elapsed:.2f} s)")

    # =================================================================
    # Save index masks
    # =================================================================

    np.savez_compressed(
        output_path,
        **fps_indices,
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    counts = np.asarray(
        original_counts,
        dtype=np.int32,
    )

    print()
    print("=" * 78)
    print("PREPROCESSING COMPLETE")
    print("=" * 78)

    print(
        f"CSV samples processed : {len(fps_indices)}"
    )

    print(
        f"Points / sample       : {num_points}"
    )

    print(
        f"Original node range   : "
        f"{counts.min()} ~ {counts.max()}"
    )

    print(
        f"Original mean nodes   : "
        f"{counts.mean():.2f}"
    )

    print(
        f"Output                : {output_path}"
    )

    print(
        f"Total runtime         : "
        f"{total_elapsed:.1f} s"
    )

    print("=" * 78)

    print()
    print("Raw CFD CSV files were NOT modified.")
    print(
        "The NPZ file contains one independent "
        f"{num_points}-row index array for each CFD CSV."
    )


# =====================================================================
# CLI
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate FPS row-index masks "
            "for DGCNN CFD point-cloud training."
        )
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help=(
            "Number of points retained from each CSV "
            f"(default: {DEFAULT_NUM_POINTS})."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output NPZ path. "
            "Default: 04_cfd_dataset/fps_indices_<N>.npz"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing FPS index file.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.num_points <= 0:
        raise ValueError(
            "--num-points must be positive."
        )

    # -------------------------------------------------------------
    # Keep generated FPS indices inside 04_cfd_dataset.
    # Do NOT create or modify the ai-cfd-data directory structure.
    # -------------------------------------------------------------

    if args.output is None:

        output_path = (
            THIS_DIR
            / f"fps_indices_{args.num_points}.npz"
        )

    else:

        output_path = args.output

    preprocess(
        num_points=args.num_points,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()