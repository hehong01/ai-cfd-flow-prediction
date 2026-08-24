"""
AI-CFD prediction preprocessing.

Usage
-----
From:
    C:/ai-cfd-flow-prediction/github/06_ai_prediction

Run:
    python preprocess.py --model mlp
    python preprocess.py --model dgcnn
    python preprocess.py --model both
    python preprocess.py --model both --input test_face.jpg

Data flow
---------
07_predictions/input_image/<name>.jpg
    -> existing 01_image_to_stl/image_to_stl.py
    -> 07_predictions/stl/<name>.stl
    -> common surface XYZ cloud [m]
        -> MLP:   mlp/input_csv/<name>.csv
        -> DGCNN: FPS 7000 -> dgcnn/input_csv/<name>.csv

The image -> STL algorithm is not reimplemented or changed here.
This script calls the already validated
01_image_to_stl/image_to_stl.py::convert_image_to_stl()
and adds only prediction-specific file management and STL -> CSV preprocessing.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


THIS_DIR = Path(__file__).resolve().parent
GITHUB_ROOT = THIS_DIR.parent

if str(GITHUB_ROOT) not in sys.path:
    sys.path.insert(0, str(GITHUB_ROOT))

from project_paths import DATA_ROOT


PREDICTION_ROOT = DATA_ROOT / "07_predictions"

INPUT_IMAGE_DIR = PREDICTION_ROOT / "input_image"
STL_DIR = PREDICTION_ROOT / "stl"

MLP_INPUT_CSV_DIR = PREDICTION_ROOT / "mlp" / "input_csv"
DGCNN_INPUT_CSV_DIR = PREDICTION_ROOT / "dgcnn" / "input_csv"

IMAGE_TO_STL_SCRIPT = GITHUB_ROOT / "01_image_to_stl" / "image_to_stl.py"
DGCNN_FPS_SCRIPT = GITHUB_ROOT / "04_cfd_dataset" / "preprocessing_dgcnn.py"

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

DGCNN_NUM_POINTS = 7000
DEFAULT_SURFACE_POINTS = 10000
DEFAULT_SEED = 42

MM_TO_M = 1.0e-3
CSV_HEADER = "x,y,z"


def ensure_directories() -> None:
    for directory in (
        INPUT_IMAGE_DIR,
        STL_DIR,
        MLP_INPUT_CSV_DIR,
        DGCNN_INPUT_CSV_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def is_newer(source: Path, output: Path) -> bool:
    if not output.exists():
        return True
    return source.stat().st_mtime_ns > output.stat().st_mtime_ns


def collect_images(input_value: str | None) -> list[Path]:
    if input_value:
        image_path = Path(input_value)
        if not image_path.is_absolute():
            image_path = INPUT_IMAGE_DIR / image_path
        image_path = image_path.resolve()

        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found:\n{image_path}")

        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {image_path.suffix}")

        return [image_path]

    image_paths = sorted(
        path
        for path in INPUT_IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )

    seen: dict[str, Path] = {}
    for image_path in image_paths:
        key = image_path.stem.lower()
        if key in seen:
            raise ValueError(
                "Two input images have the same filename stem:\n"
                f"{seen[key]}\n{image_path}"
            )
        seen[key] = image_path

    return image_paths


def candidate_python_executables() -> list[Path]:
    candidates: list[Path] = []

    current = Path(sys.executable)
    candidates.append(current)

    # Base conda layout:
    # C:/Users/.../miniconda3/python.exe
    conda_root = current.parent
    ai_cfd_python = conda_root / "envs" / "ai-cfd" / "python.exe"
    if ai_cfd_python.is_file():
        candidates.append(ai_cfd_python.resolve())

    # Running from a conda env:
    # C:/Users/.../miniconda3/envs/<env>/python.exe
    if current.parent.parent.name.lower() == "envs":
        conda_root = current.parent.parent.parent
        ai_cfd_python = conda_root / "envs" / "ai-cfd" / "python.exe"
        if ai_cfd_python.is_file():
            candidates.append(ai_cfd_python.resolve())

    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)

    return unique


def image_to_stl_environment_ok(python_exe: Path) -> bool:
    check_code = (
        "import cv2, numpy, trimesh, mediapipe as mp; "
        "assert hasattr(mp, 'solutions'); "
        "assert hasattr(mp.solutions, 'face_mesh')"
    )

    result = subprocess.run(
        [str(python_exe), "-c", check_code],
        check=False,
    )

    return result.returncode == 0


def find_image_to_stl_python() -> Path:
    checked: list[Path] = []

    for python_exe in candidate_python_executables():
        checked.append(python_exe)
        if image_to_stl_environment_ok(python_exe):
            return python_exe

    checked_text = "\n".join(f"  - {path}" for path in checked)

    raise RuntimeError(
        "No available Python environment can run the existing "
        "01_image_to_stl/image_to_stl.py.\n"
        "Checked:\n"
        f"{checked_text}"
    )


def generate_stl_with_existing_code(
    image_path: Path,
    output_path: Path,
) -> None:
    if not IMAGE_TO_STL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Existing image->STL script not found:\n{IMAGE_TO_STL_SCRIPT}"
        )

    python_exe = find_image_to_stl_python()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_output = output_path.with_name(
        f"{output_path.stem}.tmp.stl"
    )

    if temp_output.exists():
        temp_output.unlink()

    runner = f"""
import importlib.util
from pathlib import Path

script_path = Path(r"{IMAGE_TO_STL_SCRIPT}")
image_path = Path(r"{image_path}")
output_path = Path(r"{temp_output}")

spec = importlib.util.spec_from_file_location(
    "_existing_image_to_stl",
    script_path,
)

if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {{script_path}}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.convert_image_to_stl(
    image_path=image_path,
    output_path=output_path,
)
"""

    print(f"[STL] Existing code: {IMAGE_TO_STL_SCRIPT}")
    print(f"[STL] Python       : {python_exe}")

    try:
        subprocess.run(
            [str(python_exe), "-c", runner],
            check=True,
        )

        if not temp_output.is_file():
            raise RuntimeError(
                "Image->STL code completed without creating:\n"
                f"{temp_output}"
            )

        temp_output.replace(output_path)

    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise


def load_stl_mesh(stl_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(
        str(stl_path),
        force="mesh",
        process=True,
    )

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]

        if not geometries:
            raise ValueError(f"No triangle mesh found in:\n{stl_path}")

        mesh = trimesh.util.concatenate(geometries)

    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    else:
        raise TypeError(
            f"Unsupported STL object: {type(loaded).__name__}"
        )

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"STL contains no usable geometry:\n{stl_path}")

    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(f"STL vertices contain NaN/Inf:\n{stl_path}")

    return mesh


def sample_surface_points(
    mesh: trimesh.Trimesh,
    num_points: int,
    seed: int,
) -> np.ndarray:
    vertices_m = (
        np.asarray(mesh.vertices, dtype=np.float64)
        * MM_TO_M
    )

    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices_m[faces]

    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]

    areas = 0.5 * np.linalg.norm(
        np.cross(edge_1, edge_2),
        axis=1,
    )

    valid = np.isfinite(areas) & (areas > 0.0)

    if not np.any(valid):
        raise ValueError("STL contains no non-degenerate triangle area.")

    triangles = triangles[valid]
    areas = areas[valid]
    probabilities = areas / areas.sum()

    rng = np.random.default_rng(seed)

    triangle_indices = rng.choice(
        len(triangles),
        size=num_points,
        replace=True,
        p=probabilities,
    )

    selected = triangles[triangle_indices]

    r1 = np.sqrt(rng.random(num_points))
    r2 = rng.random(num_points)

    points = (
        (1.0 - r1)[:, None] * selected[:, 0]
        + (r1 * (1.0 - r2))[:, None] * selected[:, 1]
        + (r1 * r2)[:, None] * selected[:, 2]
    )

    if points.shape != (num_points, 3):
        raise RuntimeError(
            f"Unexpected surface point shape: {points.shape}"
        )

    if not np.all(np.isfinite(points)):
        raise ValueError("Sampled XYZ contains NaN or Inf.")

    return points


def load_existing_fps_function():
    if not DGCNN_FPS_SCRIPT.is_file():
        raise FileNotFoundError(
            f"DGCNN preprocessing script not found:\n{DGCNN_FPS_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "_existing_dgcnn_fps",
        DGCNN_FPS_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load:\n{DGCNN_FPS_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "farthest_point_sampling"):
        raise AttributeError(
            "Existing DGCNN preprocessing script does not contain "
            "farthest_point_sampling()."
        )

    return module.farthest_point_sampling


def save_xyz_csv(
    csv_path: Path,
    xyz: np.ndarray,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = csv_path.with_name(
        f"{csv_path.stem}.tmp.csv"
    )

    if temp_path.exists():
        temp_path.unlink()

    try:
        np.savetxt(
            temp_path,
            np.asarray(xyz, dtype=np.float64),
            delimiter=",",
            header=CSV_HEADER,
            comments="",
            fmt="%.10e",
        )

        temp_path.replace(csv_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_xyz_csv(csv_path: Path) -> np.ndarray:
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:
        header = file.readline().strip()

    if header != CSV_HEADER:
        raise ValueError(
            f"Unexpected header '{header}', expected '{CSV_HEADER}'."
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
        raise ValueError(f"Invalid XYZ shape: {xyz.shape}")

    if not np.all(np.isfinite(xyz)):
        raise ValueError("CSV contains NaN or Inf.")

    return xyz


def csv_needs_rebuild(
    csv_path: Path,
    stl_path: Path,
    expected_rows: int,
    overwrite: bool,
) -> tuple[bool, str]:
    if overwrite:
        return True, "--overwrite"

    if not csv_path.exists():
        return True, "missing"

    if is_newer(stl_path, csv_path):
        return True, "STL is newer"

    try:
        xyz = load_xyz_csv(csv_path)
    except Exception as exc:
        return True, f"invalid existing CSV: {exc}"

    if len(xyz) != expected_rows:
        return True, f"{len(xyz)} rows, expected {expected_rows}"

    return False, "up to date"


def print_xyz_summary(
    label: str,
    xyz: np.ndarray,
) -> None:
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    extent = xyz_max - xyz_min

    print(f"{label}: {len(xyz):,} points")
    print(
        "  min [m] : "
        f"x={xyz_min[0]: .6f}, "
        f"y={xyz_min[1]: .6f}, "
        f"z={xyz_min[2]: .6f}"
    )
    print(
        "  max [m] : "
        f"x={xyz_max[0]: .6f}, "
        f"y={xyz_max[1]: .6f}, "
        f"z={xyz_max[2]: .6f}"
    )
    print(
        "  size[m] : "
        f"x={extent[0]: .6f}, "
        f"y={extent[1]: .6f}, "
        f"z={extent[2]: .6f}"
    )


def process_one_image(
    image_path: Path,
    model: str,
    surface_points: int,
    seed: int,
    overwrite: bool,
) -> None:
    stem = image_path.stem

    stl_path = STL_DIR / f"{stem}.stl"
    mlp_csv_path = MLP_INPUT_CSV_DIR / f"{stem}.csv"
    dgcnn_csv_path = DGCNN_INPUT_CSV_DIR / f"{stem}.csv"

    print()
    print("=" * 78)
    print(f"IMAGE: {image_path.name}")
    print("=" * 78)

    rebuild_stl = overwrite or is_newer(image_path, stl_path)

    if rebuild_stl:
        if overwrite:
            reason = "--overwrite"
        elif not stl_path.exists():
            reason = "missing"
        else:
            reason = "image is newer"

        print(f"[STL] GENERATE ({reason})")

        generate_stl_with_existing_code(
            image_path=image_path,
            output_path=stl_path,
        )

        print(f"[STL] SAVED: {stl_path}")

    else:
        print("[STL] SKIP (up to date)")

    rebuild_mlp = False
    rebuild_dgcnn = False

    if model in {"mlp", "both"}:
        rebuild_mlp, reason = csv_needs_rebuild(
            csv_path=mlp_csv_path,
            stl_path=stl_path,
            expected_rows=surface_points,
            overwrite=overwrite,
        )

        if rebuild_mlp:
            print(f"[MLP CSV] GENERATE ({reason})")
        else:
            print("[MLP CSV] SKIP (up to date)")

    if model in {"dgcnn", "both"}:
        rebuild_dgcnn, reason = csv_needs_rebuild(
            csv_path=dgcnn_csv_path,
            stl_path=stl_path,
            expected_rows=DGCNN_NUM_POINTS,
            overwrite=overwrite,
        )

        if rebuild_dgcnn:
            print(f"[DGCNN CSV] GENERATE ({reason})")
        else:
            print("[DGCNN CSV] SKIP (up to date)")

    if not rebuild_mlp and not rebuild_dgcnn:
        return

    mesh = load_stl_mesh(stl_path)

    print(
        f"[SURFACE] {surface_points:,} points "
        f"(seed={seed})"
    )

    surface_xyz = sample_surface_points(
        mesh=mesh,
        num_points=surface_points,
        seed=seed,
    )

    print_xyz_summary(
        "Common surface cloud",
        surface_xyz,
    )

    if rebuild_mlp:
        save_xyz_csv(
            mlp_csv_path,
            surface_xyz,
        )
        print(f"[MLP CSV] SAVED: {mlp_csv_path}")

    if rebuild_dgcnn:
        if surface_points < DGCNN_NUM_POINTS:
            raise ValueError(
                f"DGCNN requires at least {DGCNN_NUM_POINTS} "
                f"surface points."
            )

        fps = load_existing_fps_function()

        print(
            f"[DGCNN FPS] "
            f"{surface_points:,} -> {DGCNN_NUM_POINTS:,}"
        )

        selected_indices = fps(
            surface_xyz,
            DGCNN_NUM_POINTS,
        )

        dgcnn_xyz = surface_xyz[selected_indices]

        if dgcnn_xyz.shape != (DGCNN_NUM_POINTS, 3):
            raise RuntimeError(
                f"Unexpected DGCNN shape: {dgcnn_xyz.shape}"
            )

        print_xyz_summary(
            "DGCNN FPS cloud",
            dgcnn_xyz,
        )

        save_xyz_csv(
            dgcnn_csv_path,
            dgcnn_xyz,
        )

        print(f"[DGCNN CSV] SAVED: {dgcnn_csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "New image -> existing validated STL reconstruction "
            "-> MLP/DGCNN prediction input CSV."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=("mlp", "dgcnn", "both"),
        help="Generate MLP CSV, DGCNN CSV, or both.",
    )

    parser.add_argument(
        "--input",
        help=(
            "Optional single image. A bare filename is resolved in "
            "07_predictions/input_image. If omitted, all images are scanned."
        ),
    )

    parser.add_argument(
        "--surface-points",
        type=int,
        default=DEFAULT_SURFACE_POINTS,
        help=(
            "Number of common STL surface points before model-specific "
            f"processing (default: {DEFAULT_SURFACE_POINTS})."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Surface sampling seed (default: {DEFAULT_SEED}).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate requested outputs even if current files exist.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.surface_points <= 0:
        raise ValueError("--surface-points must be positive.")

    if (
        args.model in {"dgcnn", "both"}
        and args.surface_points < DGCNN_NUM_POINTS
    ):
        raise ValueError(
            f"DGCNN requires --surface-points >= {DGCNN_NUM_POINTS}."
        )

    ensure_directories()

    image_paths = collect_images(args.input)

    print("=" * 78)
    print("AI-CFD PREDICTION PREPROCESSING")
    print("=" * 78)
    print(f"Input folder      : {INPUT_IMAGE_DIR}")
    print(f"STL folder        : {STL_DIR}")
    print(f"Model             : {args.model}")
    print(f"Surface points    : {args.surface_points:,}")
    print(f"DGCNN points      : {DGCNN_NUM_POINTS:,}")
    print(f"Images discovered : {len(image_paths)}")
    print("=" * 78)

    if not image_paths:
        print("No JPG/JPEG/PNG images found.")
        return 0

    failures: list[tuple[str, str]] = []

    for image_path in image_paths:
        try:
            process_one_image(
                image_path=image_path,
                model=args.model,
                surface_points=args.surface_points,
                seed=args.seed,
                overwrite=args.overwrite,
            )

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((image_path.name, message))
            print(f"[FAILED] {image_path.name}: {message}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Images : {len(image_paths)}")
    print(f"Failed : {len(failures)}")

    if failures:
        for name, message in failures:
            print(f"  - {name}: {message}")
        return 1

    print("Preprocessing completed successfully.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
