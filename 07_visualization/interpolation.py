"""
Smooth surface visualization for AI-CFD prediction results.

This script maps scattered MLP/DGCNN prediction points back onto the
reconstructed STL surface using inverse-distance-weighted (IDW)
interpolation, then renders the result as a smooth Plotly Mesh3d surface.

Input prediction CSV:
    x,y,z,velocity,predicted_htc,predicted_wall_shear

Expected STL:
    ai-cfd-data/07_predictions/stl/<geometry>.stl

Examples
--------
From github/07_visualization:

    python interpolation.py --model mlp --input test_face_vel8.csv

    python interpolation.py --model dgcnn --input test_face_vel8.csv

Optional interpolation settings:

    python interpolation.py --model mlp --input test_face_vel8.csv \
        --neighbors 8 --power 2

If automatic STL matching fails:

    python interpolation.py --model mlp --input test_face_vel8.csv \
        --stl test_face.stl
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# =============================================================================
# Project paths
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
GITHUB_ROOT = THIS_DIR.parent

if str(GITHUB_ROOT) not in sys.path:
    sys.path.insert(0, str(GITHUB_ROOT))

from project_paths import DATA_ROOT


PREDICTION_ROOT = DATA_ROOT / "07_predictions"
STL_DIR = PREDICTION_ROOT / "stl"

RESULT_ROOT = (
    DATA_ROOT
    / "08_results"
    / "figures"
    / "interpolation"
)


# =============================================================================
# Plot settings
# =============================================================================

COLOR_SCALE = "Turbo"

# Smooth triangle shading.
FLAT_SHADING = False


# =============================================================================
# Path helpers
# =============================================================================

def resolve_prediction_csv(
    model: str,
    input_value: str,
) -> Path:
    path = Path(input_value)

    if path.is_file():
        return path.resolve()

    if path.is_absolute():
        raise FileNotFoundError(
            f"Prediction CSV not found:\n{path}"
        )

    candidate = (
        PREDICTION_ROOT
        / model
        / "prediction_csv"
        / path
    ).resolve()

    if not candidate.is_file():
        raise FileNotFoundError(
            f"Prediction CSV not found:\n{candidate}"
        )

    return candidate


def prediction_stem_to_geometry_stem(
    prediction_stem: str,
) -> str:
    """
    Convert:
        test_face_vel8
        test_face_vel8p5

    to:
        test_face

    Prediction filenames are produced by 06_ai_prediction as:
        <geometry>_vel<speed>.csv
    """
    geometry_stem = re.sub(
        r"_vel[^_]+$",
        "",
        prediction_stem,
    )

    if not geometry_stem:
        raise ValueError(
            f"Could not infer geometry stem from '{prediction_stem}'."
        )

    return geometry_stem


def resolve_stl_path(
    prediction_csv: Path,
    stl_value: str | None,
) -> Path:
    if stl_value is not None:
        path = Path(stl_value)

        if path.is_file():
            return path.resolve()

        if not path.is_absolute():
            candidate = (
                STL_DIR
                / path
            ).resolve()

            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"STL file not found:\n{path}"
        )

    geometry_stem = (
        prediction_stem_to_geometry_stem(
            prediction_csv.stem
        )
    )

    candidate = (
        STL_DIR
        / f"{geometry_stem}.stl"
    ).resolve()

    if not candidate.is_file():
        raise FileNotFoundError(
            "Could not automatically find the matching STL.\n"
            f"Expected:\n{candidate}\n\n"
            "Use --stl <filename-or-path> to specify it manually."
        )

    return candidate


def build_output_dir(
    model: str,
    prediction_csv: Path,
) -> Path:
    output_dir = (
        RESULT_ROOT
        / model
        / prediction_csv.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# =============================================================================
# Prediction CSV
# =============================================================================

def load_prediction_csv(
    csv_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required = (
        "x",
        "y",
        "z",
        "velocity",
        "predicted_htc",
        "predicted_wall_shear",
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Prediction CSV is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(df.columns)
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    values = df[
        list(required)
    ].to_numpy(dtype=np.float64)

    if len(df) == 0:
        raise ValueError(
            "Prediction CSV contains no rows."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Prediction CSV contains NaN or Inf."
        )

    velocity = df[
        "velocity"
    ].to_numpy(dtype=np.float64)

    if not np.allclose(
        velocity,
        velocity[0],
    ):
        raise ValueError(
            "Prediction CSV must contain one constant "
            "inlet velocity for the whole geometry."
        )

    return df


# =============================================================================
# STL loading
# =============================================================================

def load_stl_mesh(
    stl_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load STL using trimesh.

    The image-to-STL stage stores geometry in millimeters.
    Prediction coordinates are in meters, so STL vertices are converted:

        mm -> m
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "trimesh is required for interpolation visualization.\n"
            "Install with:\n"
            "    python -m pip install trimesh"
        ) from exc

    mesh = trimesh.load(
        stl_path,
        force="mesh",
        process=False,
    )

    if mesh.is_empty:
        raise ValueError(
            f"STL mesh is empty:\n{stl_path}"
        )

    vertices_mm = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    faces = np.asarray(
        mesh.faces,
        dtype=np.int64,
    )

    if (
        vertices_mm.ndim != 2
        or vertices_mm.shape[1] != 3
    ):
        raise ValueError(
            f"Unexpected STL vertex shape: {vertices_mm.shape}"
        )

    if (
        faces.ndim != 2
        or faces.shape[1] != 3
    ):
        raise ValueError(
            f"Unexpected STL face shape: {faces.shape}"
        )

    if len(vertices_mm) == 0 or len(faces) == 0:
        raise ValueError(
            "STL contains no usable vertices/faces."
        )

    vertices_m = (
        vertices_mm
        * 1.0e-3
    )

    if not np.isfinite(
        vertices_m
    ).all():
        raise ValueError(
            "STL vertices contain NaN or Inf."
        )

    return vertices_m, faces


# =============================================================================
# IDW interpolation
# =============================================================================

def interpolate_idw(
    source_xyz: np.ndarray,
    source_values: np.ndarray,
    target_xyz: np.ndarray,
    neighbors: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate scattered source values onto target STL vertices.

    For each target vertex:
        1. find k nearest prediction points
        2. calculate inverse-distance weights
        3. weighted-average the prediction values

    Exact coordinate matches receive the exact source value.

    Returns
    -------
    interpolated_values:
        (M,)

    nearest_distance:
        nearest source-point distance for each target vertex, in meters
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "scipy is required for IDW interpolation.\n"
            "Install with:\n"
            "    python -m pip install scipy"
        ) from exc

    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError(
            f"source_xyz must have shape (N,3), found {source_xyz.shape}."
        )

    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError(
            f"target_xyz must have shape (M,3), found {target_xyz.shape}."
        )

    if source_values.shape != (
        len(source_xyz),
    ):
        raise ValueError(
            "source_values length does not match source_xyz."
        )

    if neighbors <= 0:
        raise ValueError(
            "--neighbors must be positive."
        )

    if power <= 0.0:
        raise ValueError(
            "--power must be positive."
        )

    k = min(
        neighbors,
        len(source_xyz),
    )

    tree = cKDTree(
        source_xyz
    )

    distances, indices = tree.query(
        target_xyz,
        k=k,
        workers=-1,
    )

    # cKDTree returns 1D arrays for k=1.
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    nearest_distance = (
        distances[:, 0]
    )

    neighbor_values = (
        source_values[
            indices
        ]
    )

    interpolated = np.empty(
        len(target_xyz),
        dtype=np.float64,
    )

    exact_mask = (
        distances[:, 0]
        <= 1.0e-12
    )

    # Exact nearest-point matches.
    if np.any(exact_mask):
        interpolated[
            exact_mask
        ] = (
            neighbor_values[
                exact_mask,
                0,
            ]
        )

    non_exact = ~exact_mask

    if np.any(non_exact):
        d = distances[
            non_exact
        ]

        values = neighbor_values[
            non_exact
        ]

        weights = 1.0 / np.power(
            np.maximum(
                d,
                1.0e-12,
            ),
            power,
        )

        interpolated[
            non_exact
        ] = (
            np.sum(
                weights * values,
                axis=1,
            )
            / np.sum(
                weights,
                axis=1,
            )
        )

    if not np.isfinite(
        interpolated
    ).all():
        raise RuntimeError(
            "Interpolated values contain NaN or Inf."
        )

    return (
        interpolated,
        nearest_distance,
    )


# =============================================================================
# Diagnostics
# =============================================================================

def print_summary(
    df: pd.DataFrame,
    prediction_csv: Path,
    stl_path: Path,
    model: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    neighbors: int,
    power: float,
    nearest_distance: np.ndarray,
) -> None:
    print()
    print("=" * 78)
    print("AI-CFD INTERPOLATED SURFACE VISUALIZATION")
    print("=" * 78)
    print(
        f"Model              : {model.upper()}"
    )
    print(
        f"Prediction CSV     : {prediction_csv}"
    )
    print(
        f"STL                : {stl_path}"
    )
    print(
        f"Prediction points  : {len(df):,}"
    )
    print(
        f"STL vertices       : {len(vertices):,}"
    )
    print(
        f"STL triangles      : {len(faces):,}"
    )
    print(
        f"Velocity           : {df['velocity'].iloc[0]:g} m/s"
    )
    print(
        f"IDW neighbors      : {neighbors}"
    )
    print(
        f"IDW power          : {power:g}"
    )
    print()
    print("[SOURCE → STL DISTANCE]")
    print(
        "Nearest mean       : "
        f"{nearest_distance.mean() * 1000:.3f} mm"
    )
    print(
        "Nearest median     : "
        f"{np.median(nearest_distance) * 1000:.3f} mm"
    )
    print(
        "Nearest P95        : "
        f"{np.percentile(nearest_distance, 95) * 1000:.3f} mm"
    )
    print(
        "Nearest max        : "
        f"{nearest_distance.max() * 1000:.3f} mm"
    )
    print("=" * 78)


# =============================================================================
# Plot
# =============================================================================

def make_mesh_plot(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    title: str,
    value_label: str,
    unit: str,
    velocity: float,
) -> go.Figure:
    """
    Render interpolated values on the actual reconstructed STL triangles.
    """
    customdata = np.column_stack(
        (
            values,
            np.full(
                len(values),
                velocity,
                dtype=np.float64,
            ),
        )
    )

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=values,
                intensitymode="vertex",
                colorscale=COLOR_SCALE,
                cmin=float(
                    np.min(values)
                ),
                cmax=float(
                    np.max(values)
                ),
                showscale=True,
                colorbar=dict(
                    title=unit,
                ),
                flatshading=FLAT_SHADING,
                customdata=customdata,
                hovertemplate=(
                    "x=%{x:.6f} m<br>"
                    "y=%{y:.6f} m<br>"
                    "z=%{z:.6f} m<br>"
                    f"{value_label}=%{{customdata[0]:.6g}} {unit}<br>"
                    "velocity=%{customdata[1]:.6g} m/s"
                    "<extra></extra>"
                ),
                lighting=dict(
                    ambient=0.65,
                    diffuse=0.75,
                    specular=0.15,
                    roughness=0.85,
                    fresnel=0.05,
                ),
                lightposition=dict(
                    x=100,
                    y=200,
                    z=300,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
        ),
    )

    return fig


def save_and_show(
    fig: go.Figure,
    output_path: Path,
    show: bool,
) -> None:
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    print(
        f"Saved: {output_path}"
    )

    if show:
        fig.show()


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate AI-CFD prediction points onto the reconstructed "
            "STL surface and render smooth 3D fields."
        )
    )

    parser.add_argument(
        "--model",
        choices=(
            "mlp",
            "dgcnn",
        ),
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Prediction CSV filename or path."
        ),
    )

    parser.add_argument(
        "--stl",
        default=None,
        help=(
            "Optional STL filename/path. "
            "If omitted, infer from the prediction CSV name."
        ),
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=8,
        help=(
            "Number of nearest prediction points used for IDW "
            "interpolation (default: 8)."
        ),
    )

    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help=(
            "Inverse-distance weighting power (default: 2.0)."
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "Save HTML files without opening browser windows."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.neighbors <= 0:
        raise ValueError(
            "--neighbors must be positive."
        )

    if args.power <= 0.0:
        raise ValueError(
            "--power must be positive."
        )

    prediction_csv = (
        resolve_prediction_csv(
            model=args.model,
            input_value=args.input,
        )
    )

    stl_path = resolve_stl_path(
        prediction_csv=prediction_csv,
        stl_value=args.stl,
    )

    output_dir = build_output_dir(
        model=args.model,
        prediction_csv=prediction_csv,
    )

    df = load_prediction_csv(
        prediction_csv
    )

    source_xyz = df[
        [
            "x",
            "y",
            "z",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    source_htc = df[
        "predicted_htc"
    ].to_numpy(
        dtype=np.float64
    )

    source_shear = df[
        "predicted_wall_shear"
    ].to_numpy(
        dtype=np.float64
    )

    velocity = float(
        df[
            "velocity"
        ].iloc[0]
    )

    vertices, faces = load_stl_mesh(
        stl_path
    )

    print()
    print("[INTERPOLATING HTC]")

    start = time.perf_counter()

    interpolated_htc, nearest_distance = (
        interpolate_idw(
            source_xyz=source_xyz,
            source_values=source_htc,
            target_xyz=vertices,
            neighbors=args.neighbors,
            power=args.power,
        )
    )

    print(
        f"Completed in "
        f"{time.perf_counter() - start:.3f} s"
    )

    print()
    print("[INTERPOLATING WALL SHEAR]")

    start = time.perf_counter()

    interpolated_shear, nearest_distance_shear = (
        interpolate_idw(
            source_xyz=source_xyz,
            source_values=source_shear,
            target_xyz=vertices,
            neighbors=args.neighbors,
            power=args.power,
        )
    )

    print(
        f"Completed in "
        f"{time.perf_counter() - start:.3f} s"
    )

    # Same source and target coordinates are used for both fields,
    # so nearest-neighbor distances should be identical.
    if not np.allclose(
        nearest_distance,
        nearest_distance_shear,
    ):
        raise RuntimeError(
            "Unexpected nearest-distance mismatch between fields."
        )

    print_summary(
        df=df,
        prediction_csv=prediction_csv,
        stl_path=stl_path,
        model=args.model,
        vertices=vertices,
        faces=faces,
        neighbors=args.neighbors,
        power=args.power,
        nearest_distance=nearest_distance,
    )

    model_label = args.model.upper()

    htc_fig = make_mesh_plot(
        vertices=vertices,
        faces=faces,
        values=interpolated_htc,
        title=(
            f"{model_label} — Interpolated Predicted HTC"
        ),
        value_label="Predicted HTC",
        unit="W/(m²·K)",
        velocity=velocity,
    )

    shear_fig = make_mesh_plot(
        vertices=vertices,
        faces=faces,
        values=interpolated_shear,
        title=(
            f"{model_label} — Interpolated Predicted Wall Shear"
        ),
        value_label="Predicted wall shear",
        unit="Pa",
        velocity=velocity,
    )

    print()
    print("[GENERATING INTERPOLATED SURFACE PLOTS]")

    save_and_show(
        fig=htc_fig,
        output_path=(
            output_dir
            / "01_interpolated_htc.html"
        ),
        show=not args.no_show,
    )

    save_and_show(
        fig=shear_fig,
        output_path=(
            output_dir
            / "02_interpolated_wall_shear.html"
        ),
        show=not args.no_show,
    )

    print()
    print("=" * 78)
    print("INTERPOLATION VISUALIZATION COMPLETE")
    print("=" * 78)
    print(
        f"Output folder: {output_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
