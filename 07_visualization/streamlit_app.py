"""
AI-CFD Streamlit demo.

This app provides an end-to-end interface for the reconstructed AI-CFD
prediction pipeline:

    face image
        -> STL reconstruction
        -> surface-point preprocessing
        -> MLP or DGCNN inference
        -> HTC / wall shear / pressure prediction
        -> raw point or interpolated STL visualization
        -> optional Top-X% hotspot visualization
        -> multi-quantity overlap hotspot analysis

The app deliberately reuses the validated command-line pipeline instead of
reimplementing model preprocessing/inference:

    06_ai_prediction/preprocess.py
    06_ai_prediction/mlp/predict.py
    06_ai_prediction/dgcnn/predict.py

For visualization/interpolation, it reuses functions from:

    07_visualization/3d_plot.py
    07_visualization/interpolation.py

Recommended location:
    github/07_visualization/streamlit_app.py

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Project paths
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent

# Support either:
#   github/07_visualization/streamlit_app.py
# or
#   github/streamlit_app.py
if (THIS_DIR / "06_ai_prediction").is_dir():
    GITHUB_ROOT = THIS_DIR
elif (THIS_DIR.parent / "06_ai_prediction").is_dir():
    GITHUB_ROOT = THIS_DIR.parent
else:
    raise RuntimeError(
        "Could not locate the repository root. "
        "Place streamlit_app.py in the GitHub repository root "
        "or in github/07_visualization/."
    )

VISUALIZATION_DIR = GITHUB_ROOT / "07_visualization"
PREDICTION_CODE_DIR = GITHUB_ROOT / "06_ai_prediction"

if str(GITHUB_ROOT) not in sys.path:
    sys.path.insert(0, str(GITHUB_ROOT))

from project_paths import DATA_ROOT


PREDICTION_ROOT = DATA_ROOT / "07_predictions"

INPUT_IMAGE_DIR = PREDICTION_ROOT / "input_image"
STL_DIR = PREDICTION_ROOT / "stl"

PREPROCESS_SCRIPT = PREDICTION_CODE_DIR / "preprocess.py"
MLP_PREDICT_SCRIPT = PREDICTION_CODE_DIR / "mlp" / "predict.py"
DGCNN_PREDICT_SCRIPT = PREDICTION_CODE_DIR / "dgcnn" / "predict.py"

PLOT_SCRIPT = VISUALIZATION_DIR / "3d_plot.py"
INTERPOLATION_SCRIPT = VISUALIZATION_DIR / "interpolation.py"

MLP_MODEL = (
    GITHUB_ROOT
    / "05_model_training"
    / "weights"
    / "mlp"
    / "best_model.pt"
)
MLP_SCALER = (
    GITHUB_ROOT
    / "05_model_training"
    / "weights"
    / "mlp"
    / "scalers.npz"
)

DGCNN_MODEL = (
    GITHUB_ROOT
    / "05_model_training"
    / "weights"
    / "dgcnn"
    / "best_model.pt"
)
DGCNN_SCALER = (
    GITHUB_ROOT
    / "05_model_training"
    / "weights"
    / "dgcnn"
    / "scalers.npz"
)


# =============================================================================
# Streamlit page
# =============================================================================

st.set_page_config(
    page_title="AI-CFD Flow Prediction",
    page_icon="🌬️",
    layout="wide",
)

st.title("AI-CFD Flow Prediction")
st.caption(
    "Upload a facial image, select a trained surrogate model and inlet "
    "velocity, and visualize predicted HTC, wall shear, and pressure."
)


# =============================================================================
# Constants
# =============================================================================

TARGETS = {
    "HTC": {
        "column": "predicted_htc",
        "label": "Heat-transfer coefficient",
        "short_label": "HTC",
        "unit": "W/(m²·K)",
    },
    "Wall Shear": {
        "column": "predicted_wall_shear",
        "label": "Wall shear",
        "short_label": "Wall shear",
        "unit": "Pa",
    },
    "Pressure": {
        "column": "predicted_pressure",
        "label": "Pressure",
        "short_label": "Pressure",
        "unit": "Pa",
    },
}

MODEL_INFO = {
    "MLP": {
        "key": "mlp",
        "predict_script": MLP_PREDICT_SCRIPT,
        "model_path": MLP_MODEL,
        "scaler_path": MLP_SCALER,
        "point_count": 10_000,
        "description": (
            "Point-wise MLP: 4 → 256 → 256 → 256 → 256 → 3"
        ),
    },
    "DGCNN": {
        "key": "dgcnn",
        "predict_script": DGCNN_PREDICT_SCRIPT,
        "model_path": DGCNN_MODEL,
        "scaler_path": DGCNN_SCALER,
        "point_count": 7_000,
        "description": (
            "DGCNN: 7,000 FPS points, k=20, regression head "
            "512 → 256 → 128 → 3"
        ),
    },
}

IDW_NEIGHBORS = 8
IDW_POWER = 2.0


# =============================================================================
# Utility functions
# =============================================================================

def import_module_from_path(
    module_name: str,
    module_path: Path,
):
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Required module not found:\n{module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module from:\n{module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@st.cache_resource
def load_visualization_modules():
    plot_module = import_module_from_path(
        "ai_cfd_3d_plot",
        PLOT_SCRIPT,
    )

    interpolation_module = import_module_from_path(
        "ai_cfd_interpolation",
        INTERPOLATION_SCRIPT,
    )

    return plot_module, interpolation_module


def sanitize_uploaded_filename(
    original_name: str,
) -> tuple[str, str]:
    path = Path(original_name)

    suffix = path.suffix.lower()

    if suffix not in {
        ".jpg",
        ".jpeg",
        ".png",
    }:
        raise ValueError(
            "Only JPG, JPEG, and PNG images are supported."
        )

    stem = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        path.stem,
    ).strip("_")

    if not stem:
        stem = "uploaded_face"

    # Keep paths and generated filenames reasonably short.
    stem = stem[:80]

    return stem, suffix


def velocity_tag(
    velocity: float,
) -> str:
    text = f"{velocity:g}"
    text = text.replace(
        "-",
        "m",
    )
    text = text.replace(
        ".",
        "p",
    )

    return text


def run_subprocess(
    command: list[str],
) -> tuple[str, str]:
    result = subprocess.run(
        command,
        cwd=str(GITHUB_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        command_text = " ".join(
            command
        )

        raise RuntimeError(
            "Pipeline command failed.\n\n"
            f"Command:\n{command_text}\n\n"
            f"STDOUT:\n{stdout}\n\n"
            f"STDERR:\n{stderr}"
        )

    return stdout, stderr


def validate_required_files(
    model_name: str,
) -> list[str]:
    info = MODEL_INFO[
        model_name
    ]

    required = [
        PREPROCESS_SCRIPT,
        info["predict_script"],
        info["model_path"],
        info["scaler_path"],
        PLOT_SCRIPT,
        INTERPOLATION_SCRIPT,
    ]

    missing = [
        str(path)
        for path in required
        if not Path(path).is_file()
    ]

    return missing


def prediction_output_path(
    model_key: str,
    geometry_stem: str,
    velocity: float,
) -> Path:
    return (
        PREDICTION_ROOT
        / model_key
        / "prediction_csv"
        / (
            f"{geometry_stem}_"
            f"vel{velocity_tag(velocity)}.csv"
        )
    )


def model_input_csv_path(
    model_key: str,
    geometry_stem: str,
) -> Path:
    return (
        PREDICTION_ROOT
        / model_key
        / "input_csv"
        / f"{geometry_stem}.csv"
    )


def load_prediction_dataframe(
    csv_path: Path,
) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    required = [
        "x",
        "y",
        "z",
        "velocity",
        "predicted_htc",
        "predicted_wall_shear",
        "predicted_pressure",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Prediction CSV is missing required columns: "
            + ", ".join(
                missing
            )
        )

    for column in required:
        df[
            column
        ] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    values = df[
        required
    ].to_numpy(
        dtype=np.float64
    )

    if len(df) == 0:
        raise ValueError(
            "Prediction CSV contains no rows."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Prediction CSV contains NaN or Inf."
        )

    return df


def summarize_all_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for target_name, info in TARGETS.items():
        values = df[
            info["column"]
        ].to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "Quantity": target_name,
                "Unit": info["unit"],
                "Min": float(
                    np.min(values)
                ),
                "Mean": float(
                    np.mean(values)
                ),
                "Max": float(
                    np.max(values)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def percentile_threshold(
    values: np.ndarray,
    top_percent: float,
) -> float:
    if not (
        0.0
        < top_percent
        <= 100.0
    ):
        raise ValueError(
            "Top percentage must be in (0, 100]."
        )

    return float(
        np.percentile(
            values,
            100.0
            - top_percent,
        )
    )


def raw_point_figure(
    plot_module,
    df: pd.DataFrame,
    target_info: dict[str, str],
    model_name: str,
    region_mode: str,
    top_percent: float,
) -> tuple[go.Figure, dict[str, float]]:
    x_all = df[
        "x"
    ].to_numpy(
        dtype=np.float64
    )

    y_all = df[
        "y"
    ].to_numpy(
        dtype=np.float64
    )

    z_all = df[
        "z"
    ].to_numpy(
        dtype=np.float64
    )

    values_all = df[
        target_info["column"]
    ].to_numpy(
        dtype=np.float64
    )

    velocity = float(
        df[
            "velocity"
        ].iloc[0]
    )

    threshold = float(
        np.min(
            values_all
        )
    )

    if region_mode == "Top X%":
        threshold = percentile_threshold(
            values_all,
            top_percent,
        )

        mask = (
            values_all
            >= threshold
        )

        title = (
            f"{model_name} — "
            f"Top {top_percent:g}% "
            f"{target_info['short_label']}"
        )

    else:
        mask = np.ones(
            len(values_all),
            dtype=bool,
        )

        title = (
            f"{model_name} — "
            f"Predicted "
            f"{target_info['short_label']}"
        )

    if not np.any(
        mask
    ):
        raise RuntimeError(
            "The selected region contains no points."
        )

    fig = plot_module.make_field_plot(
        x=x_all[
            mask
        ],
        y=y_all[
            mask
        ],
        z=z_all[
            mask
        ],
        values=values_all[
            mask
        ],
        full_values=values_all,
        title=title,
        value_label=target_info[
            "label"
        ],
        unit=target_info[
            "unit"
        ],
        velocity=velocity,
    )

    selected_values = (
        values_all[
            mask
        ]
    )

    statistics = {
        "full_min": float(
            np.min(
                values_all
            )
        ),
        "full_mean": float(
            np.mean(
                values_all
            )
        ),
        "full_max": float(
            np.max(
                values_all
            )
        ),
        "threshold": threshold,
        "selected_count": float(
            np.count_nonzero(
                mask
            )
        ),
        "total_count": float(
            len(
                values_all
            )
        ),
        "selected_mean": float(
            np.mean(
                selected_values
            )
        ),
    }

    return fig, statistics


def top_surface_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    target_info: dict[str, str],
    model_name: str,
    velocity: float,
    top_percent: float,
) -> tuple[go.Figure, dict[str, float]]:
    threshold = percentile_threshold(
        values,
        top_percent,
    )

    hot_vertices = (
        values
        >= threshold
    )

    # A triangle is shown as part of the hotspot if at least
    # two of its three vertices are in the selected percentile.
    hot_vertex_count = np.sum(
        hot_vertices[
            faces
        ],
        axis=1,
    )

    hot_faces = faces[
        hot_vertex_count
        >= 2
    ]

    if len(
        hot_faces
    ) == 0:
        # Conservative fallback for very small selected regions:
        # show triangles touching at least one selected vertex.
        hot_faces = faces[
            hot_vertex_count
            >= 1
        ]

    if len(
        hot_faces
    ) == 0:
        raise RuntimeError(
            "No surface triangles remain after "
            "the Top-X% selection."
        )

    cmin = float(
        np.min(
            values
        )
    )
    cmax = float(
        np.max(
            values
        )
    )

    customdata = np.column_stack(
        (
            values,
            np.full(
                len(
                    values
                ),
                velocity,
                dtype=np.float64,
            ),
        )
    )

    fig = go.Figure()

    # Full STL shown as low-opacity geometry context.
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightgray",
            opacity=0.10,
            flatshading=False,
            hoverinfo="skip",
            name="Full surface",
            showscale=False,
        )
    )

    # Selected high-value surface region.
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=hot_faces[:, 0],
            j=hot_faces[:, 1],
            k=hot_faces[:, 2],
            intensity=values,
            colorscale="Turbo",
            cmin=cmin,
            cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=target_info[
                    "unit"
                ],
            ),
            flatshading=False,
            opacity=1.0,
            customdata=customdata,
            hovertemplate=(
                "x=%{x:.6f} m<br>"
                "y=%{y:.6f} m<br>"
                "z=%{z:.6f} m<br>"
                f"{target_info['label']}="
                "%{customdata[0]:.6g} "
                f"{target_info['unit']}<br>"
                "velocity="
                "%{customdata[1]:.6g} m/s"
                "<extra></extra>"
            ),
            name=(
                f"Top {top_percent:g}%"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"{model_name} — "
            f"Top {top_percent:g}% "
            f"{target_info['short_label']} "
            "(Interpolated STL Surface)"
        ),
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
        legend=dict(
            orientation="h",
        ),
    )

    selected_values = values[
        hot_vertices
    ]

    statistics = {
        "full_min": float(
            np.min(
                values
            )
        ),
        "full_mean": float(
            np.mean(
                values
            )
        ),
        "full_max": float(
            np.max(
                values
            )
        ),
        "threshold": threshold,
        "selected_count": float(
            np.count_nonzero(
                hot_vertices
            )
        ),
        "total_count": float(
            len(
                values
            )
        ),
        "selected_mean": float(
            np.mean(
                selected_values
            )
        ),
        "selected_triangles": float(
            len(
                hot_faces
            )
        ),
    }

    return fig, statistics


def interpolated_surface_figure(
    interpolation_module,
    df: pd.DataFrame,
    stl_path: Path,
    target_info: dict[str, str],
    model_name: str,
    region_mode: str,
    top_percent: float,
) -> tuple[
    go.Figure,
    dict[str, float],
    dict[str, float],
]:
    vertices, faces = (
        interpolation_module.load_stl_mesh(
            stl_path
        )
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

    source_values = df[
        target_info[
            "column"
        ]
    ].to_numpy(
        dtype=np.float64
    )

    interpolated, nearest_distance = (
        interpolation_module.interpolate_idw(
            source_xyz=source_xyz,
            source_values=source_values,
            target_xyz=vertices,
            neighbors=IDW_NEIGHBORS,
            power=IDW_POWER,
        )
    )

    velocity = float(
        df[
            "velocity"
        ].iloc[0]
    )

    if region_mode == "Top X%":
        fig, statistics = (
            top_surface_figure(
                vertices=vertices,
                faces=faces,
                values=interpolated,
                target_info=target_info,
                model_name=model_name,
                velocity=velocity,
                top_percent=top_percent,
            )
        )

    else:
        fig = (
            interpolation_module.make_mesh_plot(
                vertices=vertices,
                faces=faces,
                values=interpolated,
                title=(
                    f"{model_name} — "
                    f"Interpolated Predicted "
                    f"{target_info['short_label']}"
                ),
                value_label=target_info[
                    "label"
                ],
                unit=target_info[
                    "unit"
                ],
                velocity=velocity,
            )
        )

        statistics = {
            "full_min": float(
                np.min(
                    interpolated
                )
            ),
            "full_mean": float(
                np.mean(
                    interpolated
                )
            ),
            "full_max": float(
                np.max(
                    interpolated
                )
            ),
            "threshold": float(
                np.min(
                    interpolated
                )
            ),
            "selected_count": float(
                len(
                    interpolated
                )
            ),
            "total_count": float(
                len(
                    interpolated
                )
            ),
            "selected_mean": float(
                np.mean(
                    interpolated
                )
            ),
        }

    distance_stats = {
        "mean_mm": float(
            np.mean(
                nearest_distance
            )
            * 1000.0
        ),
        "median_mm": float(
            np.median(
                nearest_distance
            )
            * 1000.0
        ),
        "p95_mm": float(
            np.percentile(
                nearest_distance,
                95,
            )
            * 1000.0
        ),
        "max_mm": float(
            np.max(
                nearest_distance
            )
            * 1000.0
        ),
        "vertices": float(
            len(
                vertices
            )
        ),
        "triangles": float(
            len(
                faces
            )
        ),
    }

    return (
        fig,
        statistics,
        distance_stats,
    )



def build_overlap_mask(
    field_values: dict[str, np.ndarray],
    selected_targets: list[str],
    top_percentages: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Build an AND-intersection mask across independently configured
    Top-X% regions.

    Example:
        HTC top 20%
        AND
        Wall Shear top 10%
        AND
        Pressure top 30%

    Each selected quantity uses its own percentile cutoff.
    """
    if len(selected_targets) < 2:
        raise ValueError(
            "Overlap analysis requires at least two quantities."
        )

    first_target = selected_targets[0]

    if first_target not in field_values:
        raise KeyError(
            f"Missing field values for: {first_target}"
        )

    point_count = len(
        field_values[
            first_target
        ]
    )

    overlap_mask = np.ones(
        point_count,
        dtype=bool,
    )

    thresholds: dict[
        str,
        float,
    ] = {}

    for target_name in selected_targets:
        if target_name not in field_values:
            raise KeyError(
                f"Missing field values for: {target_name}"
            )

        if target_name not in top_percentages:
            raise KeyError(
                f"Missing Top-X% setting for: {target_name}"
            )

        values = np.asarray(
            field_values[
                target_name
            ],
            dtype=np.float64,
        )

        if values.shape != (
            point_count,
        ):
            raise ValueError(
                "All overlap fields must have the same length."
            )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{target_name} contains NaN or Inf."
            )

        top_percent = float(
            top_percentages[
                target_name
            ]
        )

        threshold = percentile_threshold(
            values,
            top_percent,
        )

        thresholds[
            target_name
        ] = threshold

        overlap_mask &= (
            values
            >= threshold
        )

    return (
        overlap_mask,
        thresholds,
    )

def raw_overlap_figure(
    df: pd.DataFrame,
    selected_targets: list[str],
    color_by: str,
    model_name: str,
    top_percentages: dict[str, float],
) -> tuple[
    go.Figure,
    dict[str, Any],
]:
    """
    Show the intersection of independently configured Top-X% regions
    directly on prediction points.

    A zero-size intersection is a valid analysis result. In that case,
    the full point cloud remains visible in gray and the function returns
    zero overlap statistics instead of raising an exception.
    """
    xyz = df[
        [
            "x",
            "y",
            "z",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    velocity = float(
        df[
            "velocity"
        ].iloc[0]
    )

    field_values = {
        target_name: df[
            TARGETS[
                target_name
            ][
                "column"
            ]
        ].to_numpy(
            dtype=np.float64
        )
        for target_name in selected_targets
    }

    overlap_mask, thresholds = (
        build_overlap_mask(
            field_values=field_values,
            selected_targets=selected_targets,
            top_percentages=top_percentages,
        )
    )

    selected_count = int(
        np.count_nonzero(
            overlap_mask
        )
    )

    color_info = TARGETS[
        color_by
    ]

    color_values = field_values[
        color_by
    ]

    cmin = float(
        np.min(
            color_values
        )
    )

    cmax = float(
        np.max(
            color_values
        )
    )

    fig = go.Figure()

    # Always keep the full sampled geometry as a faint context cloud.
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color="lightgray",
                opacity=0.12,
            ),
            hoverinfo="skip",
            name="Full point cloud",
        )
    )

    if selected_count > 0:
        selected_xyz = xyz[
            overlap_mask
        ]

        selected_color_values = color_values[
            overlap_mask
        ]

        customdata = np.column_stack(
            (
                selected_color_values,
                np.full(
                    selected_count,
                    velocity,
                    dtype=np.float64,
                ),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=selected_xyz[:, 0],
                y=selected_xyz[:, 1],
                z=selected_xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=selected_color_values,
                    colorscale="Turbo",
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(
                        title=color_info[
                            "unit"
                        ],
                    ),
                    opacity=1.0,
                ),
                customdata=customdata,
                hovertemplate=(
                    "x=%{x:.6f} m<br>"
                    "y=%{y:.6f} m<br>"
                    "z=%{z:.6f} m<br>"
                    f"{color_info['label']}="
                    "%{customdata[0]:.6g} "
                    f"{color_info['unit']}<br>"
                    "velocity="
                    "%{customdata[1]:.6g} m/s"
                    "<extra></extra>"
                ),
                name="Overlap hotspot",
            )
        )

        selected_mean = float(
            np.mean(
                selected_color_values
            )
        )

    else:
        selected_mean = None

        fig.add_annotation(
            text=(
                "No overlap at current thresholds.<br>"
                "Increase one or more Top-% settings."
            ),
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=16,
            ),
            bgcolor="rgba(30,30,30,0.70)",
            borderpad=8,
        )

    condition_text = (
        " ∩ ".join(
            (
                f"{target_name} Top "
                f"{top_percentages[target_name]:g}%"
            )
            for target_name in selected_targets
        )
    )

    fig.update_layout(
        title=(
            f"{model_name} — Overlap Hotspot: "
            f"{condition_text}"
        ),
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
            t=60,
        ),
        legend=dict(
            orientation="h",
        ),
    )

    total_count = len(
        xyz
    )

    statistics: dict[
        str,
        Any,
    ] = {
        "thresholds": thresholds,
        "top_percentages": dict(
            top_percentages
        ),
        "selected_count": selected_count,
        "total_count": total_count,
        "overlap_ratio": (
            100.0
            * selected_count
            / total_count
        ),
        "selected_mean": selected_mean,
        "color_min": float(
            np.min(
                color_values
            )
        ),
        "color_mean": float(
            np.mean(
                color_values
            )
        ),
        "color_max": float(
            np.max(
                color_values
            )
        ),
        "selected_triangles": None,
        "has_overlap": (
            selected_count
            > 0
        ),
    }

    return (
        fig,
        statistics,
    )

def overlap_surface_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    field_values: dict[str, np.ndarray],
    selected_targets: list[str],
    color_by: str,
    model_name: str,
    velocity: float,
    top_percentages: dict[str, float],
) -> tuple[
    go.Figure,
    dict[str, Any],
]:
    """
    Show the intersection of independently configured Top-X% regions
    on the reconstructed STL.

    Zero overlap is treated as a valid analysis result rather than an error.
    If overlap vertices exist but do not form displayable triangles, those
    vertices are shown as colored points on top of the gray STL.
    """
    overlap_mask, thresholds = (
        build_overlap_mask(
            field_values=field_values,
            selected_targets=selected_targets,
            top_percentages=top_percentages,
        )
    )

    selected_count = int(
        np.count_nonzero(
            overlap_mask
        )
    )

    color_info = TARGETS[
        color_by
    ]

    color_values = np.asarray(
        field_values[
            color_by
        ],
        dtype=np.float64,
    )

    cmin = float(
        np.min(
            color_values
        )
    )

    cmax = float(
        np.max(
            color_values
        )
    )

    fig = go.Figure()

    # Always show the full reconstructed face as geometric context.
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightgray",
            opacity=0.12,
            flatshading=False,
            hoverinfo="skip",
            name="Full surface",
            showscale=False,
        )
    )

    selected_triangles = 0
    selected_mean = None

    if selected_count > 0:
        selected_color_values = (
            color_values[
                overlap_mask
            ]
        )

        selected_mean = float(
            np.mean(
                selected_color_values
            )
        )

        overlap_vertex_count = np.sum(
            overlap_mask[
                faces
            ],
            axis=1,
        )

        overlap_faces = faces[
            overlap_vertex_count
            >= 2
        ]

        # If no triangle has two selected vertices, keep a looser fallback.
        if len(
            overlap_faces
        ) == 0:
            overlap_faces = faces[
                overlap_vertex_count
                >= 1
            ]

        selected_triangles = int(
            len(
                overlap_faces
            )
        )

        customdata = np.column_stack(
            (
                color_values,
                np.full(
                    len(
                        vertices
                    ),
                    velocity,
                    dtype=np.float64,
                ),
            )
        )

        if selected_triangles > 0:
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=overlap_faces[:, 0],
                    j=overlap_faces[:, 1],
                    k=overlap_faces[:, 2],
                    intensity=color_values,
                    colorscale="Turbo",
                    cmin=cmin,
                    cmax=cmax,
                    showscale=True,
                    colorbar=dict(
                        title=color_info[
                            "unit"
                        ],
                    ),
                    flatshading=False,
                    opacity=1.0,
                    customdata=customdata,
                    hovertemplate=(
                        "x=%{x:.6f} m<br>"
                        "y=%{y:.6f} m<br>"
                        "z=%{z:.6f} m<br>"
                        f"{color_info['label']}="
                        "%{customdata[0]:.6g} "
                        f"{color_info['unit']}<br>"
                        "velocity="
                        "%{customdata[1]:.6g} m/s"
                        "<extra></extra>"
                    ),
                    name="Overlap hotspot",
                )
            )

        else:
            # Extremely sparse but non-zero overlap:
            # show the selected vertices instead of failing.
            selected_vertices = vertices[
                overlap_mask
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=selected_vertices[:, 0],
                    y=selected_vertices[:, 1],
                    z=selected_vertices[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=selected_color_values,
                        colorscale="Turbo",
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=dict(
                            title=color_info[
                                "unit"
                            ],
                        ),
                        opacity=1.0,
                    ),
                    hovertemplate=(
                        "x=%{x:.6f} m<br>"
                        "y=%{y:.6f} m<br>"
                        "z=%{z:.6f} m"
                        "<extra></extra>"
                    ),
                    name="Overlap vertices",
                )
            )

    else:
        fig.add_annotation(
            text=(
                "No overlap at current thresholds.<br>"
                "Increase one or more Top-% settings."
            ),
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=16,
            ),
            bgcolor="rgba(30,30,30,0.70)",
            borderpad=8,
        )

    condition_text = (
        " ∩ ".join(
            (
                f"{target_name} Top "
                f"{top_percentages[target_name]:g}%"
            )
            for target_name in selected_targets
        )
    )

    fig.update_layout(
        title=(
            f"{model_name} — Overlap Hotspot: "
            f"{condition_text} "
            "(Interpolated STL Surface)"
        ),
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
            t=60,
        ),
        legend=dict(
            orientation="h",
        ),
    )

    total_count = len(
        vertices
    )

    statistics: dict[
        str,
        Any,
    ] = {
        "thresholds": thresholds,
        "top_percentages": dict(
            top_percentages
        ),
        "selected_count": selected_count,
        "total_count": total_count,
        "overlap_ratio": (
            100.0
            * selected_count
            / total_count
        ),
        "selected_mean": selected_mean,
        "color_min": float(
            np.min(
                color_values
            )
        ),
        "color_mean": float(
            np.mean(
                color_values
            )
        ),
        "color_max": float(
            np.max(
                color_values
            )
        ),
        "selected_triangles": selected_triangles,
        "has_overlap": (
            selected_count
            > 0
        ),
    }

    return (
        fig,
        statistics,
    )

def interpolated_overlap_figure(
    interpolation_module,
    df: pd.DataFrame,
    stl_path: Path,
    selected_targets: list[str],
    color_by: str,
    model_name: str,
    top_percentages: dict[str, float],
) -> tuple[
    go.Figure,
    dict[str, Any],
    dict[str, float],
]:
    """
    Interpolate every selected target to the same STL vertices, then compute
    the AND-intersection using each target's own Top-X% setting.
    """
    vertices, faces = (
        interpolation_module.load_stl_mesh(
            stl_path
        )
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

    field_values: dict[
        str,
        np.ndarray,
    ] = {}

    nearest_reference = None

    for target_name in selected_targets:
        target_info = TARGETS[
            target_name
        ]

        source_values = df[
            target_info[
                "column"
            ]
        ].to_numpy(
            dtype=np.float64
        )

        interpolated, nearest_distance = (
            interpolation_module.interpolate_idw(
                source_xyz=source_xyz,
                source_values=source_values,
                target_xyz=vertices,
                neighbors=IDW_NEIGHBORS,
                power=IDW_POWER,
            )
        )

        field_values[
            target_name
        ] = interpolated

        if nearest_reference is None:
            nearest_reference = nearest_distance

        elif not np.allclose(
            nearest_reference,
            nearest_distance,
        ):
            raise RuntimeError(
                "Unexpected nearest-distance mismatch "
                "between interpolated overlap fields."
            )

    if nearest_reference is None:
        raise RuntimeError(
            "No fields were interpolated for overlap analysis."
        )

    velocity = float(
        df[
            "velocity"
        ].iloc[0]
    )

    figure, statistics = (
        overlap_surface_figure(
            vertices=vertices,
            faces=faces,
            field_values=field_values,
            selected_targets=selected_targets,
            color_by=color_by,
            model_name=model_name,
            velocity=velocity,
            top_percentages=top_percentages,
        )
    )

    distance_stats = {
        "mean_mm": float(
            np.mean(
                nearest_reference
            )
            * 1000.0
        ),
        "median_mm": float(
            np.median(
                nearest_reference
            )
            * 1000.0
        ),
        "p95_mm": float(
            np.percentile(
                nearest_reference,
                95,
            )
            * 1000.0
        ),
        "max_mm": float(
            np.max(
                nearest_reference
            )
            * 1000.0
        ),
        "vertices": float(
            len(
                vertices
            )
        ),
        "triangles": float(
            len(
                faces
            )
        ),
    }

    return (
        figure,
        statistics,
        distance_stats,
    )

def metric_text(
    value: float,
) -> str:
    magnitude = abs(
        value
    )

    if (
        magnitude != 0.0
        and (
            magnitude
            >= 10000.0
            or magnitude
            < 0.001
        )
    ):
        return f"{value:.4e}"

    return f"{value:.6g}"


# =============================================================================
# Input controls
# =============================================================================

st.subheader(
    "1. Prediction Input"
)

control_col1, control_col2, control_col3 = st.columns(
    [
        1.6,
        1.0,
        1.0,
    ]
)

with control_col1:
    uploaded_file = st.file_uploader(
        "Face image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        help=(
            "The image is passed to the existing "
            "01_image_to_stl reconstruction stage."
        ),
    )

with control_col2:
    selected_model = st.radio(
        "Surrogate model",
        options=[
            "MLP",
            "DGCNN",
        ],
        horizontal=True,
    )

with control_col3:
    velocity = st.selectbox(
        "Inlet velocity (m/s)",
        options=[
            5.0,
            8.0,
            10.0,
        ],
        index=1,
        help=(
            "The final training dataset uses "
            "5, 8, and 10 m/s."
        ),
    )

st.caption(
    MODEL_INFO[
        selected_model
    ][
        "description"
    ]
)

missing_files = validate_required_files(
    selected_model
)

if missing_files:
    st.error(
        "The selected pipeline is incomplete. "
        "Required files are missing:"
    )

    st.code(
        "\n".join(
            missing_files
        )
    )

run_disabled = (
    uploaded_file is None
    or bool(
        missing_files
    )
)

run_prediction = st.button(
    "Run Prediction",
    type="primary",
    disabled=run_disabled,
)


# =============================================================================
# Execute end-to-end prediction
# =============================================================================

if run_prediction:
    assert uploaded_file is not None

    try:
        geometry_stem, image_suffix = (
            sanitize_uploaded_filename(
                uploaded_file.name
            )
        )

        model_info = MODEL_INFO[
            selected_model
        ]

        model_key = str(
            model_info[
                "key"
            ]
        )

        INPUT_IMAGE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = (
            INPUT_IMAGE_DIR
            / (
                geometry_stem
                + image_suffix
            )
        )

        image_path.write_bytes(
            uploaded_file.getvalue()
        )

        preprocess_command = [
            sys.executable,
            str(
                PREPROCESS_SCRIPT
            ),
            "--model",
            model_key,
            "--input",
            image_path.name,
            "--overwrite",
        ]

        input_csv_path = (
            model_input_csv_path(
                model_key=model_key,
                geometry_stem=geometry_stem,
            )
        )

        predict_command = [
            sys.executable,
            str(
                model_info[
                    "predict_script"
                ]
            ),
            "--velocity",
            f"{float(velocity):g}",
            "--input",
            input_csv_path.name,
            "--overwrite",
        ]

        with st.spinner(
            "Reconstructing STL and preparing the surface point cloud..."
        ):
            preprocess_stdout, preprocess_stderr = (
                run_subprocess(
                    preprocess_command
                )
            )

        with st.spinner(
            f"Running {selected_model} inference..."
        ):
            predict_stdout, predict_stderr = (
                run_subprocess(
                    predict_command
                )
            )

        prediction_csv = (
            prediction_output_path(
                model_key=model_key,
                geometry_stem=geometry_stem,
                velocity=float(
                    velocity
                ),
            )
        )

        stl_path = (
            STL_DIR
            / f"{geometry_stem}.stl"
        )

        if not input_csv_path.is_file():
            raise FileNotFoundError(
                "Preprocessing completed but the "
                f"expected input CSV was not found:\n{input_csv_path}"
            )

        if not prediction_csv.is_file():
            raise FileNotFoundError(
                "Inference completed but the expected "
                f"prediction CSV was not found:\n{prediction_csv}"
            )

        if not stl_path.is_file():
            raise FileNotFoundError(
                "Preprocessing completed but the expected "
                f"STL was not found:\n{stl_path}"
            )

        # Validate the final prediction file before storing it as
        # the current Streamlit result.
        result_df = (
            load_prediction_dataframe(
                prediction_csv
            )
        )

        st.session_state[
            "ai_cfd_result"
        ] = {
            "model_name": selected_model,
            "model_key": model_key,
            "velocity": float(
                velocity
            ),
            "geometry_stem": geometry_stem,
            "uploaded_name": uploaded_file.name,
            "image_path": str(
                image_path
            ),
            "stl_path": str(
                stl_path
            ),
            "input_csv": str(
                input_csv_path
            ),
            "prediction_csv": str(
                prediction_csv
            ),
            "point_count": int(
                len(
                    result_df
                )
            ),
            "preprocess_stdout": preprocess_stdout,
            "preprocess_stderr": preprocess_stderr,
            "predict_stdout": predict_stdout,
            "predict_stderr": predict_stderr,
        }

        st.success(
            "Prediction completed successfully."
        )

    except Exception as exc:
        st.error(
            "Prediction failed."
        )

        st.exception(
            exc
        )


# =============================================================================
# Result view
# =============================================================================

result = st.session_state.get(
    "ai_cfd_result"
)

if result is None:
    st.info(
        "Upload an image and click **Run Prediction** "
        "to generate a result."
    )

    st.stop()


st.divider()

st.subheader(
    "2. Prediction Result"
)

# Warn if controls were changed after the stored prediction was generated.
control_changed = (
    result[
        "model_name"
    ]
    != selected_model
    or not np.isclose(
        float(
            result[
                "velocity"
            ]
        ),
        float(
            velocity
        ),
    )
    or (
        uploaded_file is not None
        and result[
            "uploaded_name"
        ]
        != uploaded_file.name
    )
)

if control_changed:
    st.warning(
        "The controls have changed since the displayed prediction "
        "was generated. Click **Run Prediction** to refresh the result."
    )

result_model = str(
    result[
        "model_name"
    ]
)

result_velocity = float(
    result[
        "velocity"
    ]
)

prediction_csv = Path(
    str(
        result[
            "prediction_csv"
        ]
    )
)

stl_path = Path(
    str(
        result[
            "stl_path"
        ]
    )
)

image_path = Path(
    str(
        result[
            "image_path"
        ]
    )
)

df = load_prediction_dataframe(
    prediction_csv
)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(
    4
)

summary_col1.metric(
    "Model",
    result_model,
)

summary_col2.metric(
    "Velocity",
    f"{result_velocity:g} m/s",
)

summary_col3.metric(
    "Prediction points",
    f"{len(df):,}",
)

summary_col4.metric(
    "Targets",
    "3",
)

left_col, right_col = st.columns(
    [
        1.0,
        2.0,
    ]
)

with left_col:
    st.image(
        str(
            image_path
        ),
        caption=(
            f"Input image: "
            f"{result['uploaded_name']}"
        ),
        use_container_width=True,
    )

with right_col:
    st.markdown(
        "**Prediction summary**"
    )

    summary_table = (
        summarize_all_targets(
            df
        )
    )

    st.dataframe(
        summary_table.style.format(
            {
                "Min": "{:.6g}",
                "Mean": "{:.6g}",
                "Max": "{:.6g}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with prediction_csv.open(
    "rb"
) as file:
    st.download_button(
        "Download prediction CSV",
        data=file.read(),
        file_name=prediction_csv.name,
        mime="text/csv",
    )


# =============================================================================
# Visualization controls
# =============================================================================

st.subheader(
    "3. Interactive 3D Visualization"
)

analysis_mode = st.radio(
    "Analysis mode",
    options=[
        "Single quantity",
        "Overlap hotspot",
    ],
    horizontal=True,
    help=(
        "Single quantity shows one predicted field. "
        "Overlap hotspot displays only locations that simultaneously "
        "belong to the Top-X% regions of two or three quantities."
    ),
)

distance_stats = None

if analysis_mode == "Single quantity":
    viz_col1, viz_col2, viz_col3 = st.columns(
        3
    )

    with viz_col1:
        target_name = st.selectbox(
            "Quantity",
            options=list(
                TARGETS.keys()
            ),
            key="single_target",
        )

    with viz_col2:
        view_mode = st.radio(
            "View",
            options=[
                "Interpolated STL surface",
                "Raw prediction points",
            ],
            key="single_view",
        )

    with viz_col3:
        region_mode = st.radio(
            "Region",
            options=[
                "Full field",
                "Top X%",
            ],
            key="single_region",
        )

    top_percent = 10.0

    if region_mode == "Top X%":
        top_percent = float(
            st.slider(
                "Top percentage",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                format="%d%%",
                key="single_top_percent",
                help=(
                    "Displays the highest numerical values of "
                    "the selected quantity. For pressure, this means "
                    "highest pressure, not highest absolute pressure magnitude."
                ),
            )
        )

    target_info = TARGETS[
        target_name
    ]

    plot_module, interpolation_module = (
        load_visualization_modules()
    )

    if view_mode == "Raw prediction points":
        figure, statistics = raw_point_figure(
            plot_module=plot_module,
            df=df,
            target_info=target_info,
            model_name=result_model,
            region_mode=region_mode,
            top_percent=top_percent,
        )

    else:
        figure, statistics, distance_stats = (
            interpolated_surface_figure(
                interpolation_module=interpolation_module,
                df=df,
                stl_path=stl_path,
                target_info=target_info,
                model_name=result_model,
                region_mode=region_mode,
                top_percent=top_percent,
            )
        )

    # -----------------------------------------------------------------
    # Single-quantity statistics
    # -----------------------------------------------------------------

    stats_cols = st.columns(
        5
    )

    stats_cols[
        0
    ].metric(
        "Minimum",
        (
            f"{metric_text(statistics['full_min'])} "
            f"{target_info['unit']}"
        ),
    )

    stats_cols[
        1
    ].metric(
        "Mean",
        (
            f"{metric_text(statistics['full_mean'])} "
            f"{target_info['unit']}"
        ),
    )

    stats_cols[
        2
    ].metric(
        "Maximum",
        (
            f"{metric_text(statistics['full_max'])} "
            f"{target_info['unit']}"
        ),
    )

    if region_mode == "Top X%":
        stats_cols[
            3
        ].metric(
            (
                f"Top {top_percent:g}% "
                "threshold"
            ),
            (
                f"{metric_text(statistics['threshold'])} "
                f"{target_info['unit']}"
            ),
        )

        stats_cols[
            4
        ].metric(
            "Selected",
            (
                f"{int(statistics['selected_count']):,}"
                " / "
                f"{int(statistics['total_count']):,}"
            ),
        )

    else:
        stats_cols[
            3
        ].metric(
            "Displayed",
            (
                f"{int(statistics['selected_count']):,}"
            ),
        )

        stats_cols[
            4
        ].metric(
            "Selected mean",
            (
                f"{metric_text(statistics['selected_mean'])} "
                f"{target_info['unit']}"
            ),
        )

else:
    st.markdown(
        "**Overlap hotspot** selects only locations satisfying every "
        "chosen percentile condition simultaneously. "
        "Each quantity can use a different Top-X% cutoff."
    )

    overlap_col1, overlap_col2, overlap_col3 = st.columns(
        [
            1.4,
            1.0,
            1.0,
        ]
    )

    with overlap_col1:
        selected_targets = st.multiselect(
            "Quantities to overlap",
            options=list(
                TARGETS.keys()
            ),
            default=list(
                TARGETS.keys()
            ),
            key="overlap_targets",
            help=(
                "Select any two or all three predicted quantities. "
                "The displayed hotspot is their logical AND intersection."
            ),
        )

    if len(
        selected_targets
    ) < 2:
        st.warning(
            "Select at least two quantities for overlap analysis."
        )
        st.stop()

    with overlap_col2:
        view_mode = st.radio(
            "View",
            options=[
                "Interpolated STL surface",
                "Raw prediction points",
            ],
            key="overlap_view",
        )

    with overlap_col3:
        color_by = st.selectbox(
            "Color by",
            options=selected_targets,
            index=0,
            key="overlap_color_by",
            help=(
                "All selected quantities define the overlap. "
                "This option only chooses which field controls the color scale."
            ),
        )

    st.markdown(
        "**Top percentage by quantity**"
    )

    percentage_columns = st.columns(
        len(
            selected_targets
        )
    )

    top_percentages: dict[
        str,
        float,
    ] = {}

    for column, overlap_target in zip(
        percentage_columns,
        selected_targets,
    ):
        with column:
            top_percentages[
                overlap_target
            ] = float(
                st.slider(
                    f"{overlap_target} Top %",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    format="%d%%",
                    key=(
                        "overlap_top_percent_"
                        + overlap_target.lower().replace(
                            " ",
                            "_",
                        )
                    ),
                    help=(
                        f"Keep the highest numerical values of "
                        f"{overlap_target}. "
                        "For pressure, this is not absolute pressure."
                    ),
                )
            )

    condition_text = (
        " AND ".join(
            (
                f"{target_name} Top "
                f"{top_percentages[target_name]:g}%"
            )
            for target_name in selected_targets
        )
    )

    st.caption(
        f"Current overlap condition: {condition_text}"
    )

    plot_module, interpolation_module = (
        load_visualization_modules()
    )

    if view_mode == "Raw prediction points":
        figure, statistics = (
            raw_overlap_figure(
                df=df,
                selected_targets=selected_targets,
                color_by=color_by,
                model_name=result_model,
                top_percentages=top_percentages,
            )
        )

    else:
        figure, statistics, distance_stats = (
            interpolated_overlap_figure(
                interpolation_module=interpolation_module,
                df=df,
                stl_path=stl_path,
                selected_targets=selected_targets,
                color_by=color_by,
                model_name=result_model,
                top_percentages=top_percentages,
            )
        )

    color_info = TARGETS[
        color_by
    ]

    if not statistics.get(
        "has_overlap",
        False,
    ):
        st.warning(
            "No common hotspot exists for the current percentile settings. "
            "Increase one or more Top-% values to broaden the selected regions."
        )

    overlap_stats_cols = st.columns(
        5
    )

    overlap_stats_cols[
        0
    ].metric(
        f"{color_by} minimum",
        (
            f"{metric_text(statistics['color_min'])} "
            f"{color_info['unit']}"
        ),
    )

    overlap_stats_cols[
        1
    ].metric(
        f"{color_by} mean",
        (
            f"{metric_text(statistics['color_mean'])} "
            f"{color_info['unit']}"
        ),
    )

    overlap_stats_cols[
        2
    ].metric(
        f"{color_by} maximum",
        (
            f"{metric_text(statistics['color_max'])} "
            f"{color_info['unit']}"
        ),
    )

    overlap_stats_cols[
        3
    ].metric(
        "Overlap selected",
        (
            f"{int(statistics['selected_count']):,}"
            " / "
            f"{int(statistics['total_count']):,}"
        ),
    )

    overlap_stats_cols[
        4
    ].metric(
        "Overlap ratio",
        (
            f"{statistics['overlap_ratio']:.3f}%"
        ),
    )

    threshold_rows = []

    for overlap_target in selected_targets:
        overlap_info = TARGETS[
            overlap_target
        ]

        threshold_rows.append(
            {
                "Quantity": overlap_target,
                "Top region": (
                    f"Top "
                    f"{statistics['top_percentages'][overlap_target]:g}%"
                ),
                "Threshold": statistics[
                    "thresholds"
                ][
                    overlap_target
                ],
                "Unit": overlap_info[
                    "unit"
                ],
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    st.markdown(
        "**Overlap thresholds**"
    )

    st.dataframe(
        threshold_df.style.format(
            {
                "Threshold": "{:.6g}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    detail_col1, detail_col2 = st.columns(
        2
    )

    selected_mean_value = statistics[
        "selected_mean"
    ]

    if selected_mean_value is None:
        selected_mean_text = "N/A"
    else:
        selected_mean_text = (
            f"{metric_text(selected_mean_value)} "
            f"{color_info['unit']}"
        )

    detail_col1.metric(
        f"Mean {color_by} inside overlap",
        selected_mean_text,
    )

    if statistics[
        "selected_triangles"
    ] is not None:
        detail_col2.metric(
            "Displayed overlap triangles",
            f"{int(statistics['selected_triangles']):,}",
        )

    else:
        detail_col2.metric(
            "Displayed overlap points",
            f"{int(statistics['selected_count']):,}",
        )


st.plotly_chart(
    figure,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
    },
)


# =============================================================================
# Interpolation diagnostics
# =============================================================================

if distance_stats is not None:
    with st.expander(
        "Interpolation diagnostics"
    ):
        st.write(
            f"IDW neighbors: **{IDW_NEIGHBORS}**"
        )

        st.write(
            f"IDW power: **{IDW_POWER:g}**"
        )

        diagnostic_cols = st.columns(
            4
        )

        diagnostic_cols[
            0
        ].metric(
            "Nearest mean",
            (
                f"{distance_stats['mean_mm']:.3f} mm"
            ),
        )

        diagnostic_cols[
            1
        ].metric(
            "Nearest median",
            (
                f"{distance_stats['median_mm']:.3f} mm"
            ),
        )

        diagnostic_cols[
            2
        ].metric(
            "Nearest P95",
            (
                f"{distance_stats['p95_mm']:.3f} mm"
            ),
        )

        diagnostic_cols[
            3
        ].metric(
            "Nearest maximum",
            (
                f"{distance_stats['max_mm']:.3f} mm"
            ),
        )

        st.write(
            "STL vertices: "
            f"**{int(distance_stats['vertices']):,}**"
        )

        st.write(
            "STL triangles: "
            f"**{int(distance_stats['triangles']):,}**"
        )


# =============================================================================
# Files / logs
# =============================================================================

with st.expander(
    "Generated files"
):
    st.code(
        "\n".join(
            [
                f"Image          : {result['image_path']}",
                f"STL            : {result['stl_path']}",
                f"Model input CSV: {result['input_csv']}",
                f"Prediction CSV : {result['prediction_csv']}",
            ]
        )
    )

with st.expander(
    "Pipeline logs"
):
    st.markdown(
        "**Preprocessing**"
    )

    st.code(
        str(
            result[
                "preprocess_stdout"
            ]
        )
    )

    if str(
        result[
            "preprocess_stderr"
        ]
    ).strip():
        st.markdown(
            "**Preprocessing stderr**"
        )

        st.code(
            str(
                result[
                    "preprocess_stderr"
                ]
            )
        )

    st.markdown(
        "**Prediction**"
    )

    st.code(
        str(
            result[
                "predict_stdout"
            ]
        )
    )

    if str(
        result[
            "predict_stderr"
        ]
    ).strip():
        st.markdown(
            "**Prediction stderr**"
        )

        st.code(
            str(
                result[
                    "predict_stderr"
                ]
            )
        )


# =============================================================================
# Notes
# =============================================================================

with st.expander(
    "Method notes"
):
    st.markdown(
        """
- The app does **not** run a new Fluent CFD simulation.
- The uploaded image is reconstructed into an STL using the validated
  `01_image_to_stl` implementation.
- The MLP uses 10,000 sampled surface points.
- The DGCNN uses 7,000 FPS-selected surface points.
- Both models receive `[x, y, z, velocity]`.
- Both models predict `[HTC, wall shear, pressure]`.
- The interpolated view maps prediction points back onto the reconstructed
  STL using inverse-distance-weighted interpolation.
- **Top X%** means the highest numerical values of the currently selected
  quantity.
- **Overlap hotspot** supports any two or all three quantities.
- Each selected quantity has its own independent Top-X% cutoff, and the app
  displays only the logical AND intersection of those conditions.
- A zero-size intersection is a valid result. The app keeps the full geometry
  visible and reports zero overlap instead of raising an exception.
- In overlap mode, **Color by** changes only the color scale; it does not change
  the overlap condition.
- For pressure, Top-X% means highest numerical pressure, not highest absolute
  pressure magnitude.
- Training points originate from Fluent wall nodes, whereas new-image
  inference points are sampled from the reconstructed STL surface.
        """
    )
