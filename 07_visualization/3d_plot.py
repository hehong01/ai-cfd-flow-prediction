"""
Interactive 3D visualization for CFD and AI-prediction CSV files.

Modes
-----
1) CFD
   Input:
       ai-cfd-data/05_cfd_csv/<case>.csv

   Plots:
       geometry
       HTC
       wall shear
       pressure

2) Prediction
   Input:
       ai-cfd-data/07_predictions/<mlp|dgcnn>/prediction_csv/<case>.csv

   Plots:
       predicted HTC
       predicted wall shear

Results
-------
Saved under:

    ai-cfd-data/08_results/figures/3d_plot/

Examples
--------
From github/07_visualization:

    python 3d_plot.py --type cfd --input face_0003_05mps.csv

    python 3d_plot.py --type prediction --model mlp --input test_face_vel8.csv

    python 3d_plot.py --type prediction --model dgcnn --input test_face_vel8.csv

If the browser feels slow, display fewer points without changing the source CSV:

    python 3d_plot.py --type prediction --model mlp \
        --input test_face_vel8.csv --max-display-points 5000
"""

from __future__ import annotations

import argparse
import sys
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


CFD_DIR = DATA_ROOT / "05_cfd_csv"
PREDICTION_ROOT = DATA_ROOT / "07_predictions"

RESULT_ROOT = (
    DATA_ROOT
    / "08_results"
    / "figures"
    / "3d_plot"
)


# =============================================================================
# Plot settings
# =============================================================================

MARKER_SIZE = 2.2
MARKER_OPACITY = 0.90

# False:
#     use the full physical value range.
#
# True:
#     clip only the DISPLAY color scale to the 1st-99th percentiles.
#     Raw values themselves are never modified.
ROBUST_COLOR_RANGE = False


# =============================================================================
# Path resolution
# =============================================================================

def resolve_input_path(
    data_type: str,
    input_value: str,
    model: str | None,
) -> Path:
    """
    Resolve either:
        - an explicit path supplied by the user, or
        - a bare filename in the project's standard data folder.
    """
    path = Path(input_value)

    if path.is_file():
        return path.resolve()

    if path.is_absolute():
        raise FileNotFoundError(
            f"CSV file not found:\n{path}"
        )

    if data_type == "cfd":
        candidate = CFD_DIR / path

    else:
        if model is None:
            raise ValueError(
                "--model is required when --type prediction is used."
            )

        candidate = (
            PREDICTION_ROOT
            / model
            / "prediction_csv"
            / path
        )

    candidate = candidate.resolve()

    if not candidate.is_file():
        raise FileNotFoundError(
            f"CSV file not found:\n{candidate}"
        )

    return candidate


def build_output_dir(
    data_type: str,
    csv_path: Path,
    model: str | None,
) -> Path:
    """
    Result structure:

        08_results/
        └── figures/
            └── 3d_plot/
                ├── cfd/
                │   └── <case>/
                ├── mlp/
                │   └── <case>/
                └── dgcnn/
                    └── <case>/
    """
    if data_type == "cfd":
        branch = "cfd"
    else:
        if model is None:
            raise ValueError(
                "Prediction output requires a model name."
            )
        branch = model

    output_dir = (
        RESULT_ROOT
        / branch
        / csv_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# =============================================================================
# CSV loading
# =============================================================================

def load_cfd_csv(
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
        "nodenumber",
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "pressure",
        "wall-shear",
        "heat-transfer-coef",
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "CFD CSV is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(df.columns)
        )

    numeric_columns = (
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "pressure",
        "wall-shear",
        "heat-transfer-coef",
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    values = df[
        list(numeric_columns)
    ].to_numpy(dtype=np.float64)

    if len(df) == 0:
        raise ValueError(
            "CFD CSV contains no rows."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "CFD CSV contains NaN or Inf."
        )

    return df


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
# Optional display downsampling
# =============================================================================

def select_display_rows(
    df: pd.DataFrame,
    max_display_points: int | None,
) -> pd.DataFrame:
    """
    Reduce only the number of points sent to Plotly.

    The source CSV and diagnostics remain unchanged.

    Deterministic evenly spaced row indices are used so repeated runs
    produce the same visualization.
    """
    if max_display_points is None:
        return df

    if max_display_points <= 0:
        return df

    if len(df) <= max_display_points:
        return df

    indices = np.linspace(
        0,
        len(df) - 1,
        num=max_display_points,
        dtype=np.int64,
    )

    return (
        df.iloc[indices]
        .reset_index(drop=True)
    )


# =============================================================================
# Diagnostics
# =============================================================================

def print_cfd_summary(
    df: pd.DataFrame,
    csv_path: Path,
    display_count: int,
) -> None:
    xyz = df[
        [
            "x-coordinate",
            "y-coordinate",
            "z-coordinate",
        ]
    ].to_numpy(dtype=np.float64)

    extents = np.ptp(
        xyz,
        axis=0,
    )

    print()
    print("=" * 78)
    print("CFD 3D VISUALIZATION")
    print("=" * 78)
    print(f"CSV            : {csv_path}")
    print(f"Source points  : {len(df):,}")
    print(f"Display points : {display_count:,}")
    print()
    print("[GEOMETRY]")
    print(
        f"X : {xyz[:, 0].min(): .6f} ~ "
        f"{xyz[:, 0].max(): .6f} m"
        f"   extent={extents[0] * 1000:.2f} mm"
    )
    print(
        f"Y : {xyz[:, 1].min(): .6f} ~ "
        f"{xyz[:, 1].max(): .6f} m"
        f"   extent={extents[1] * 1000:.2f} mm"
    )
    print(
        f"Z : {xyz[:, 2].min(): .6f} ~ "
        f"{xyz[:, 2].max(): .6f} m"
        f"   extent={extents[2] * 1000:.2f} mm"
    )
    print()
    print("[CFD FIELDS]")

    for column, label, unit in (
        (
            "heat-transfer-coef",
            "HTC",
            "W/(m²·K)",
        ),
        (
            "wall-shear",
            "Wall shear",
            "Pa",
        ),
        (
            "pressure",
            "Pressure",
            "Pa",
        ),
    ):
        values = df[
            column
        ].to_numpy(dtype=np.float64)

        print(
            f"{label:<11}: "
            f"min={values.min():.6f}, "
            f"mean={values.mean():.6f}, "
            f"max={values.max():.6f} {unit}"
        )

    print("=" * 78)


def print_prediction_summary(
    df: pd.DataFrame,
    csv_path: Path,
    model: str,
    display_count: int,
) -> None:
    xyz = df[
        ["x", "y", "z"]
    ].to_numpy(dtype=np.float64)

    extents = np.ptp(
        xyz,
        axis=0,
    )

    htc = df[
        "predicted_htc"
    ].to_numpy(dtype=np.float64)

    shear = df[
        "predicted_wall_shear"
    ].to_numpy(dtype=np.float64)

    print()
    print("=" * 78)
    print("AI PREDICTION 3D VISUALIZATION")
    print("=" * 78)
    print(f"Model          : {model.upper()}")
    print(f"CSV            : {csv_path}")
    print(f"Source points  : {len(df):,}")
    print(f"Display points : {display_count:,}")
    print(
        f"Velocity       : "
        f"{df['velocity'].iloc[0]:g} m/s"
    )
    print()
    print("[GEOMETRY]")
    print(
        f"X : {xyz[:, 0].min(): .6f} ~ "
        f"{xyz[:, 0].max(): .6f} m"
        f"   extent={extents[0] * 1000:.2f} mm"
    )
    print(
        f"Y : {xyz[:, 1].min(): .6f} ~ "
        f"{xyz[:, 1].max(): .6f} m"
        f"   extent={extents[1] * 1000:.2f} mm"
    )
    print(
        f"Z : {xyz[:, 2].min(): .6f} ~ "
        f"{xyz[:, 2].max(): .6f} m"
        f"   extent={extents[2] * 1000:.2f} mm"
    )
    print()
    print("[PREDICTION]")
    print(
        "HTC        : "
        f"min={htc.min():.6f}, "
        f"mean={htc.mean():.6f}, "
        f"max={htc.max():.6f} W/(m²·K)"
    )
    print(
        "Wall shear : "
        f"min={shear.min():.6f}, "
        f"mean={shear.mean():.6f}, "
        f"max={shear.max():.6f} Pa"
    )
    print("=" * 78)


# =============================================================================
# Plot helpers
# =============================================================================

def base_scene() -> dict:
    return dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",
    )


def color_marker(
    values: np.ndarray,
    unit: str,
    full_values: np.ndarray,
) -> dict:
    marker = dict(
        size=MARKER_SIZE,
        color=values,
        colorscale="Turbo",
        opacity=MARKER_OPACITY,
        showscale=True,
        colorbar=dict(
            title=unit,
        ),
    )

    # Keep the displayed color range tied to the full source data,
    # even when Plotly itself receives only a subset of rows.
    if ROBUST_COLOR_RANGE:
        cmin, cmax = np.percentile(
            full_values,
            [1, 99],
        )
    else:
        cmin = float(
            np.min(full_values)
        )
        cmax = float(
            np.max(full_values)
        )

    marker["cmin"] = float(cmin)
    marker["cmax"] = float(cmax)

    return marker


def make_geometry_plot(
    x,
    y,
    z,
    title: str,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=MARKER_SIZE,
                    opacity=MARKER_OPACITY,
                ),
                hovertemplate=(
                    "x=%{x:.6f} m<br>"
                    "y=%{y:.6f} m<br>"
                    "z=%{z:.6f} m"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=base_scene(),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
        ),
    )

    return fig


def make_field_plot(
    x,
    y,
    z,
    values: np.ndarray,
    full_values: np.ndarray,
    title: str,
    value_label: str,
    unit: str,
    velocity: float | None = None,
) -> go.Figure:
    if velocity is None:
        customdata = np.column_stack(
            [values]
        )

        hovertemplate = (
            "x=%{x:.6f} m<br>"
            "y=%{y:.6f} m<br>"
            "z=%{z:.6f} m<br>"
            f"{value_label}=%{{customdata[0]:.6g}} {unit}"
            "<extra></extra>"
        )

    else:
        velocity_column = np.full(
            len(values),
            velocity,
            dtype=np.float64,
        )

        customdata = np.column_stack(
            (
                values,
                velocity_column,
            )
        )

        hovertemplate = (
            "x=%{x:.6f} m<br>"
            "y=%{y:.6f} m<br>"
            "z=%{z:.6f} m<br>"
            f"{value_label}=%{{customdata[0]:.6g}} {unit}<br>"
            "velocity=%{customdata[1]:.6g} m/s"
            "<extra></extra>"
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=color_marker(
                    values=values,
                    unit=unit,
                    full_values=full_values,
                ),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=base_scene(),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
        ),
    )

    return fig


# =============================================================================
# CFD plots
# =============================================================================

def build_cfd_plots(
    full_df: pd.DataFrame,
    display_df: pd.DataFrame,
) -> tuple[tuple[go.Figure, str], ...]:
    x = display_df["x-coordinate"]
    y = display_df["y-coordinate"]
    z = display_df["z-coordinate"]

    htc = display_df[
        "heat-transfer-coef"
    ].to_numpy(dtype=np.float64)

    shear = display_df[
        "wall-shear"
    ].to_numpy(dtype=np.float64)

    pressure = display_df[
        "pressure"
    ].to_numpy(dtype=np.float64)

    full_htc = full_df[
        "heat-transfer-coef"
    ].to_numpy(dtype=np.float64)

    full_shear = full_df[
        "wall-shear"
    ].to_numpy(dtype=np.float64)

    full_pressure = full_df[
        "pressure"
    ].to_numpy(dtype=np.float64)

    return (
        (
            make_geometry_plot(
                x,
                y,
                z,
                (
                    "CFD Wall Geometry — "
                    f"{len(display_df):,} displayed / "
                    f"{len(full_df):,} source points"
                ),
            ),
            "01_geometry_3d.html",
        ),
        (
            make_field_plot(
                x=x,
                y=y,
                z=z,
                values=htc,
                full_values=full_htc,
                title="CFD — Heat Transfer Coefficient",
                value_label="HTC",
                unit="W/(m²·K)",
            ),
            "02_htc_3d.html",
        ),
        (
            make_field_plot(
                x=x,
                y=y,
                z=z,
                values=shear,
                full_values=full_shear,
                title="CFD — Wall Shear Stress",
                value_label="Wall shear",
                unit="Pa",
            ),
            "03_wall_shear_3d.html",
        ),
        (
            make_field_plot(
                x=x,
                y=y,
                z=z,
                values=pressure,
                full_values=full_pressure,
                title="CFD — Pressure",
                value_label="Pressure",
                unit="Pa",
            ),
            "04_pressure_3d.html",
        ),
    )


# =============================================================================
# Prediction plots
# =============================================================================

def build_prediction_plots(
    full_df: pd.DataFrame,
    display_df: pd.DataFrame,
    model: str,
) -> tuple[tuple[go.Figure, str], ...]:
    x = display_df["x"]
    y = display_df["y"]
    z = display_df["z"]

    velocity = float(
        full_df["velocity"].iloc[0]
    )

    htc = display_df[
        "predicted_htc"
    ].to_numpy(dtype=np.float64)

    shear = display_df[
        "predicted_wall_shear"
    ].to_numpy(dtype=np.float64)

    full_htc = full_df[
        "predicted_htc"
    ].to_numpy(dtype=np.float64)

    full_shear = full_df[
        "predicted_wall_shear"
    ].to_numpy(dtype=np.float64)

    model_label = model.upper()

    return (
        (
            make_field_plot(
                x=x,
                y=y,
                z=z,
                values=htc,
                full_values=full_htc,
                title=f"{model_label} — Predicted HTC",
                value_label="Predicted HTC",
                unit="W/(m²·K)",
                velocity=velocity,
            ),
            "01_predicted_htc_3d.html",
        ),
        (
            make_field_plot(
                x=x,
                y=y,
                z=z,
                values=shear,
                full_values=full_shear,
                title=f"{model_label} — Predicted Wall Shear",
                value_label="Predicted wall shear",
                unit="Pa",
                velocity=velocity,
            ),
            "02_predicted_wall_shear_3d.html",
        ),
    )


# =============================================================================
# Save / show
# =============================================================================

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
            "Interactive 3D visualization for Fluent CFD "
            "and AI prediction CSV files."
        )
    )

    parser.add_argument(
        "--type",
        choices=(
            "cfd",
            "prediction",
        ),
        required=True,
        dest="data_type",
        help=(
            "CSV type: 'cfd' for Fluent wall CSV, "
            "'prediction' for MLP/DGCNN output CSV."
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV filename or path."
        ),
    )

    parser.add_argument(
        "--model",
        choices=(
            "mlp",
            "dgcnn",
        ),
        default=None,
        help=(
            "Required only for --type prediction."
        ),
    )

    parser.add_argument(
        "--max-display-points",
        type=int,
        default=None,
        help=(
            "Optional maximum number of points sent to Plotly. "
            "Use this if browser interaction is slow. "
            "The source CSV and reported statistics are unchanged."
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

    if (
        args.data_type == "prediction"
        and args.model is None
    ):
        raise ValueError(
            "--model mlp or --model dgcnn is required "
            "for prediction visualization."
        )

    if (
        args.max_display_points is not None
        and args.max_display_points < 0
    ):
        raise ValueError(
            "--max-display-points must be zero or positive."
        )

    csv_path = resolve_input_path(
        data_type=args.data_type,
        input_value=args.input,
        model=args.model,
    )

    output_dir = build_output_dir(
        data_type=args.data_type,
        csv_path=csv_path,
        model=args.model,
    )

    if args.data_type == "cfd":
        full_df = load_cfd_csv(
            csv_path
        )

        display_df = select_display_rows(
            full_df,
            args.max_display_points,
        )

        print_cfd_summary(
            df=full_df,
            csv_path=csv_path,
            display_count=len(display_df),
        )

        plots = build_cfd_plots(
            full_df=full_df,
            display_df=display_df,
        )

    else:
        full_df = load_prediction_csv(
            csv_path
        )

        display_df = select_display_rows(
            full_df,
            args.max_display_points,
        )

        print_prediction_summary(
            df=full_df,
            csv_path=csv_path,
            model=args.model,
            display_count=len(display_df),
        )

        plots = build_prediction_plots(
            full_df=full_df,
            display_df=display_df,
            model=args.model,
        )

    print()
    print("[GENERATING 3D PLOTS]")

    for fig, filename in plots:
        save_and_show(
            fig=fig,
            output_path=(
                output_dir
                / filename
            ),
            show=not args.no_show,
        )

    print()
    print("=" * 78)
    print("3D VISUALIZATION COMPLETE")
    print("=" * 78)
    print(
        f"Output folder: {output_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
