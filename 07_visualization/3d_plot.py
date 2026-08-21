"""
Interactive 3D diagnostic plots for Fluent wall CSV.

Purpose
-------
1. Check whether the exported wall point cloud covers the full face/head.
2. Check whether the point density is obviously too sparse or uneven.
3. Inspect HTC, wall shear, and pressure fields on the actual CFD wall nodes.

Required packages
-----------------
pip install pandas numpy plotly

Optional:
pip install scipy
(scikit not required; scipy is used only for nearest-neighbor spacing statistics)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ======================================================================
# Default path
# ======================================================================

DEFAULT_CSV = Path(
    r"C:\ai-cfd-flow-prediction\ai-cfd-data\05_cfd_csv\face_0003_05mps.csv"
)


# ======================================================================
# Settings
# ======================================================================

MARKER_SIZE = 2.2
MARKER_OPACITY = 0.90

# False = use the full physical value range.
# True  = clip only the DISPLAY color range to the 1st~99th percentiles.
#         Raw values themselves are never modified.
ROBUST_COLOR_RANGE = False

SHOW_IN_BROWSER = True
SAVE_HTML = True


# ======================================================================
# CSV loading
# ======================================================================

def load_cfd_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Fluent exports can contain leading spaces in column names.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required_columns = [
        "nodenumber",
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "pressure",
        "wall-shear",
        "heat-transfer-coef",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(df.columns)
        )

    numeric_columns = [
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "pressure",
        "wall-shear",
        "heat-transfer-coef",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return df


# ======================================================================
# Diagnostics
# ======================================================================

def print_basic_diagnostics(
    df: pd.DataFrame,
    csv_path: Path,
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

    duplicate_xyz = int(
        df.duplicated(
            [
                "x-coordinate",
                "y-coordinate",
                "z-coordinate",
            ]
        ).sum()
    )

    numeric = df.select_dtypes(
        include=[np.number]
    )

    nan_count = int(
        numeric.isna().sum().sum()
    )

    inf_count = int(
        np.isinf(
            numeric.to_numpy(
                dtype=np.float64,
                copy=False,
            )
        ).sum()
    )

    print()
    print("=" * 78)
    print("CFD WALL CSV DIAGNOSTICS")
    print("=" * 78)
    print("CSV             :", csv_path)
    print(f"Wall points     : {len(df):,}")
    print(f"Duplicate XYZ   : {duplicate_xyz:,}")
    print(f"NaN values      : {nan_count:,}")
    print(f"Inf values      : {inf_count:,}")
    print()

    print("[COORDINATE RANGE]")
    print(
        "X : "
        f"{xyz[:, 0].min(): .6f} ~ {xyz[:, 0].max(): .6f} m"
        f"   extent = {extents[0] * 1000:.2f} mm"
    )
    print(
        "Y : "
        f"{xyz[:, 1].min(): .6f} ~ {xyz[:, 1].max(): .6f} m"
        f"   extent = {extents[1] * 1000:.2f} mm"
    )
    print(
        "Z : "
        f"{xyz[:, 2].min(): .6f} ~ {xyz[:, 2].max(): .6f} m"
        f"   extent = {extents[2] * 1000:.2f} mm"
    )
    print()

    for column, unit in (
        ("heat-transfer-coef", "W/m²K"),
        ("wall-shear", "Pa"),
        ("pressure", "Pa"),
    ):
        values = df[column].to_numpy(
            dtype=np.float64
        )

        print(
            f"{column:<20s}"
            f" min={values.min(): .6g}"
            f"  mean={values.mean(): .6g}"
            f"  max={values.max(): .6g}"
            f"  [{unit}]"
        )

    print("=" * 78)


def print_spacing_diagnostics(
    df: pd.DataFrame,
) -> None:
    """
    Nearest-neighbor spacing is a useful quick check of point-cloud resolution.
    This step is optional; the plots still work when scipy is not installed.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print()
        print(
            "[SPACING] scipy not installed -> "
            "nearest-neighbor spacing check skipped."
        )
        print(
            "Optional install: pip install scipy"
        )
        return

    xyz = df[
        [
            "x-coordinate",
            "y-coordinate",
            "z-coordinate",
        ]
    ].to_numpy(dtype=np.float64)

    tree = cKDTree(
        xyz
    )

    distances, _ = tree.query(
        xyz,
        k=2,
    )

    nearest = distances[:, 1]

    p05, p25, p50, p75, p95, p99 = np.percentile(
        nearest,
        [5, 25, 50, 75, 95, 99],
    )

    print()
    print("=" * 78)
    print("NEAREST-NEIGHBOR SPACING")
    print("=" * 78)
    print(
        f"Mean   : {nearest.mean() * 1000:.3f} mm"
    )
    print(
        f"P05    : {p05 * 1000:.3f} mm"
    )
    print(
        f"P25    : {p25 * 1000:.3f} mm"
    )
    print(
        f"Median : {p50 * 1000:.3f} mm"
    )
    print(
        f"P75    : {p75 * 1000:.3f} mm"
    )
    print(
        f"P95    : {p95 * 1000:.3f} mm"
    )
    print(
        f"P99    : {p99 * 1000:.3f} mm"
    )
    print(
        f"Max    : {nearest.max() * 1000:.3f} mm"
    )
    print("=" * 78)


# ======================================================================
# Plot helpers
# ======================================================================

def base_scene() -> dict:
    return dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",
    )


def make_geometry_plot(
    df: pd.DataFrame,
) -> go.Figure:
    """
    Uniform-color geometry view.
    Use this first to judge point density without field colors distracting you.
    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["x-coordinate"],
                y=df["y-coordinate"],
                z=df["z-coordinate"],
                mode="markers",
                marker=dict(
                    size=MARKER_SIZE,
                    opacity=MARKER_OPACITY,
                ),
                customdata=np.column_stack(
                    [
                        df["nodenumber"],
                    ]
                ),
                hovertemplate=(
                    "node=%{customdata[0]}<br>"
                    "x=%{x:.6f} m<br>"
                    "y=%{y:.6f} m<br>"
                    "z=%{z:.6f} m"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=(
            f"CFD Wall Geometry — {len(df):,} points"
        ),
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
    df: pd.DataFrame,
    value_column: str,
    title: str,
    unit: str,
) -> go.Figure:
    values = df[value_column].to_numpy(
        dtype=np.float64
    )

    marker = dict(
        size=MARKER_SIZE,
        color=values,
        colorscale="Viridis",
        opacity=MARKER_OPACITY,
        showscale=True,
        colorbar=dict(
            title=f"{value_column}<br>{unit}"
        ),
    )

    if ROBUST_COLOR_RANGE:
        cmin, cmax = np.percentile(
            values,
            [1, 99],
        )

        marker["cmin"] = float(cmin)
        marker["cmax"] = float(cmax)

    customdata = np.column_stack(
        [
            df["nodenumber"].to_numpy(),
            values,
        ]
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["x-coordinate"],
                y=df["y-coordinate"],
                z=df["z-coordinate"],
                mode="markers",
                marker=marker,
                customdata=customdata,
                hovertemplate=(
                    "node=%{customdata[0]}<br>"
                    "x=%{x:.6f} m<br>"
                    "y=%{y:.6f} m<br>"
                    "z=%{z:.6f} m<br>"
                    f"{value_column}=%{{customdata[1]:.6g}} {unit}"
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


def make_projection_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
) -> go.Figure:
    """
    2D orthographic projection.
    This is often easier than 3D for spotting holes or sparse regions.
    """
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=df[x_column],
                y=df[y_column],
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.85,
                ),
                hovertemplate=(
                    f"{x_column}=%{{x:.6f}} m<br>"
                    f"{y_column}=%{{y:.6f}} m"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=f"{x_column} (m)",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title=f"{y_column} (m)",
        ),
        margin=dict(
            l=60,
            r=20,
            b=60,
            t=50,
        ),
    )

    return fig


# ======================================================================
# Output
# ======================================================================

def save_and_show(
    fig: go.Figure,
    html_path: Path,
) -> None:
    if SAVE_HTML:
        fig.write_html(
            html_path,
            include_plotlyjs="cdn",
        )
        print(
            "Saved:",
            html_path,
        )

    if SHOW_IN_BROWSER:
        fig.show()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Fluent wall CSV geometry and CFD fields in interactive 3D."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=(
            "Path to one Fluent wall CSV. "
            f"Default: {DEFAULT_CSV}"
        ),
    )

    args = parser.parse_args()

    csv_path = args.csv.resolve()

    df = load_cfd_csv(
        csv_path
    )

    print_basic_diagnostics(
        df,
        csv_path,
    )

    print_spacing_diagnostics(
        df
    )

    output_dir = (
        csv_path.parent
        / f"{csv_path.stem}_diagnostic_plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots = [
        (
            make_geometry_plot(
                df
            ),
            output_dir / "01_geometry_3d.html",
        ),
        (
            make_field_plot(
                df,
                "heat-transfer-coef",
                "3D Heat Transfer Coefficient",
                "W/m²K",
            ),
            output_dir / "02_htc_3d.html",
        ),
        (
            make_field_plot(
                df,
                "wall-shear",
                "3D Wall Shear Stress",
                "Pa",
            ),
            output_dir / "03_wall_shear_3d.html",
        ),
        (
            make_field_plot(
                df,
                "pressure",
                "3D Pressure",
                "Pa",
            ),
            output_dir / "04_pressure_3d.html",
        ),
        (
            make_projection_plot(
                df,
                "x-coordinate",
                "y-coordinate",
                "XY Projection — point coverage check",
            ),
            output_dir / "05_xy_projection.html",
        ),
        (
            make_projection_plot(
                df,
                "x-coordinate",
                "z-coordinate",
                "XZ Projection — point coverage check",
            ),
            output_dir / "06_xz_projection.html",
        ),
        (
            make_projection_plot(
                df,
                "y-coordinate",
                "z-coordinate",
                "YZ Projection — point coverage check",
            ),
            output_dir / "07_yz_projection.html",
        ),
    ]

    print()
    print("=" * 78)
    print("GENERATING PLOTS")
    print("=" * 78)

    for fig, html_path in plots:
        save_and_show(
            fig,
            html_path,
        )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print("Output directory:")
    print(output_dir)
    print()
    print("Recommended inspection order:")
    print("1. 01_geometry_3d.html")
    print("2. 05_xy_projection.html")
    print("3. 06_xz_projection.html")
    print("4. 07_yz_projection.html")
    print("5. HTC / wall shear / pressure plots")
    print("=" * 78)


if __name__ == "__main__":
    main()
