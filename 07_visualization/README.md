# Visualization

This directory contains the visualization and interactive analysis stage of the AI-CFD pipeline.

It uses CFD or AI-prediction outputs to visualize:

```text
HTC
wall shear
pressure
```

Three tools are provided:

```text
3d_plot.py
    → raw 3D point visualization

interpolation.py
    → prediction values mapped onto the reconstructed STL surface

streamlit_app.py
    → end-to-end interactive prediction and hotspot analysis
```

---

## 1. Directory Structure

```text
07_visualization/
├── README.md
├── 3d_plot.py
├── interpolation.py
├── streamlit_app.py
└── assets/
    ├── raw_prediction_3d.png
    ├── raw_prediction_3d.html
    ├── interpolated_surface.png
    ├── interpolated_surface.html
    └── streamlit_overlap_hotspot.png
```

Generated visualization results are stored under:

```text
ai-cfd-data/08_results/figures/
```

---

# 2. Raw 3D Prediction Visualization

`3d_plot.py` visualizes prediction values directly on the sampled prediction points.

For AI prediction:

```text
MLP
→ 10,000 prediction points

DGCNN
→ 7,000 prediction points
```

Each point is colored by one of the three predicted quantities:

```text
predicted_htc
predicted_wall_shear
predicted_pressure
```

### Example

![Raw 3D prediction](assets/raw_prediction_3d.png)

Interactive Plotly file:

[Open raw 3D prediction](assets/raw_prediction_3d.html)

### Run

From:

```text
github/07_visualization/
```

MLP:

```powershell
python 3d_plot.py --type prediction --model mlp --input test_face_vel8.csv
```

DGCNN:

```powershell
python 3d_plot.py --type prediction --model dgcnn --input test_face_vel8.csv
```

Generated files:

```text
01_predicted_htc_3d.html
02_predicted_wall_shear_3d.html
03_predicted_pressure_3d.html
```

The same script can also visualize CFD CSV data using `--type cfd`.

---

# 3. Interpolated STL Surface

The AI models predict values only at sampled surface points.

`interpolation.py` maps those prediction points back onto the reconstructed STL surface using inverse-distance-weighted interpolation.

```text
prediction points
      ↓
IDW interpolation
      ↓
STL vertices
      ↓
continuous surface visualization
```

Current interpolation settings:

```text
Neighbors : 8
Power     : 2
```

### Example

![Interpolated STL surface](assets/interpolated_surface.png)

Interactive Plotly file:

[Open interpolated STL surface](assets/interpolated_surface.html)

### Run

MLP:

```powershell
python interpolation.py --model mlp --input test_face_vel8.csv
```

DGCNN:

```powershell
python interpolation.py --model dgcnn --input test_face_vel8.csv
```

Generated files:

```text
01_interpolated_htc.html
02_interpolated_wall_shear.html
03_interpolated_pressure.html
```

The script also reports the nearest-point distance between prediction points and STL vertices as a basic interpolation diagnostic.

---

# 4. Interactive Streamlit App

`streamlit_app.py` provides an interactive interface for the complete new-image prediction workflow.

The user only needs to:

```text
upload face image
      ↓
select MLP or DGCNN
      ↓
select inlet velocity
      ↓
Run Prediction
```

The app then automatically performs:

```text
image
→ STL reconstruction
→ surface sampling
→ model input generation
→ AI prediction
→ 3D visualization
```

### Run

```powershell
python -m streamlit run streamlit_app.py
```

---

## 4.1 Visualization Modes

Two visualization modes are available:

```text
Interpolated STL surface
Raw prediction points
```

Three quantities can be displayed:

```text
HTC
Wall Shear
Pressure
```

For a single quantity, the app supports:

```text
Full field
Top X%
```

`Top X%` displays only the highest numerical values of the selected quantity.

For pressure, this means:

```text
highest pressure values
```

not highest absolute pressure magnitude.

---

## 4.2 Overlap Hotspot Analysis

The app can also display regions where multiple high-value fields overlap.

Any two or all three quantities can be selected:

```text
HTC
Wall Shear
Pressure
```

Each quantity has its own percentile threshold.

Example:

```text
HTC Top 20%
AND
Wall Shear Top 10%
AND
Pressure Top 30%
```

Only the logical intersection of these regions is displayed.

The displayed hotspot can be colored by any selected quantity using:

```text
Color by
```

If the selected percentile regions do not overlap, the app reports zero overlap instead of raising an error.

### Example

![Streamlit overlap hotspot](assets/streamlit_overlap_hotspot.png)

The app also reports:

```text
threshold for each quantity
overlap point / vertex count
overlap ratio
selected-field statistics
interpolation diagnostics
```

---

# 5. Prediction CSV Interface

Both MLP and DGCNN prediction files use the same format:

```text
x,y,z,velocity,predicted_htc,predicted_wall_shear,predicted_pressure
```

This common schema allows the same visualization tools to be used for either model.

Input prediction files are read from:

```text
ai-cfd-data/07_predictions/
```

Visualization outputs are written to:

```text
ai-cfd-data/08_results/figures/
```

---

# 6. Summary

This stage provides three levels of visualization:

```text
Raw prediction points
        ↓
Interpolated STL surface
        ↓
Interactive hotspot analysis
```

`3d_plot.py` shows the model outputs directly on prediction points.

`interpolation.py` maps those values onto the reconstructed facial surface.

`streamlit_app.py` combines image upload, AI prediction, visualization, percentile filtering, and multi-field overlap analysis in one interactive interface.
