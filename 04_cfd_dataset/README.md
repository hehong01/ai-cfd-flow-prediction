# CFD Dataset Preparation

This stage validates the CFD surface-field dataset generated in [`03_fluent_cfd`](../03_fluent_cfd/) and prepares it for machine-learning model training.

The CFD dataset consists of wall-surface results for 100 face geometries under three inlet velocities. The raw Fluent CSV files are first audited for structural and numerical consistency. For DGCNN training, each variable-size surface point cloud is then reduced to a fixed 7,000-point representation using Farthest Point Sampling (FPS).

The resulting dataset is used in [`05_model_training`](../05_model_training/) for:

- a point-wise MLP baseline
- a geometry-aware DGCNN model

---

## 1. Dataset Overview

The CFD pipeline produces three simulations for each face geometry:

- 5 m/s
- 8 m/s
- 10 m/s

With 100 geometries, the complete dataset contains:

```text
100 faces × 3 inlet velocities = 300 CFD samples
```

Example filenames:

```text
face_0001_05mps.csv
face_0001_08mps.csv
face_0001_10mps.csv

...

face_0100_05mps.csv
face_0100_08mps.csv
face_0100_10mps.csv
```

Each CSV contains Fluent wall-surface data with the following columns:

```text
nodenumber
x-coordinate
y-coordinate
z-coordinate
pressure
temperature
y-plus
wall-shear
heat-flux
heat-transfer-coef
```

For the current learning task, the model input and targets are defined as:

```text
Input
[x, y, z, velocity]

Target
[HTC, wall shear]
```

where:

```text
HTC = heat-transfer-coef
```

The remaining CFD quantities are retained in the original CSV files but are not used as model inputs.

---

## 2. Dataset Audit

`audit_dataset.py` performs a full read-only validation of the raw CFD CSV dataset.

It checks:

- expected number of CSV files
- expected filename format
- complete 5/8/10 m/s triplets for every face
- CSV header consistency
- row counts
- row-width consistency
- numeric parsing
- NaN values
- Inf values
- aggregate column statistics
- row-count consistency across the three velocity cases of each face

Run from the repository root:

```powershell
python .\04_cfd_dataset\audit_dataset.py
```

or from this directory:

```powershell
python .\audit_dataset.py
```

### Final validated dataset

The final audit produced:

```text
CSV files found          : 300
Matching expected names  : 300
Unexpected CSV names     : 0
Unique faces             : 100
Complete 3-speed triplets: 100
Incomplete faces         : 0
```

Row-count statistics:

```text
Total data rows  : 2,455,545
Min rows/sample  : 7,127
Max rows/sample  : 9,933
Mean rows/sample : 8,185.15
Median           : 8,099.00

Faces with 5/8/10 row-count mismatch: 0
```

Data-quality checks:

```text
Files with bad header : 0
Files with zero rows  : 0
Bad-width rows        : 0
Numeric parse errors  : 0
NaN values            : 0
Inf values            : 0
```

Final result:

```text
[PASS] Raw CFD dataset audit passed.
```

---

## 3. Mesh Outlier Detection and Correction

The initial dataset audit revealed one abnormal surface-mesh outlier.

For `face_0055`, the original 5 mm shrinkwrap-generated geometry produced only:

```text
2,704 wall nodes
```

while the rest of the dataset contained approximately 7,000-10,000 wall nodes per sample.

The same STL was regenerated using a 4 mm shrinkwrap resolution and rerun through the Fluent pipeline.

The corrected result was:

```text
5 mm shrinkwrap → 2,704 wall nodes
4 mm shrinkwrap → 9,487 wall nodes
```

for all three velocity cases:

```text
face_0055_05mps.csv → 9,487 rows
face_0055_08mps.csv → 9,487 rows
face_0055_10mps.csv → 9,487 rows
```

After regeneration, the complete 300-sample dataset passed the audit and the minimum node count increased to 7,127.

This check prevented a single abnormal mesh from determining the point-cloud resolution used for the entire DGCNN dataset.

---

## 4. Train / Validation / Test Split

The dataset is split by face geometry rather than by individual CFD file.

```text
Train
face_0001 ~ face_0080
80 faces × 3 velocities
= 240 samples

Validation
face_0081 ~ face_0090
10 faces × 3 velocities
= 30 samples

Test
face_0091 ~ face_0100
10 faces × 3 velocities
= 30 samples
```

Therefore:

```text
Train : 240 samples
Val   :  30 samples
Test  :  30 samples
Total : 300 samples
```

All velocity cases belonging to the same face remain in the same split.

This prevents the geometry of one face from appearing in both training and evaluation subsets.

---

## 5. Why DGCNN Requires Point-Count Standardization

The original Fluent wall meshes do not contain the same number of surface nodes.

After dataset validation:

```text
Minimum : 7,127 points
Median  : 8,099 points
Mean    : 8,185 points
Maximum : 9,933 points
```

A DGCNN sample represents an entire surface point cloud.

For batch training, each sample therefore needs the same number of points:

```text
Sample A → N₁ points
Sample B → N₂ points
Sample C → N₃ points

        ↓

fixed-size point clouds

Sample A → 7,000 points
Sample B → 7,000 points
Sample C → 7,000 points
```

The point coordinates themselves do **not** need to match between different faces.

Only the number of points is standardized.

---

## 6. Farthest Point Sampling

`preprocessing_dgcnn.py` reduces each CFD surface point cloud to exactly 7,000 points using Farthest Point Sampling (FPS).

### Why FPS?

Pure random sampling can accidentally leave locally sparse regions in the sampled surface.

FPS instead repeatedly selects the point farthest from the already-selected set, producing a spatially distributed subset of the original point cloud.

Conceptually:

```text
Original CFD surface
7,127 ~ 9,933 points

        ↓
Farthest Point Sampling

7,000 spatially distributed points
```

The current dataset retains a large fraction of the original CFD surface nodes:

```text
Minimum sample:
7000 / 7127 ≈ 98.2 %

Median sample:
7000 / 8099 ≈ 86.4 %

Maximum sample:
7000 / 9933 ≈ 70.5 %
```

The goal is therefore not aggressive geometric simplification, but point-count standardization while preserving surface coverage.

---

## 7. Independent Sampling for Every CFD CSV

Every CFD CSV is processed independently.

For example:

```text
face_0001_05mps.csv
        ↓
FPS based on its own xyz coordinates
        ↓
7000 row indices

face_0001_08mps.csv
        ↓
FPS based on its own xyz coordinates
        ↓
7000 row indices

face_0002_05mps.csv
        ↓
FPS based on its own xyz coordinates
        ↓
7000 row indices
```

No assumption is required that:

- different faces share matching point coordinates
- different CSV files share the same row order
- point #1 in one face corresponds to point #1 in another face

DGCNN reconstructs local geometric relationships from the point coordinates themselves.

---

## 8. FPS Index Generation

Run:

```powershell
python .\preprocessing_dgcnn.py --num-points 7000
```

The script:

1. finds all 300 CFD CSV files
2. reads the surface coordinates of each CSV
3. performs deterministic FPS
4. selects 7,000 unique points
5. validates the selected indices
6. stores the selected row indices

The raw CFD CSV files are not modified.

The generated local file is:

```text
fps_indices_7000.npz
```

It contains one independent index array for every CFD sample:

```text
face_0001_05mps → 7000 indices
face_0001_08mps → 7000 indices
face_0001_10mps → 7000 indices
...
face_0100_10mps → 7000 indices
```

A total of 300 index arrays are stored.

The indices are 0-based row indices into the corresponding original CFD CSV.

For example:

```text
Original CSV
8843 rows

        +

FPS row indices
7000 indices

        ↓

7000 selected CFD surface points
```

The generated NPZ file can be reproduced from the raw CFD CSV files and is therefore treated as a generated preprocessing artifact rather than source code.

---

## 9. PyTorch Dataset Loader

`dataset.py` connects:

```text
Raw CFD CSV
        +
FPS row-index mask
        ↓
Selected 7000 surface points
        ↓
PyTorch tensors
```

For every CFD sample, the loader produces:

```text
Input X
shape: (7000, 4)

[x, y, z, velocity]
```

and:

```text
Target Y
shape: (7000, 2)

[HTC, wall shear]
```

For one point:

```text
Input
[x_i, y_i, z_i, U]

Target
[HTC_i, wall_shear_i]
```

The FPS index is applied to the coordinates and target quantities simultaneously, so the correspondence between geometry and CFD results is preserved.

---

## 10. Dataset Tensor Structure

A single DGCNN sample has:

```text
X.shape = (7000, 4)
Y.shape = (7000, 2)
```

For a batch of size `B`:

```text
X.shape = (B, 7000, 4)
Y.shape = (B, 7000, 2)
```

Depending on the DGCNN implementation, the input tensor may later be rearranged to:

```text
(B, 4, 7000)
```

before entering the network.

This changes only tensor dimension order and does not change the underlying data.

---

## 11. Full Dataset Validation

Running `dataset.py` directly performs a complete loading test of all 300 samples.

```powershell
python .\dataset.py
```

The validation checks:

- expected number of samples in each split
- presence of every CFD CSV
- presence of every FPS index mask
- exactly 7,000 unique indices per sample
- FPS index bounds
- input tensor shape
- target tensor shape
- `torch.float32` dtype
- NaN / Inf values

Final runtime result:

```text
[TRAIN]
samples : 240
TRAIN PASSED: 240 samples

[VAL]
samples : 30
VAL PASSED: 30 samples

[TEST]
samples : 30
TEST PASSED: 30 samples
```

Final summary:

```text
FULL DATASET TEST PASSED

Total samples checked : 300
Points / sample       : 7000
Input                : [x, y, z, velocity]
Input shape          : (7000, 4)
Target               : [HTC, wall_shear]
Target shape         : (7000, 2)
Tensor dtype         : torch.float32
```

Therefore, the full path:

```text
CFD CSV
→ FPS mask
→ train/val/test split
→ model input tensor
→ target tensor
```

was validated across all 300 samples.

---

## 12. MLP Baseline

The MLP baseline uses the same physical input and target definitions:

```text
Input
[x, y, z, velocity]

Target
[HTC, wall shear]
```

However, the MLP treats each surface node independently.

Conceptually:

```text
HTC        = f(x, y, z, U)
wall shear = g(x, y, z, U)
```

Therefore, different CFD geometries do not need to contain the same number of nodes for MLP training.

The MLP can use the original CFD nodes directly and does not require the 7,000-point FPS preprocessing.

The MLP is used as a simple point-wise baseline, while DGCNN is intended to learn local geometric relationships within the surface point cloud.

---

## 13. File Structure

Source files in this stage:

```text
04_cfd_dataset/
├── audit_dataset.py
├── dataset.py
├── preprocessing_dgcnn.py
└── README.md
```

Generated locally:

```text
fps_indices_7000.npz
```

### `audit_dataset.py`

Validates the complete raw CFD CSV dataset and reports structural, numerical, and aggregate statistics.

### `preprocessing_dgcnn.py`

Applies independent 7,000-point Farthest Point Sampling to every CFD CSV and stores the selected row indices.

### `dataset.py`

Provides the PyTorch dataset interface for DGCNN training and performs full-dataset validation when executed directly.

### `fps_indices_7000.npz`

Generated FPS index masks used to reproduce the fixed 7,000-point representation of each CFD sample.

---

## 14. Dependencies

This stage uses:

```text
numpy
torch
```

The original CFD CSV files are produced by ANSYS Fluent in the previous stage.

---

## 15. Pipeline Position

```text
01_image_to_stl
        ↓
02_spaceclaim
        ↓
03_fluent_cfd
        ↓
04_cfd_dataset
        ↓
05_model_training
```

This stage converts the validated Fluent wall-field outputs into a reproducible machine-learning dataset interface.

The next stage trains:

- a point-wise MLP baseline
- a DGCNN point-cloud model

for prediction of:

```text
heat-transfer coefficient
wall shear
```

from:

```text
3D surface geometry
+
inlet velocity
```