# AI Prediction

This directory contains the **inference stage** of the AI-CFD pipeline.

A new face image is converted into surface coordinates, then the trained **MLP** and **DGCNN** models from `05_model_training` are used to predict:

```text
HTC
wall shear
```

No new CFD simulation or model training is performed here.

---

## 1. Pipeline

```text
JPG / PNG image
      ↓
image → STL
      ↓
STL surface sampling
      ↓
surface XYZ
      ↓
+ inlet velocity
      ↓
trained MLP / DGCNN
      ↓
predicted HTC / wall shear
      ↓
prediction CSV
```

The image-to-STL stage reuses the existing implementation in:

```text
01_image_to_stl/image_to_stl.py
```

---

## 2. Directory Structure

```text
06_ai_prediction/
├── README.md
├── preprocess.py
├── mlp/
│   └── predict.py
├── dgcnn/
│   └── predict.py
└── assets/
```

Generated prediction data are stored under:

```text
ai-cfd-data/07_predictions/
├── input_image/
├── stl/
├── mlp/
│   ├── input_csv/
│   └── prediction_csv/
└── dgcnn/
    ├── input_csv/
    └── prediction_csv/
```

---

## 3. Preprocessing

`preprocess.py` performs:

```text
image
  ↓
STL
  ↓
10,000 sampled surface points
  ├─ MLP input CSV: 10,000 points
  └─ DGCNN input CSV: FPS → 7,000 points
```

STL coordinates are converted from **mm to m** before model input is generated.

The DGCNN branch reuses the same farthest-point-sampling implementation used in `04_cfd_dataset`.

Generated CSV format:

```text
x,y,z
```

The inlet velocity is added later during prediction.

### Run

From:

```text
github/06_ai_prediction/
```

Both models:

```powershell
python preprocess.py --model both --input test_face.jpg
```

MLP only:

```powershell
python preprocess.py --model mlp --input test_face.jpg
```

DGCNN only:

```powershell
python preprocess.py --model dgcnn --input test_face.jpg
```

Existing up-to-date STL/CSV files are automatically skipped.

---

## 4. MLP Prediction

The MLP uses:

```text
[x, y, z, velocity]
        ↓
4 → 256 → 256 → 256 → 256 → 2
        ↓
[HTC, wall shear]
```

Trained artifacts:

```text
05_model_training/weights/mlp/best_model.pt
05_model_training/weights/mlp/scalers.npz
```

Final model:

```text
Parameters : 199,170
Best epoch : 6
```

### Run

From:

```text
github/06_ai_prediction/mlp/
```

```powershell
python predict.py --velocity 8 --input test_face.csv
```

Output:

```text
ai-cfd-data/07_predictions/mlp/prediction_csv/test_face_vel8.csv
```

---

## 5. DGCNN Prediction

The DGCNN uses one complete **7,000-point cloud** as a single sample.

```text
7000 × [x, y, z, velocity]
              ↓
            DGCNN
              ↓
7000 × [HTC, wall shear]
```

Trained artifacts:

```text
05_model_training/weights/dgcnn/best_model.pt
05_model_training/weights/dgcnn/scalers.npz
```

Final model:

```text
Parameters      : 189,826
Best epoch      : 96
k               : 20
Input points    : 7,000
First k-NN      : raw XYZ
```

The first k-NN graph is constructed from **raw physical XYZ**, while model features use the saved DGCNN scaler.

### Run

From:

```text
github/06_ai_prediction/dgcnn/
```

```powershell
python predict.py --velocity 8 --input test_face.csv
```

Output:

```text
ai-cfd-data/07_predictions/dgcnn/prediction_csv/test_face_vel8.csv
```

---

## 6. Prediction CSV

MLP and DGCNN use the same output format:

```text
x,y,z,velocity,predicted_htc,predicted_wall_shear
```

This allows both outputs to be passed directly to the later visualization stage.

```text
prediction CSV
      ↓
07_visualization
```

---

## 7. End-to-End Example

From `github/06_ai_prediction/`:

```powershell
python preprocess.py --model both --input test_face.jpg

python mlp/predict.py --velocity 8 --input test_face.csv

python dgcnn/predict.py --velocity 8 --input test_face.csv
```

Generated files:

```text
07_predictions/
├── stl/
│   └── test_face.stl
├── mlp/
│   ├── input_csv/test_face.csv
│   └── prediction_csv/test_face_vel8.csv
└── dgcnn/
    ├── input_csv/test_face.csv
    └── prediction_csv/test_face_vel8.csv
```

---

## 8. Note

Training data use Fluent wall-surface nodes, while new-image inference uses points sampled from the reconstructed STL surface.

Therefore the inference point distribution is not exactly identical to the original CFD mesh-node distribution.
# Placeholder

Documentation will be completed as the original project pipeline is reconstructed and verified.
