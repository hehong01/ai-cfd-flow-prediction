# AI-CFD Flow Prediction

Reconstruction and extension of an undergraduate AI-CFD project for facial external-flow and heat-transfer field prediction.

## Pipeline

1. **Image to STL** — completed
2. **SpaceClaim preprocessing** — completed
3. **Fluent CFD automation** — completed
4. **CFD dataset preparation** — next
5. **MLP / DGCNN model training** — scaffold prepared
6. **AI prediction** — scaffold prepared
7. **3D visualization** — scaffold prepared

## Current validated state

- 100 face geometries processed through the geometry/CFD pipeline.
- Fluent simulations are generated at **5, 8, and 10 m/s**.
- The current CFD dataset therefore contains **300 simulation samples**.
- SpaceClaim automation uses a **5 mm shrinkwrap** by default.
- If Fluent detects the validated Share Topology self-intersection failure, `master_run.py` rebuilds only that geometry with the **4 mm shrinkwrap fallback** and retries the CFD stage.
- Completed CFD cases are detected and skipped, so the master pipeline can resume without recomputing finished cases.

## Repository structure

```text
github/
├─ 01_image_to_stl/
├─ 02_spaceclaim/
├─ 03_fluent_cfd/
├─ 04_cfd_dataset/
├─ 05_model_training/
├─ 06_ai_prediction/
├─ 07_visualization/
├─ docs/
├─ results/
├─ master_run.py
├─ project_paths.py
├─ requirements.txt
└─ README.md
```

The stage scripts remain separated by workflow step. Shared local data paths are managed in `project_paths.py`.

## Local data layout

By default, the repository and generated data are kept as sibling directories:

```text
ai-cfd-flow-prediction/
├─ github/
└─ ai-cfd-data/
   ├─ 01_images/
   ├─ 02_stl/
   ├─ 03_spaceclaim/
   ├─ 04_fluent/
   ├─ 05_cfd_csv/
   ├─ 06_weights/
   ├─ 07_predictions/
   └─ 08_results/
```

A different data location can be supplied with the `AI_CFD_DATA_ROOT` environment variable.

## Running the current pipeline

Run the master script with the Python environment used by the image-to-STL stage:

```powershell
cd C:\ai-cfd-flow-prediction\github
conda activate ai-cfd
python .\master_run.py
```

The geometry and CFD stages require the locally installed **ANSYS SpaceClaim / Fluent 2021 R1 (v211)** environment used during reconstruction and validation.

The three automation stages can also be run independently:

```powershell
python .\01_image_to_stl\image_to_stl.py
python .\02_spaceclaim\stl_to_scdoc.py --face face_0001
python .\03_fluent_cfd\scdoc_to_cfd.py --face face_0001
```

## Next step

The next implementation stage is `04_cfd_dataset`: audit the generated CFD CSV files, define the preprocessing pipeline, and create face-level train/validation/test splits before model training.

## Project note

The repository distinguishes the original undergraduate project from later reconstruction, automation, validation, and extension work.
