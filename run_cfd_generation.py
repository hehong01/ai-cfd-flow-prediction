#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_cfd_generation.py

Batch wrapper for AI-CFD geometry and CFD-data generation.

Stages
------
1) image_to_stl.py
2) stl_to_scdoc.py          (default shrinkwrap: 5 mm)
3) scdoc_to_cfd.py          (5, 8, 10 m/s)

This wrapper generates the geometry and Fluent CFD cases used by the
later dataset-preparation and model-training stages. It does not run
dataset preprocessing, model training, AI inference, or visualization.

Validated geometry fallback
---------------------------
If scdoc_to_cfd.py returns exit code 42 because Fluent detected
Share Topology self-intersection:

    5 mm SCDOC
        -> Fluent
        -> exit 42
        -> rebuild only that face with 4 mm shrinkwrap
        -> retry Fluent once

The three stage scripts remain separate; this file only orchestrates them.

IMPORTANT
---------
Run this wrapper with the Python environment used by image_to_stl.py
(e.g. `conda activate ai-cfd` first).
"""

from pathlib import Path
import subprocess
import sys


from project_paths import (
    REPO_ROOT,
    IMAGE_DIR,
    STL_DIR,
    SPACECLAIM_DIR,
    FLUENT_DIR,
    CFD_CSV_DIR,
)

GITHUB = REPO_ROOT
SCDOC_DIR = SPACECLAIM_DIR
CSV_DIR = CFD_CSV_DIR

IMAGE_TO_STL = GITHUB / "01_image_to_stl" / "image_to_stl.py"
STL_TO_SCDOC = GITHUB / "02_spaceclaim" / "stl_to_scdoc.py"
SCDOC_TO_CFD = GITHUB / "03_fluent_cfd" / "scdoc_to_cfd.py"

PRIMARY_SHRINKWRAP_MM = 5.0
FALLBACK_SHRINKWRAP_MM = 4.0

SELF_INTERSECTION_EXIT_CODE = 42
CFD_TIMEOUT_SEC = 20 * 60

TAGS = ("05mps", "08mps", "10mps")


def run(cmd, cwd):
    cmd = [str(x) for x in cmd]

    print("\n" + "=" * 80)
    print("RUN:", subprocess.list2cmdline(cmd))
    print("=" * 80, flush=True)

    result = subprocess.run(cmd, cwd=str(cwd))

    print("RETURN CODE:", result.returncode, flush=True)
    return result.returncode


def source_faces():
    return sorted(
        p.stem
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and p.stem.startswith("face_")
    )


def complete_cfd(face):
    expected = []

    for tag in TAGS:
        expected += [
            FLUENT_DIR / f"{face}_{tag}.cas.h5",
            FLUENT_DIR / f"{face}_{tag}.dat.h5",
            CSV_DIR / f"{face}_{tag}.csv",
        ]

    return all(
        p.is_file() and p.stat().st_size > 0
        for p in expected
    )


def run_spaceclaim_face(face, shrinkwrap_mm, overwrite=True):
    cmd = [
        sys.executable,
        STL_TO_SCDOC,
        "--face",
        face,
        "--shrinkwrap-mm",
        str(shrinkwrap_mm),
    ]

    if overwrite:
        cmd.append("--overwrite")

    return run(cmd, STL_TO_SCDOC.parent)


def run_cfd_face(face):
    return run(
        [
            sys.executable,
            SCDOC_TO_CFD,
            "--face",
            face,
            "--overwrite",
            "--timeout",
            str(CFD_TIMEOUT_SEC),
        ],
        SCDOC_TO_CFD.parent,
    )


def main():
    # ------------------------------------------------------------------
    # Basic path checks
    # ------------------------------------------------------------------
    for script in (IMAGE_TO_STL, STL_TO_SCDOC, SCDOC_TO_CFD):
        if not script.is_file():
            print("ERROR: script not found:", script)
            return 1

    faces = source_faces()

    if not faces:
        print("ERROR: no face images found in:", IMAGE_DIR)
        return 1

    print("=" * 80)
    print("AI-CFD CFD GENERATION PIPELINE")
    print("Python             :", sys.executable)
    print("Source faces       :", len(faces))
    print("Primary shrinkwrap :", PRIMARY_SHRINKWRAP_MM, "mm")
    print("Fallback shrinkwrap:", FALLBACK_SHRINKWRAP_MM, "mm")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Stage 1: JPG -> STL
    # Only run if at least one source face is missing an STL.
    # image_to_stl.py itself handles the batch.
    #
    # Do not continue if the stage returns non-zero or if any source image
    # still lacks its expected STL after the batch finishes.
    # ------------------------------------------------------------------
    missing_stl = [
        face for face in faces
        if not (STL_DIR / f"{face}.stl").is_file()
    ]

    if missing_stl:
        print(f"\n[STAGE 1] JPG -> STL: {len(missing_stl)} STL(s) missing")

        stage1_rc = run(
            [sys.executable, IMAGE_TO_STL],
            IMAGE_TO_STL.parent,
        )

        missing_stl_after = [
            face for face in faces
            if not (STL_DIR / f"{face}.stl").is_file()
        ]

        if stage1_rc != 0 or missing_stl_after:
            print("\n[STAGE 1 FAILED] JPG -> STL did not complete cleanly")
            print("Return code:", stage1_rc)

            if missing_stl_after:
                print("Missing STL outputs:")
                for face in missing_stl_after:
                    print(" -", face)

            return 1

    else:
        print("\n[STAGE 1] SKIP: all source faces already have STL files")

    # Only the current source-image set is allowed to advance. This avoids
    # accidentally processing stale/orphan face_*.stl files left in STL_DIR.
    stl_faces = [
        face for face in faces
        if (STL_DIR / f"{face}.stl").is_file()
    ]

    # ------------------------------------------------------------------
    # Stage 2: STL -> SCDOC
    # Existing SCDOCs are preserved; missing ones are created with 5 mm.
    #
    # As in Stage 1, require both a zero return code and the presence of every
    # expected output before advancing to Fluent.
    # ------------------------------------------------------------------
    missing_scdoc = [
        face for face in stl_faces
        if not (SCDOC_DIR / f"{face}.scdoc").is_file()
    ]

    if missing_scdoc:
        print(
            f"\n[STAGE 2] STL -> SCDOC: {len(missing_scdoc)} SCDOC(s) missing "
            f"(default {PRIMARY_SHRINKWRAP_MM} mm)"
        )

        # No --overwrite: existing SCDOCs are skipped by the stage script.
        stage2_rc = run(
            [
                sys.executable,
                STL_TO_SCDOC,
                "--shrinkwrap-mm",
                str(PRIMARY_SHRINKWRAP_MM),
            ],
            STL_TO_SCDOC.parent,
        )

        missing_scdoc_after = [
            face for face in stl_faces
            if not (SCDOC_DIR / f"{face}.scdoc").is_file()
        ]

        if stage2_rc != 0 or missing_scdoc_after:
            print("\n[STAGE 2 FAILED] STL -> SCDOC did not complete cleanly")
            print("Return code:", stage2_rc)

            if missing_scdoc_after:
                print("Missing SCDOC outputs:")
                for face in missing_scdoc_after:
                    print(" -", face)

            return 1

    else:
        print("\n[STAGE 2] SKIP: all STL faces already have SCDOC files")

    # Use exactly the validated current source set in Stage 3 instead of
    # scanning every historical face_*.scdoc that may exist in SCDOC_DIR.
    scdocs = [
        SCDOC_DIR / f"{face}.scdoc"
        for face in stl_faces
    ]

    # ------------------------------------------------------------------
    # Stage 3: SCDOC -> CFD
    # ------------------------------------------------------------------
    print(f"\n[STAGE 3] SCDOC -> CFD: {len(scdocs)} case(s) found")

    success = []
    fallback_success = []
    failures = []

    for index, scdoc in enumerate(scdocs, start=1):
        face = scdoc.stem

        print("\n" + "#" * 80)
        print(f"[{index}/{len(scdocs)}] {face}")
        print("#" * 80)

        # Resume-friendly: do not recompute a fully completed case.
        if complete_cfd(face):
            print("[SKIP] all 9 CFD outputs already exist")
            success.append(face)
            continue

        # First CFD attempt: use the existing SCDOC, normally created at 5 mm.
        rc = run_cfd_face(face)

        if rc == 0 and complete_cfd(face):
            print(f"[OK] {face}")
            success.append(face)
            continue

        # A zero return code without all expected outputs is still a failed case.
        if rc == 0:
            print(
                f"[FAILED] {face}: Fluent returned 0 but one or more "
                "expected CFD outputs are missing or empty"
            )
            failures.append(face)
            continue

        # Validated geometry fallback.
        if rc == SELF_INTERSECTION_EXIT_CODE:
            print(
                f"[FALLBACK] {face}: Fluent detected Share Topology "
                f"self-intersection"
            )
            print(
                f"[FALLBACK] rebuilding SCDOC with "
                f"{FALLBACK_SHRINKWRAP_MM} mm shrinkwrap"
            )

            sc_rc = run_spaceclaim_face(
                face,
                FALLBACK_SHRINKWRAP_MM,
                overwrite=True,
            )

            if sc_rc != 0:
                print(f"[FAILED] {face}: fallback SCDOC generation failed")
                failures.append(face)
                continue

            print(f"[FALLBACK] retrying Fluent for {face}")

            retry_rc = run_cfd_face(face)

            if retry_rc == 0 and complete_cfd(face):
                print(
                    f"[OK-FALLBACK] {face}: completed with "
                    f"{FALLBACK_SHRINKWRAP_MM} mm shrinkwrap"
                )
                success.append(face)
                fallback_success.append(face)
                continue

            print(
                f"[FAILED] {face}: fallback CFD retry failed "
                f"(return code {retry_rc})"
            )
            failures.append(face)
            continue

        # Any other Fluent failure: record it and continue.
        print(f"[FAILED] {face}: CFD return code {rc}; continuing")
        failures.append(face)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PIPELINE FINISHED")
    print("SCDOC cases       :", len(scdocs))
    print("Successful cases  :", len(success))
    print("Fallback successes:", len(fallback_success))
    print("Failed cases      :", len(failures))

    if fallback_success:
        print("\n4 mm fallback used successfully:")
        for face in fallback_success:
            print(" -", face)

    if failures:
        print("\nFailed faces:")
        for face in failures:
            print(" -", face)

    print("=" * 80)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
