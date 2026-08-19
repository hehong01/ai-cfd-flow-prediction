#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
master_run.py

Simple end-to-end wrapper for the AI-CFD pipeline.

Stages
------
1) image_to_stl.py
2) stl_to_scdoc.py          (default shrinkwrap: 5 mm)
3) scdoc_to_cfd.py          (5, 8, 10 m/s)

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
Run this master with the Python environment used by image_to_stl.py
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
    print("AI-CFD MASTER PIPELINE")
    print("Python             :", sys.executable)
    print("Source faces       :", len(faces))
    print("Primary shrinkwrap :", PRIMARY_SHRINKWRAP_MM, "mm")
    print("Fallback shrinkwrap:", FALLBACK_SHRINKWRAP_MM, "mm")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Stage 1: JPG -> STL
    # Only run if at least one source face is missing an STL.
    # image_to_stl.py itself handles the batch.
    # ------------------------------------------------------------------
    missing_stl = [
        face for face in faces
        if not (STL_DIR / f"{face}.stl").is_file()
    ]

    if missing_stl:
        print(f"\n[STAGE 1] JPG -> STL: {len(missing_stl)} STL(s) missing")
        run([sys.executable, IMAGE_TO_STL], IMAGE_TO_STL.parent)
    else:
        print("\n[STAGE 1] SKIP: all source faces already have STL files")

    # ------------------------------------------------------------------
    # Stage 2: STL -> SCDOC
    # Existing SCDOCs are preserved; missing ones are created with 5 mm.
    # ------------------------------------------------------------------
    stl_faces = sorted(p.stem for p in STL_DIR.glob("face_*.stl"))

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
        run(
            [
                sys.executable,
                STL_TO_SCDOC,
                "--shrinkwrap-mm",
                str(PRIMARY_SHRINKWRAP_MM),
            ],
            STL_TO_SCDOC.parent,
        )
    else:
        print("\n[STAGE 2] SKIP: all STL faces already have SCDOC files")

    # ------------------------------------------------------------------
    # Stage 3: SCDOC -> CFD
    # ------------------------------------------------------------------
    scdocs = sorted(SCDOC_DIR.glob("face_*.scdoc"))

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
