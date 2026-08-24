# -*- coding: utf-8 -*-
"""
STL -> SCDOC SpaceClaim 2021 R1 automation.

Run this file with normal Python:

    cd C:\ai-cfd-flow-prediction\github
    python .\stl_to_scdoc.py --face face_0002

It automatically scans:
    ai-cfd-data\02_stl\*.stl

and writes:
    ai-cfd-data\03_spaceclaim\*.scdoc

No per-face manual repetition is required.
"""

from __future__ import print_function

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import STL_DIR, SPACECLAIM_DIR

OUTPUT_DIR = SPACECLAIM_DIR
LOG_DIR = OUTPUT_DIR / "logs"

DEFAULT_SHRINKWRAP_MM = 5.0
SIDE_MM = 500.0
TOP_BOTTOM_MM = 500.0
FRONT_MM = 500.0
BACK_MM = 1500.0


def find_spaceclaim_exe(explicit=None):
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError("SpaceClaim.exe not found: {}".format(p))

    candidates = [
        Path(r"C:\Program Files\ANSYS Inc\v211\scdm\SpaceClaim.exe"),
        Path(r"C:\Program Files\ANSYS Inc\v211\scdm\bin\winx64\SpaceClaim.exe"),
    ]

    for p in candidates:
        if p.is_file():
            return p

    patterns = [
        r"C:\Program Files\ANSYS Inc\v*\scdm\SpaceClaim.exe",
        r"C:\Program Files\ANSYS Inc\v*\scdm\bin\winx64\SpaceClaim.exe",
        r"C:\Program Files\SpaceClaim*\SpaceClaim.exe",
    ]

    found = []
    for pattern in patterns:
        found.extend(Path(p) for p in glob.glob(pattern))

    found = sorted({p.resolve() for p in found if p.is_file()}, reverse=True)

    if found:
        return found[0]

    raise FileNotFoundError(
        "SpaceClaim.exe was not found automatically. "
        "Use --spaceclaim \"C:\\...\\SpaceClaim.exe\""
    )


WORKER_TEMPLATE = r"""# -*- coding: utf-8 -*-
# Generated automatically by stl_to_scdoc.py
# Runs inside SpaceClaim Script API / IronPython.

import os

INPUT_STL = __INPUT_STL__
OUTPUT_SCDOC = __OUTPUT_SCDOC__

SHRINKWRAP_MM = __SHRINKWRAP_MM__
SIDE_MM = __SIDE_MM__
TOP_BOTTOM_MM = __TOP_BOTTOM_MM__
FRONT_MM = __FRONT_MM__
BACK_MM = __BACK_MM__


def all_meshes():
    result = list(GetRootPart().Meshes)
    for comp in GetRootPart().Components:
        try:
            result.extend(comp.Content.Meshes)
        except:
            pass
    return result


def all_solids():
    result = list(GetRootPart().Bodies)
    for comp in GetRootPart().Components:
        try:
            result.extend(comp.Content.Bodies)
        except:
            pass
    return result


def center_xyz(face):
    # EvalMid().Point is more reliable across older SpaceClaim APIs
    # than relying on face.Box.Center.
    p = face.EvalMid().Point
    return float(p.X), float(p.Y), float(p.Z)


def make_selection(entity_or_entities):
    try:
        return FaceSelection.Create(entity_or_entities)
    except:
        return Selection.Create(entity_or_entities)


def create_named(selection, old_name, new_name):
    NamedSelection.Create(selection, Selection.Empty())

    # SpaceClaim localizes default Named Selection names.
    # English UI: Group1, Group2, ...
    # Korean UI : 그룹1, 그룹2, ...
    #
    # Rename all plausible default-name variants. Rename() is harmless
    # when a candidate name does not exist in this SpaceClaim session.
    index_text = old_name.replace("Group", "")
    candidates = [
        old_name,
        "Group " + index_text,
        u"그룹" + index_text,
        u"그룹 " + index_text,
    ]

    for candidate in candidates:
        try:
            NamedSelection.Rename(candidate, new_name)
        except:
            pass

    print("Named Selection requested: " + new_name)


def save_document(path):
    try:
        DocumentSave.Execute(path)
        return
    except Exception as e1:
        print("Save attempt 1 failed: " + str(e1))

    try:
        DocumentSave.Execute(path, None)
        return
    except Exception as e2:
        print("Save attempt 2 failed: " + str(e2))

    try:
        GetActiveWindow().Document.SaveAs(path)
        return
    except Exception as e3:
        print("Save attempt 3 failed: " + str(e3))

    raise Exception("Could not save SCDOC automatically.")


def main():
    print("============================================================")
    print("Input : " + INPUT_STL)
    print("Output: " + OUTPUT_SCDOC)
    print("============================================================")

    if not os.path.isfile(INPUT_STL):
        raise Exception("Input STL not found: " + INPUT_STL)

    out_dir = os.path.dirname(OUTPUT_SCDOC)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # 1. Import STL
    before_meshes = all_meshes()
    DocumentInsert.Execute(INPUT_STL)
    meshes = all_meshes()

    if len(meshes) <= len(before_meshes):
        raise Exception("STL import did not create a mesh body.")

    face_mesh = meshes[-1]
    print("[1] STL imported")

    # 2. Imported face mesh -> temporary solid
    before_solids = all_solids()

    ConvertToSolid.Execute(
        Selection.Create(face_mesh),
        True,
        None
    )

    solids = all_solids()
    if len(solids) <= len(before_solids):
        raise Exception("Face ConvertToSolid failed.")
    face_solid = solids[-1]
    print("[2] Temporary face solid created")

    # 3. Shrinkwrap using the requested facet size
    before_meshes = all_meshes()

    sw_options = ShrinkwrapOptions()
    sw_options.MaxSize = MM(SHRINKWRAP_MM)
    sw_options.MinSize = MM(SHRINKWRAP_MM)

    Shrinkwrap.Create(
        BodySelection.Create(face_solid),
        sw_options
    )

    meshes = all_meshes()
    if len(meshes) <= len(before_meshes):
        raise Exception("Shrinkwrap failed.")

    shrinkwrapped_face = meshes[-1]

    try:
        Delete.Execute(Selection.Create(face_solid))
    except:
        pass

    print("[3] Shrinkwrap created")

    # 4. CFD enclosure
    # BoxEnclosureCushion order preserved from the original recorded script:
    # X-, X+, Y-, Y+, Z-, Z+
    # +Z = face/front/inlet, -Z = back/wake/outlet
    before_meshes = all_meshes()

    enc_options = EnclosureOptions()
    enc_options.EnclosureType = EnclosureType.Box

    enc_options.EnclosureCushion = BoxEnclosureCushion(
        MM(SIDE_MM),
        MM(SIDE_MM),
        MM(TOP_BOTTOM_MM),
        MM(TOP_BOTTOM_MM),
        MM(BACK_MM),
        MM(FRONT_MM)
    )

    Enclosure.Create(
        Selection.Create(shrinkwrapped_face),
        enc_options
    )

    meshes = all_meshes()
    if len(meshes) <= len(before_meshes):
        raise Exception("Enclosure creation failed.")

    enclosure_mesh = meshes[-1]
    print("[4] Enclosure created")

    # 5. Enclosure mesh -> solid with merged planar faces
    before_solids = all_solids()

    ConvertToSolid.Execute(
        Selection.Create(enclosure_mesh),
        True,
        None
    )

    solids = all_solids()
    if len(solids) <= len(before_solids):
        raise Exception("Enclosure ConvertToSolid failed.")

    enclosure_solid = solids[-1]
    enclosure_faces = list(enclosure_solid.Faces)

    if len(enclosure_faces) < 6:
        raise Exception(
            "Expected at least 6 enclosure faces, got %d."
            % len(enclosure_faces)
        )

    print("[5] Enclosure converted to solid")

    # 6. Identify outer boundary faces by face midpoint coordinates.
    #
    # The enclosure may contain extra/internal faces. We only need the six
    # axis-aligned OUTER faces, so choose the geometric extrema in +/-X,+/-Y,+/-Z.
    data = []

    for i in range(len(enclosure_faces)):
        f = enclosure_faces[i]
        x, y, z = center_xyz(f)
        data.append([i, f, x, y, z])
        print(
            "    face[%d] midpoint = (%.6f, %.6f, %.6f)"
            % (i, x, y, z)
        )

    if len(data) == 0:
        raise Exception("No enclosure faces available for boundary classification.")

    inlet = data[0]
    outlet = data[0]
    x_plus = data[0]
    x_minus = data[0]
    y_plus = data[0]
    y_minus = data[0]

    for item in data[1:]:
        if item[4] > inlet[4]:
            inlet = item
        if item[4] < outlet[4]:
            outlet = item

        if item[2] > x_plus[2]:
            x_plus = item
        if item[2] < x_minus[2]:
            x_minus = item

        if item[3] > y_plus[3]:
            y_plus = item
        if item[3] < y_minus[3]:
            y_minus = item

    inlet_face = inlet[1]
    outlet_face = outlet[1]

    # Store face indices instead of comparing .NET face objects directly.
    farfield_items = [x_plus, x_minus, y_plus, y_minus]
    farfield_indices = []
    farfield_faces = []

    for item in farfield_items:
        idx = item[0]
        if idx not in farfield_indices:
            farfield_indices.append(idx)
            farfield_faces.append(item[1])

    print("    inlet index  = %d, Z = %.6f" % (inlet[0], inlet[4]))
    print("    outlet index = %d, Z = %.6f" % (outlet[0], outlet[4]))
    print("    farfield indices = " + str(farfield_indices))

    if len(farfield_faces) != 4:
        raise Exception(
            "Boundary classification failed: expected 4 unique farfield faces, got %d."
            % len(farfield_faces)
        )

    print("[6] Boundary faces classified")

    # 7. Named Selections
    #
    # IMPORTANT:
    # In this SpaceClaim 2021 R1 environment, after the default first group
    # is renamed, the next newly created Named Selection reuses that same
    # default first-group name.
    #
    # Therefore use the EXACT same rename logic that already succeeded for
    # "wall", and repeat it sequentially for every boundary.
    wall_selection = Selection.Create(shrinkwrapped_face)

    create_named(wall_selection, "Group1", "wall")
    create_named(make_selection(inlet_face), "Group1", "inlet")
    create_named(make_selection(outlet_face), "Group1", "outlet")
    create_named(make_selection(farfield_faces), "Group1", "farfield")

    # Do not suppress the shrinkwrap body.
    # This matches the manually validated reconstructed state.

    print("[7] Named Selections created")

    # 8. Save
    save_document(OUTPUT_SCDOC)
    print("[8] Saved: " + OUTPUT_SCDOC)
    print("DONE")


try:
    main()
except Exception as e:
    print("============================================================")
    print("WORKER FAILED")
    print(str(e))
    try:
        import traceback
        traceback.print_exc()
    except:
        pass
    print("============================================================")
    raise
"""


def build_worker(input_stl, output_scdoc, shrinkwrap_mm):
    text = WORKER_TEMPLATE

    replacements = {
        "__INPUT_STL__": repr(str(input_stl)),
        "__OUTPUT_SCDOC__": repr(str(output_scdoc)),
        "__SHRINKWRAP_MM__": repr(float(shrinkwrap_mm)),
        "__SIDE_MM__": repr(float(SIDE_MM)),
        "__TOP_BOTTOM_MM__": repr(float(TOP_BOTTOM_MM)),
        "__FRONT_MM__": repr(float(FRONT_MM)),
        "__BACK_MM__": repr(float(BACK_MM)),
    }

    for key, value in replacements.items():
        text = text.replace(key, value)

    fd, temp_path = tempfile.mkstemp(
        prefix="ai_cfd_spaceclaim_",
        suffix=".py",
        text=True,
    )
    os.close(fd)

    Path(temp_path).write_text(text, encoding="utf-8")
    return Path(temp_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert every STL in ai-cfd-data/02_stl "
            "to a SpaceClaim SCDOC automatically."
        )
    )
    parser.add_argument(
        "--spaceclaim",
        default=None,
        help="Optional explicit path to SpaceClaim.exe.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate SCDOC files that already exist.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Run SpaceClaim with UI instead of headless mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N STL files.",
    )
    parser.add_argument(
        "--face",
        default=None,
        help=(
            "Process only one face ID or STL filename, "
            "for example: --face face_0002"
        ),
    )
    parser.add_argument(
        "--shrinkwrap-mm",
        type=float,
        default=DEFAULT_SHRINKWRAP_MM,
        help=(
            "SpaceClaim shrinkwrap facet size in mm. "
            "Default: %(default)s. "
            "Use 4.0 as the validated fallback for self-intersection cases."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.shrinkwrap_mm <= 0:
        print("--shrinkwrap-mm must be greater than 0.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not STL_DIR.is_dir():
        raise FileNotFoundError("STL directory not found: {}".format(STL_DIR))

    stl_files = sorted(STL_DIR.glob("*.stl"))

    if not stl_files:
        print("No STL files found in:")
        print(STL_DIR)
        return 1

    if args.face:
        requested = args.face
        if not requested.lower().endswith(".stl"):
            requested += ".stl"

        stl_files = [
            p for p in stl_files
            if p.name.lower() == requested.lower()
        ]

        if not stl_files:
            print("Requested STL not found:", requested)
            return 1

    if args.limit is not None:
        stl_files = stl_files[:max(args.limit, 0)]

    spaceclaim_exe = find_spaceclaim_exe(args.spaceclaim)

    print("=" * 72)
    print("AI-CFD SpaceClaim batch automation")
    print("SpaceClaim :", spaceclaim_exe)
    print("STL input  :", STL_DIR)
    print("SCDOC out  :", OUTPUT_DIR)
    print("Shrinkwrap :", "{} mm".format(args.shrinkwrap_mm))
    print("Files      :", len(stl_files))
    print("=" * 72)

    success = 0
    skipped = 0
    failed = []

    for index, stl_path in enumerate(stl_files, start=1):
        output_path = OUTPUT_DIR / (stl_path.stem + ".scdoc")
        log_path = LOG_DIR / (stl_path.stem + ".log")

        print()
        print("[{}/{}] {}".format(index, len(stl_files), stl_path.name))

        if output_path.exists() and not args.overwrite:
            print("  SKIP: already exists -> {}".format(output_path.name))
            skipped += 1
            continue

        worker_path = build_worker(stl_path, output_path, args.shrinkwrap_mm)

        cmd = [
            str(spaceclaim_exe),
            "/RunScript={}".format(worker_path),
            "/ExitAfterScript=True",
            "/ScriptOutput={}".format(log_path),
        ]

        if not args.visible:
            cmd.append("/Headless=True")

        try:
            completed = subprocess.run(cmd, check=False)
        finally:
            try:
                worker_path.unlink()
            except OSError:
                pass

        if (
            completed.returncode == 0
            and output_path.is_file()
            and output_path.stat().st_size > 0
        ):
            print("  OK -> {}".format(output_path.name))
            success += 1
        else:
            print("  FAILED")
            print("  Return code:", completed.returncode)
            print("  Log:", log_path)

            # Show SpaceClaim's exact worker output in PowerShell.
            try:
                if log_path.exists():
                    print("----- SpaceClaim worker log -----")
                    print(log_path.read_text(encoding="utf-8", errors="replace"))
                    print("---------------------------------")
            except Exception as log_exc:
                print("Could not read worker log:", log_exc)

            failed.append(stl_path.name)

    print()
    print("=" * 72)
    print("Finished")
    print("  success :", success)
    print("  skipped :", skipped)
    print("  failed  :", len(failed))

    if failed:
        print("Failed files:")
        for name in failed:
            print("  -", name)
        print("Logs:", LOG_DIR)
        return 2

    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
