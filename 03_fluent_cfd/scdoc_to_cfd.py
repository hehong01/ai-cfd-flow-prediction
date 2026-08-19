#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scdoc_to_cfd.py  --  ANSYS Fluent 2021 R1 (v211)

Runs the recorded face_0003 Fluent workflow to turn one .scdoc into
CAS/DAT + CSV at 5, 8 and 10 m/s.

This file is SELF-CONTAINED: the Fluent journal is embedded at the bottom of
this script, so no .jou file is needed. The embedded journal is the recording
of the successful manual face_0003 run, with exactly two changes:

  1. The 5 m/s CSV was exported twice in the recording. The first export had
     x/y/z selected as physical quantities, which duplicated the coordinate
     columns in the CSV. That first export is removed, so the quantity
     selection is (0 35 60 66 71 72) and each speed produces exactly one CSV.
  2. The 10 m/s Case & Data name ".cas" was made ".cas.h5", matching the
     5 and 8 m/s lines in the same recording.
  3. Immediately after Update Regions, all face zones carrying the SpaceClaim
     label "wall" are merged into one face zone and normalized to the validated
     face_0003 name/id (wall-zone------------:2030, id 2030). This removes the
     geometry-dependent 1-face wall-zone artifact without deleting any mesh face.

Everything else remains from the validated face_0003 recording. This script does not
re-implement any Fluent setting; it only substitutes file paths, launches
Fluent, and checks the resulting files.

Usage (PowerShell):
    python .\\scdoc_to_cfd.py --face face_0003 --dry-run
    python .\\scdoc_to_cfd.py --face face_0003 --overwrite
    python .\\scdoc_to_cfd.py --face face_0003 --check-only

Exit codes:
    0  success
    1  general failure
    42 Fluent Share Topology self-intersection; geometry fallback recommended
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import DATA_ROOT

FLUENT_EXE = Path(r"C:\Program Files\ANSYS Inc\v211\fluent\ntbin\win64\fluent.exe")

SCDOC_DIR = "03_spaceclaim"
CAS_DIR = "04_fluent"
CSV_DIR = "05_cfd_csv"
TAGS = ["05mps", "08mps", "10mps"]

SELF_INTERSECTION_EXIT_CODE = 42
SELF_INTERSECTION_MARKER = "self-intersecting triangles"
SHARE_TOPOLOGY_MARKER = "apply share topology"

EXPECTED_CSV_HEADER = [
    "nodenumber", "x-coordinate", "y-coordinate", "z-coordinate",
    "pressure", "temperature", "y-plus", "wall-shear",
    "heat-flux", "heat-transfer-coef",
]

DIALOG_RE = re.compile(
    r'^\(cx-gui-do\s+cx-set-file-dialog-entries\s+"Select\s+File"\s+'
    r"'\(\s*\"(?P<path>[^\"]*)\"\)\s+\"(?P<filter>.*)\"\)\s*$"
)
SCDOC_RE = re.compile(r"(r'FileName': r')(?P<path>[^']*)(')")


def log(tag, msg):
    print("[" + tag + "] " + str(msg), flush=True)


def die(msg):
    print("[FAIL] " + str(msg), file=sys.stderr, flush=True)
    sys.exit(1)


def fwd(p):
    return str(p).replace("\\", "/")


def targets(face, data_root):
    t = {}
    for tag in TAGS:
        t["cas_" + tag] = data_root / CAS_DIR / (face + "_" + tag + ".cas.h5")
        t["dat_" + tag] = data_root / CAS_DIR / (face + "_" + tag + ".dat.h5")
        t["csv_" + tag] = data_root / CSV_DIR / (face + "_" + tag + ".csv")
    return t


def prepare_journal(face, data_root, auto_exit, journal_text):
    lines = journal_text.replace("\r\n", "\n").split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    notes = []

    hits = [i for i, l in enumerate(lines) if "'Import Geometry'" in l and "FileName" in l]
    if len(hits) != 1:
        die("expected 1 Import Geometry line in the journal, found %d" % len(hits))
    i = hits[0]
    m = SCDOC_RE.search(lines[i])
    if not m:
        die("could not read the .scdoc path on journal line %d" % (i + 1))
    new_scdoc = fwd(data_root / SCDOC_DIR / (face + ".scdoc"))
    if m.group("path") != new_scdoc:
        lines[i] = lines[i][:m.start("path")] + new_scdoc + lines[i][m.end("path"):]
        notes.append("line %d: scdoc -> %s" % (i + 1, new_scdoc))

    dialogs = []
    for i, l in enumerate(lines):
        m2 = DIALOG_RE.match(l)
        if m2:
            dialogs.append((i, m2))
    if len(dialogs) != 6:
        die("expected 6 output file dialogs in the journal (3 Case&Data + 3 CSV), found %d"
            % len(dialogs))

    tgt = targets(face, data_root)
    expect = [("cas", "05mps"), ("csv", "05mps"), ("cas", "08mps"),
              ("csv", "08mps"), ("cas", "10mps"), ("csv", "10mps")]
    for (i, m2), (kind, tag) in zip(dialogs, expect):
        old = m2.group("path")
        if tag not in old.lower():
            die("journal line %d: expected a %s dialog, found %s" % (i + 1, tag, old))
        new = fwd(tgt[kind + "_" + tag])
        if new != old:
            lines[i] = ('(cx-gui-do cx-set-file-dialog-entries "Select File" '
                        "'( \"" + new + "\") \"" + m2.group("filter") + "\")")
            notes.append("line %d: %s %s -> %s" % (i + 1, kind, tag, new))

    header = [
        "; ---------------------------------------------------------",
        "; generated by scdoc_to_cfd.py " + datetime.now().isoformat(timespec="seconds"),
        "; face: " + face,
        "; ---------------------------------------------------------",
    ]
    tail = []
    if auto_exit:
        tail = ["", "; appended so this script can check the outputs", "/exit", "yes"]
    return header + lines + tail, notes


def check_binary(path):
    if not path.exists():
        return False, "missing"
    n = path.stat().st_size
    if n == 0:
        return False, "0 bytes"
    return True, "{:,} bytes".format(n)


def check_csv(path):
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "0 bytes"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first = f.readline()
            rows = sum(1 for _ in f)
    except OSError as exc:
        return False, "unreadable: %s" % exc
    got = [c.strip().strip('"').strip().lower() for c in first.split(",") if c.strip()]
    if got == EXPECTED_CSV_HEADER:
        if rows:
            return True, "10 columns OK, %d rows" % rows
        return False, "header only, no data rows"
    dups = [c for c in ("x-coordinate", "y-coordinate", "z-coordinate") if got.count(c) > 1]
    if dups:
        return False, ("x/y/z appear twice %s -- coordinates were selected as quantities "
                       "in the Export panel (%d columns)" % (dups, len(got)))
    return False, "header mismatch: got %s" % got


def validate(face, data_root):
    tgt = targets(face, data_root)
    ok = True
    d = data_root / CAS_DIR
    found = sorted(p.name for p in d.glob(face + "*")) if d.is_dir() else []
    log("CHECK", CAS_DIR + ": " + str(found if found else "(nothing for this face)"))
    for tag in TAGS:
        for kind in ("cas", "dat"):
            p = tgt[kind + "_" + tag]
            good, detail = check_binary(p)
            ok = ok and good
            log("CHECK", "%s %s %s %s: %s"
                % (tag, kind.upper(), "OK " if good else "BAD", p.name, detail))
        p = tgt["csv_" + tag]
        good, detail = check_csv(p)
        ok = ok and good
        log("CSV OK" if good else "CHECK", "%s %s: %s" % (tag, p.name, detail))
    return ok



def latest_transcript(run_dir):
    trns = sorted(run_dir.glob("*.trn"), key=lambda p: p.stat().st_mtime)
    return trns[-1] if trns else None


def has_share_topology_self_intersection(run_dir):
    """Return True only for the validated Share Topology self-intersection failure."""
    trn = latest_transcript(run_dir)
    if trn is None:
        return False

    try:
        text = trn.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False

    return (
        SELF_INTERSECTION_MARKER in text
        and SHARE_TOPOLOGY_MARKER in text
    )


def terminate_fluent_for_face(face):
    """Best-effort cleanup of only the failed face's Fluent GUI/process tree."""
    if sys.platform != "win32":
        return

    ps = (
        "$p = Get-Process | Where-Object { "
        "$_.MainWindowTitle -like '" + face + "*Fluent*' }; "
        "$p | ForEach-Object { "
        "taskkill /PID $_.Id /T /F | Out-Null "
        "}"
    )

    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        pass


def wait_for_outputs(tgt, timeout, run_dir, poll=15):
    """Wait for the 6 output files to appear and stop changing.

    On Windows fluent.exe is a launcher: it starts the real Fluent process and
    returns straight away, so the launcher's exit code says nothing about the
    run. We therefore watch the output folder instead. Progress printed here is
    real (a file appeared / grew), never invented.
    """
    order = []
    for tag in TAGS:
        order += ["cas_" + tag, "dat_" + tag, "csv_" + tag]
    seen = set()
    t0 = time.time()
    stable_since = None
    last_sizes = None
    log("WAIT", "watching %s and %s for output files and Fluent transcript "
                "(Ctrl-C to stop waiting; Fluent keeps running)" % (CAS_DIR, CSV_DIR))
    while time.time() - t0 < timeout:
        if has_share_topology_self_intersection(run_dir):
            log(
                "GEOMETRY",
                "Share Topology self-intersection detected in Fluent transcript.",
            )
            return "self_intersection"

        present = set(k for k in order if tgt[k].exists())
        for k in order:
            if k in present and k not in seen:
                log("PROGRESS", "%s written  (%.1f min elapsed)"
                    % (tgt[k].name, (time.time() - t0) / 60.0))
        seen = present
        if len(present) == len(order):
            sizes = tuple(tgt[k].stat().st_size for k in order)
            if last_sizes is not None and sizes == last_sizes:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= 30:
                    log("WAIT", "all %d outputs present and no longer changing" % len(order))
                    return "finished"
            else:
                stable_since = None
            last_sizes = sizes
        time.sleep(poll)
    log("WAIT", "gave up after %.1f min (--timeout)" % (timeout / 60.0))
    return "timeout"


def diagnose_transcript(run_dir, max_report=10):
    """Read the .trn transcript Fluent writes into the run folder and report the
    exact journal command that failed.

    Fluent echoes each journal command as a line starting with "> ", then prints
    any error right after it. So the last "> " line before an error line IS the
    command that failed. Nothing here is guessed.
    """
    trns = sorted(run_dir.glob("*.trn"), key=lambda p: p.stat().st_mtime)
    if not trns:
        log("DIAG", "no .trn transcript found in %s" % run_dir)
        return []
    trn = trns[-1]
    try:
        lines = trn.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        log("DIAG", "could not read %s: %s" % (trn, exc))
        return []

    markers = ("cannot find widget", "invalid command", "Error:", "Error Object:")
    problems = []
    last_cmd = "(none yet)"
    for line in lines:
        s = line.rstrip()
        if s.startswith("> "):
            cmd = s[2:].strip()
            if cmd:
                last_cmd = cmd
            continue
        if any(m in s for m in markers):
            if not problems or problems[-1][0] != last_cmd:
                problems.append((last_cmd, [s.strip()]))
            else:
                problems[-1][1].append(s.strip())

    log("DIAG", "transcript: %s" % trn)
    if not problems:
        log("DIAG", "no journal errors found in the transcript")
        return []
    log("DIAG", "%d journal error(s) found; the failing command is the line above each:"
        % len(problems))
    for cmd, msgs in problems[:max_report]:
        log("DIAG", "  command : %s" % cmd)
        for m in msgs[:4]:
            log("DIAG", "     -> %s" % m)
    if len(problems) > max_report:
        log("DIAG", "  ... and %d more (see the transcript)" % (len(problems) - max_report))
    return problems


def main():
    ap = argparse.ArgumentParser(
        description="Run the recorded Fluent 2021 R1 workflow for one face "
                    "and produce CAS/DAT + CSV at 5, 8 and 10 m/s.")
    ap.add_argument("--face", default="face_0003", help="default: %(default)s")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT,
                    help="folder holding 03_spaceclaim / 04_fluent / 05_cfd_csv")
    ap.add_argument("--fluent-exe", type=Path, default=FLUENT_EXE)
    ap.add_argument("--journal", type=Path, default=None,
                    help="use an external .jou instead of the embedded journal")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true",
                    help="delete existing outputs for this face before running")
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare and report everything, but do not launch Fluent")
    ap.add_argument("--check-only", action="store_true",
                    help="only validate outputs that already exist")
    ap.add_argument("--no-auto-exit", dest="auto_exit", action="store_false",
                    help="leave Fluent open when the journal finishes")
    ap.add_argument("--no-wait", action="store_true",
                    help="launch Fluent and exit immediately instead of waiting for "
                         "the output files (check later with --check-only)")
    ap.add_argument("--diagnose", action="store_true",
                    help="read the newest run's Fluent .trn transcript and print which "
                         "journal command failed, then exit")
    ap.add_argument("--save-journal", type=Path, default=None,
                    help="write the embedded journal to this path and exit")
    ap.set_defaults(auto_exit=True)
    args = ap.parse_args()

    if args.save_journal:
        with args.save_journal.open("w", encoding="utf-8", newline="\r\n") as f:
            f.write(EMBEDDED_JOURNAL.replace("\r\n", "\n").rstrip("\n") + "\n")
        log("DONE", "embedded journal written to %s" % args.save_journal.resolve())
        return 0

    here = Path(__file__).resolve().parent

    if args.diagnose:
        runs = sorted((here / "_runs").glob("*"), key=lambda p: p.stat().st_mtime)
        runs = [p for p in runs if p.is_dir()]
        if not runs:
            die("no _runs folder to diagnose yet")
        log("DIAG", "newest run: %s" % runs[-1])
        diagnose_transcript(runs[-1])
        return 0

    face = args.face
    data_root = args.data_root
    log("INPUT", "face      = " + face)
    log("INPUT", "data root = " + str(data_root))

    if args.check_only:
        return 0 if validate(face, data_root) else 1

    scdoc = data_root / SCDOC_DIR / (face + ".scdoc")
    if not scdoc.is_file():
        die("input not found: %s" % scdoc)
    log("INPUT", "scdoc     = %s ({:,} bytes)".format(scdoc.stat().st_size) % scdoc)

    if args.journal:
        if not args.journal.is_file():
            die("--journal not found: %s" % args.journal)
        journal_text = args.journal.read_text(encoding="utf-8", errors="replace")
        log("JOURNAL", "source    = " + str(args.journal))
    else:
        journal_text = EMBEDDED_JOURNAL
        log("JOURNAL", "source    = embedded in this script")

    if not args.dry_run and not args.fluent_exe.is_file():
        die("fluent.exe not found: %s" % args.fluent_exe)

    lines, notes = prepare_journal(face, data_root, args.auto_exit, journal_text)

    run_dir = here / "_runs" / (face + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    gen = run_dir / (face + ".jou")
    with gen.open("w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    log("JOURNAL", "written   = %s  (%d lines)" % (gen, len(lines)))
    for n in notes:
        log("JOURNAL", "  " + n)

    (data_root / CAS_DIR).mkdir(parents=True, exist_ok=True)
    (data_root / CSV_DIR).mkdir(parents=True, exist_ok=True)
    tgt = targets(face, data_root)
    existing = [p for p in tgt.values() if p.exists()]
    if existing:
        if not args.overwrite:
            die("these outputs already exist. Fluent would pop an overwrite confirmation "
                "dialog and stall the journal, so pass --overwrite to delete them first:\n        "
                + "\n        ".join(str(p) for p in existing))
        for p in existing:
            if args.dry_run:
                log("OVERWRITE", "would delete " + str(p))
            else:
                p.unlink()
                log("OVERWRITE", "deleted " + str(p))

    cmd = [str(args.fluent_exe), "3ddp", "-meshing", "-t%d" % args.nproc, "-i", str(gen)]
    pretty = " ".join(('"' + c + '"') if " " in c else c for c in cmd)
    if args.dry_run:
        log("FLUENT", "would run: " + pretty)
        log("DONE", "--dry-run finished. Fluent was not launched.")
        return 0

    log("FLUENT", "launching -- a Fluent window will open (the GUI is required for this journal)")
    log("FLUENT", pretty)
    logfile = run_dir / "fluent_stdout.log"
    t0 = time.time()
    try:
        with logfile.open("wb") as fh:
            proc = subprocess.Popen(cmd, cwd=str(run_dir), stdout=fh,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            rc = proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        die("the Fluent launcher never returned after %ds." % args.timeout)
    except OSError as exc:
        die("could not launch Fluent: %s" % exc)

    launcher_sec = time.time() - t0
    log("FLUENT", "launcher returned %d after %.1f s" % (rc, launcher_sec))
    log("FLUENT", "log = %s  (usually near-empty on Windows; the Fluent console "
                  "window is the real transcript)" % logfile)

    if rc != 0:
        die("the Fluent launcher exited with code %d.\n"
            "        journal: %s\n        log    : %s" % (rc, gen, logfile))

    if args.no_wait:
        log("DONE", "Fluent launched. When it finishes, run this to check the results:\n"
                    "       python scdoc_to_cfd.py --face %s --check-only" % face)
        return 0

    # On Windows fluent.exe is only a launcher, so its exit tells us nothing about
    # the actual run. Watch the output files instead.
    try:
        wait_state = wait_for_outputs(tgt, args.timeout, run_dir)
    except KeyboardInterrupt:
        log("WAIT", "stopped waiting. Fluent is still running; use --check-only later.")
        return 0

    if wait_state == "self_intersection":
        diagnose_transcript(run_dir)
        log(
            "FALLBACK",
            "returning code %d so the master can rebuild %s with the "
            "validated finer shrinkwrap and retry."
            % (SELF_INTERSECTION_EXIT_CODE, face),
        )
        terminate_fluent_for_face(face)
        return SELF_INTERSECTION_EXIT_CODE

    if wait_state != "finished":
        diagnose_transcript(run_dir)
        die("timed out waiting for the output files. Fluent may still be running.\n"
            "        Check the Fluent window, then run --check-only.\n"
            "        journal: %s" % gen)

    if not args.auto_exit:
        log("DONE", "Fluent left open. Run again with --check-only to validate the outputs.")
        return 0

    if not validate(face, data_root):
        diagnose_transcript(run_dir)
        die("the outputs did not validate (see [CHECK] and [DIAG] above).\n"
            "        journal: %s\n        log    : %s" % (gen, logfile))

    log("DONE", face + ": CAS/DAT and CSV for 5, 8 and 10 m/s all present and valid.")
    return 0


# =====================================================================
# The recorded Fluent journal, embedded verbatim (see the notes at the top
# of this file for the two changes made to the recording).
# =====================================================================

EMBEDDED_JOURNAL = """\
/file/set-tui-version "21.1"
(%py-exec "workflow.InitializeWorkflow(WorkflowType=r'Watertight Geometry')")
(%py-exec "workflow.TaskObject['Import Geometry'].Arguments.setState({r'FileName': r'C:/ai-cfd-flow-prediction/ai-cfd-data/03_spaceclaim/face_0003.scdoc',})")
(%py-exec "workflow.TaskObject['Import Geometry'].Execute()")
(%py-exec "workflow.TaskObject['Add Local Sizing'].Execute()")
(%py-exec "workflow.TaskObject['Generate the Surface Mesh'].Arguments.setState({r'CFDSurfaceMeshControls': {r'MaxSize': 54.47142,r'MinSize': 2.12779,},r'SurfaceMeshPreferences': {r'ShowSurfaceMeshPreferences': False,},})")
(%py-exec "workflow.TaskObject['Generate the Surface Mesh'].Execute()")
(%py-exec "workflow.TaskObject['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=False)")
(%py-exec "workflow.TaskObject['Generate the Surface Mesh'].InsertNextTask(CommandName=r'SurfaceMeshImprove')")
(%py-exec "workflow.TaskObject['Improve Surface Mesh'].Arguments.setState({r'SMImprovePreferences': {r'ShowSMImprovePreferences': True,},})")
(%py-exec "workflow.TaskObject['Improve Surface Mesh'].Execute()")
(%py-exec "workflow.TaskObject['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=False)")
(%py-exec "workflow.TaskObject['Describe Geometry'].Arguments.setState({r'CappingRequired': r'No',})")
(%py-exec "workflow.TaskObject['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=False)")
(%py-exec "workflow.TaskObject['Describe Geometry'].Execute()")
(%py-exec "workflow.TaskObject['Apply Share Topology'].Arguments.setState({r'ShareTopologyPreferences': {r'ShowShareTopologyPreferences': True,},})")
(%py-exec "workflow.TaskObject['Apply Share Topology'].Execute()")
(%py-exec "workflow.TaskObject['Create Regions'].Execute()")
(%py-exec "workflow.TaskObject['Update Regions'].Arguments.setState({r'OldRegionNameList': [r'wall', r'zone------------'],r'OldRegionTypeList': [r'solid', r'solid'],r'RegionNameList': [r'wall', r'zone------------'],r'RegionTypeList': [r'dead', r'fluid'],})")
(%py-exec "workflow.TaskObject['Update Regions'].Execute()")
; ---------------------------------------------------------------------
; AUTO-MERGE FACIAL WALL ZONES
; The SpaceClaim Named Selection/label is 'wall'.  After Update Regions,
; Fluent can split that one label into multiple face zones (e.g. 7999+1).
; Query the face-zone IDs belonging to label 'wall' in object 'geom',
; merge all of them if needed, verify exactly one remains, then normalize
; its name and ID to the values used by the validated face_0003 journal.
; All tgapi-util-* calls below are Fluent meshing utility APIs.
(let* ((wall-zones (tgapi-util-get-face-zone-id-list-of-labels "geom" '(wall))))
  (format #t "\\n[AUTO-WALL] before merge: ids=~a counts=~a\\n"
          wall-zones
          (if (> (length wall-zones) 0)
              (tgapi-util-get-face-zone-count wall-zones)
              '()))
  (if (= (length wall-zones) 0)
      (error "[AUTO-WALL] no face zone found for SpaceClaim label 'wall'"))
  (if (> (length wall-zones) 1)
      (begin
        (format #t "[AUTO-WALL] merging ~a wall face zones...\\n" (length wall-zones))
        (tgapi-util-merge-face-zones wall-zones)))
  (let* ((wall-after (tgapi-util-get-face-zone-id-list-of-labels "geom" '(wall))))
    (format #t "[AUTO-WALL] after merge: ids=~a counts=~a\\n"
            wall-after
            (if (> (length wall-after) 0)
                (tgapi-util-get-face-zone-count wall-after)
                '()))
    (if (not (= (length wall-after) 1))
        (error "[AUTO-WALL] merge did not leave exactly one wall face zone"))
    (let ((wid (car wall-after)))
      ; Preserve the successful face_0003 journal below by normalizing the
      ; merged wall to its validated name and ID.
      (if (and (not (= wid 2030)) (tgapi-util-boundary-zone-exists? 2030))
          (error "[AUTO-WALL] target boundary ID 2030 is already occupied"))
      (tgapi-util-rename-face-zone wid "wall-zone------------:2030")
      (if (not (= wid 2030))
          (tgapi-util-renumber-zone-ids wall-after 2030))
      (format #t "[AUTO-WALL] normalized: id=2030 name=wall-zone------------:2030 count=~a\\n"
              (tgapi-util-get-face-zone-count '(2030))))))
(%py-exec "workflow.TaskObject['Add Boundary Layers'].Arguments.setState({r'FaceScope': {r'GrowOn': r'selected-zones',},r'ZoneSelectionList': [r'wall-zone------------:2030'],})")
(%py-exec "workflow.TaskObject['Add Boundary Layers'].AddChildToTask()")
(%py-exec "workflow.TaskObject['Add Boundary Layers'].InsertCompoundChildTask()")
(%py-exec "workflow.TaskObject['smooth-transition_1'].Arguments.setState({r'BLControlName': r'smooth-transition_1',r'FaceScope': {r'GrowOn': r'selected-zones',},r'ZoneSelectionList': [r'wall-zone------------:2030'],})")
(%py-exec "workflow.TaskObject['Add Boundary Layers'].Arguments.setState({})")
(%py-exec "workflow.TaskObject['smooth-transition_1'].Execute()")
(cx-use-window-id 51)
(%py-exec "workflow.TaskObject['Generate the Volume Mesh'].Arguments.setState({r'VolumeFill': r'poly-hexcore',})")
(%py-exec "workflow.TaskObject['Generate the Volume Mesh'].Execute()")
(cx-gui-do cx-activate-item "Ribbon*Frame1*Frame2(Task Page)*Table1*Table3(Solution)*PushButton1(Switch to Solution)")
(cx-gui-do cx-activate-item "Question*OK")
(newline)  
(cx-gui-do cx-activate-tab-index "NavigationPane*Frame1(TreeTab)" 1)
(cx-use-window-id 51)
(handle-key "??")
(cx-use-window-id 51)
(handle-key "??")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|farfield (wall, id=26)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|farfield (wall, id=26)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-activate-item "MenuBar*TypeSubMenu*symmetry")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list ))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|farfield-29 (wall, id=29)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-activate-item "MenuBar*TypeSubMenu*symmetry")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|farfield-30 (wall, id=30)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|farfield-30 (wall, id=30)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-activate-item "MenuBar*TypeSubMenu*symmetry")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|farfield-31 (wall, id=31)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-activate-item "MenuBar*TypeSubMenu*symmetry")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|inlet (wall, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|inlet (wall, id=24)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-activate-item "MenuBar*TypeSubMenu*velocity-inlet")
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 1)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 2)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 3)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 4)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 5)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 6)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 7)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Symmetry|farfield-30 (symmetry, id=30)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|wall-zone------------:2030 (wall, id=2030)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|outlet (wall, id=25)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|outlet (wall, id=25)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-activate-item "MenuBar*TypeSubMenu*pressure-outlet")
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 1)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 2)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 3)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 4)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 5)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 6)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 7)
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-item "Pressure Outlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-enable-apply-button "Pressure Outlet")
(cx-gui-do cx-activate-item "Pressure Outlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-enable-apply-button "Velocity Inlet")
(cx-gui-do cx-set-expression-entry "Velocity Inlet*Frame2*Frame2*Frame1(Momentum)*Table1*Table8*ExpressionEntry1(Velocity Magnitude)" '("5" . 0))
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Models|Energy (Off)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-activate-item "MenuBar*PopupMenuTree-Energy (Off)*On")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Models|Viscous (SST k-omega)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Models|Viscous (SST k-omega)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Models|Viscous (SST k-omega)"))
(cx-gui-do cx-activate-item "Viscous Model*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-enable-apply-button "Velocity Inlet")
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 1)
(cx-gui-do cx-enable-apply-button "Velocity Inlet")
(cx-gui-do cx-set-expression-entry "Velocity Inlet*Frame2*Frame2*Frame2(Thermal)*Table1*Table1*ExpressionEntry1(Temperature)" '("268.15" . 0))
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 1)
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Outlet|outlet (pressure-outlet, id=25)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Outlet|outlet (pressure-outlet, id=25)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Outlet|outlet (pressure-outlet, id=25)"))
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 1)
(cx-gui-do cx-enable-apply-button "Pressure Outlet")
(cx-gui-do cx-set-expression-entry "Pressure Outlet*Frame2*Frame2*Frame2(Thermal)*Table1*Table1*ExpressionEntry1(Backflow Total Temperature)" '("268.15" . 0))
(cx-gui-do cx-activate-item "Pressure Outlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-tab-index "Pressure Outlet*Frame2*Frame2" 0)
(cx-gui-do cx-activate-item "Pressure Outlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Internal|interior--zone------------ (interior, id=2683)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|wall-zone------------:2030 (wall, id=2030)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|wall-zone------------:2030 (wall, id=2030)"))
(cx-gui-do cx-list-tree-right-click "NavigationPane*List_Tree1" )
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|wall-zone------------:2030 (wall, id=2030)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|wall-zone------------:2030 (wall, id=2030)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Wall|wall-zone------------:2030 (wall, id=2030)"))
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 1)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 2)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 3)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 4)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 5)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 6)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 7)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 8)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 9)
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 1)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 2)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 3)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 4)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 0)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 5)
(cx-gui-do cx-activate-tab-index "Wall*Frame4*Frame8(Wall Film)*Frame1*Frame2" 0)
(cx-gui-do cx-enable-apply-button "Wall")
(cx-gui-do cx-activate-tab-index "Wall*Frame4" 1)
(cx-gui-do cx-enable-apply-button "Wall")
(cx-gui-do cx-set-toggle-button2 "Wall*Frame4*Frame2(Thermal)*Frame1*Frame1(Thermal Conditions)*ToggleBox1*Temperature" #t)
(cx-gui-do cx-activate-item "Wall*Frame4*Frame2(Thermal)*Frame1*Frame1(Thermal Conditions)*ToggleBox1*Temperature")
(cx-gui-do cx-set-expression-entry "Wall*Frame4*Frame2(Thermal)*Frame1*Frame1(Thermal Conditions)*Table3*Table1*Table1*ExpressionEntry1(Temperature)" '("307.15" . 0))
(cx-gui-do cx-activate-item "Wall*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Wall*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Methods"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Methods"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Methods"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Reference Values"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Reference Values"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Reference Values"))
(cx-gui-do cx-set-real-entry-list "Reference Values*Table2(Reference Values)*RealEntry7(Temperature)" '( 268.15))
(cx-gui-do cx-activate-item "Reference Values*Table2(Reference Values)*RealEntry7(Temperature)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-activate-item "Solution Initialization*Table1*Frame12*PushButton2(Initialize)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-set-integer-entry "Run Calculation*Table1*Table3(Parameters)*Table1*Table1*IntegerEntry1(Number of Iterations)" 100)
(cx-gui-do cx-activate-item "Run Calculation*Table1*Table3(Parameters)*Table1*Table1*IntegerEntry1(Number of Iterations)")
(cx-gui-do cx-activate-item "Run Calculation*Table1*Table6(Solution Advancement)*Table1*PushButton1(Calculate)")
(cx-gui-do cx-activate-item "Information*OK")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-set-list-selections "Flux Reports*List3(Boundaries)" '( 4))
(cx-gui-do cx-activate-item "Flux Reports*List3(Boundaries)")
(cx-gui-do cx-set-list-selections "Flux Reports*List3(Boundaries)" '( 4 6))
(cx-gui-do cx-activate-item "Flux Reports*List3(Boundaries)")
(cx-gui-do cx-activate-item "Flux Reports*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Flux Reports*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-activate-item "MenuBar*WriteSubMenu*Case & Data...")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "C:/ai-cfd-flow-prediction/ai-cfd-data/04_fluent/face_0003_05mps.cas.h5") "CFF Case/Data Files (*.cas.h5 *.dat.h5 )")
(cx-gui-do cx-activate-item "MenuBar*ExportSubMenu*Solution Data...")
(cx-gui-do cx-set-list-selections "Export*Table1*Table2*DropDownList1(File Type)" '( 1))
(cx-gui-do cx-activate-item "Export*Table1*Table2*DropDownList1(File Type)")
(cx-gui-do cx-set-list-selections "Export*Table1*List4(Surfaces)" '( 6))
(cx-gui-do cx-activate-item "Export*Table1*List4(Surfaces)")
(cx-gui-do cx-set-list-selections "Export*Table1*Table5*List1(Quantities)" '( 0 35 60 66 71 72))
(cx-gui-do cx-activate-item "Export*Table1*Table5*List1(Quantities)")
(cx-gui-do cx-activate-item "Export*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "C:/ai-cfd-flow-prediction/ai-cfd-data/05_cfd_csv/face_0003_05mps.csv") "ASCII Files ()")
(cx-gui-do cx-activate-item "Export*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-tab-index "Velocity Inlet*Frame2*Frame2" 0)
(cx-gui-do cx-enable-apply-button "Velocity Inlet")
(cx-gui-do cx-set-expression-entry "Velocity Inlet*Frame2*Frame2*Frame1(Momentum)*Table1*Table8*ExpressionEntry1(Velocity Magnitude)" '("8" . 0))
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-activate-item "Solution Initialization*Table1*Frame12*PushButton2(Initialize)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-activate-item "Run Calculation*Table1*Table6(Solution Advancement)*Table1*PushButton1(Calculate)")
(cx-gui-do cx-activate-item "Information*OK")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-activate-item "Flux Reports*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Flux Reports*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-activate-item "MenuBar*WriteSubMenu*Case & Data...")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "C:/ai-cfd-flow-prediction/ai-cfd-data/04_fluent/face_0003_08mps.cas.h5") "CFF Case/Data Files (*.cas.h5 *.dat.h5 )")
(cx-gui-do cx-activate-item "MenuBar*ExportSubMenu*Solution Data...")
(cx-gui-do cx-set-list-selections "Export*Table1*Table2*DropDownList1(File Type)" '( 1))
(cx-gui-do cx-activate-item "Export*Table1*Table2*DropDownList1(File Type)")
(cx-gui-do cx-set-list-selections "Export*Table1*List4(Surfaces)" '( 6))
(cx-gui-do cx-activate-item "Export*Table1*List4(Surfaces)")
(cx-gui-do cx-activate-item "Export*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "C:/ai-cfd-flow-prediction/ai-cfd-data/05_cfd_csv/face_0003_08mps.csv") "ASCII Files ()")
(cx-gui-do cx-activate-item "Export*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Setup|Boundary Conditions|Inlet|inlet (velocity-inlet, id=24)"))
(cx-gui-do cx-enable-apply-button "Velocity Inlet")
(cx-gui-do cx-set-expression-entry "Velocity Inlet*Frame2*Frame2*Frame1(Momentum)*Table1*Table8*ExpressionEntry1(Velocity Magnitude)" '("10" . 0))
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Velocity Inlet*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Initialization"))
(cx-gui-do cx-activate-item "Solution Initialization*Table1*Frame12*PushButton2(Initialize)")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Solution|Run Calculation"))
(cx-gui-do cx-activate-item "Run Calculation*Table1*Table6(Solution Advancement)*Table1*PushButton1(Calculate)")
(cx-gui-do cx-activate-item "Information*OK")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-activate-item "NavigationPane*List_Tree1")
(cx-gui-do cx-set-list-tree-selections "NavigationPane*List_Tree1" (list "Results|Reports|Fluxes"))
(cx-gui-do cx-activate-item "Flux Reports*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Flux Reports*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-activate-item "MenuBar*WriteSubMenu*Case & Data...")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "C:/ai-cfd-flow-prediction/ai-cfd-data/04_fluent/face_0003_10mps.cas.h5") "CFF Case/Data Files (*.cas.h5 *.dat.h5 )")
(cx-gui-do cx-activate-item "MenuBar*ExportSubMenu*Solution Data...")
(cx-gui-do cx-set-list-selections "Export*Table1*Table2*DropDownList1(File Type)" '( 1))
(cx-gui-do cx-activate-item "Export*Table1*Table2*DropDownList1(File Type)")
(cx-gui-do cx-set-list-selections "Export*Table1*List4(Surfaces)" '( 6))
(cx-gui-do cx-activate-item "Export*Table1*List4(Surfaces)")
(cx-gui-do cx-activate-item "Export*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "C:/ai-cfd-flow-prediction/ai-cfd-data/05_cfd_csv/face_0003_10mps.csv") "ASCII Files ()")
(cx-gui-do cx-activate-item "Export*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-activate-item "MenuBar*WriteSubMenu*Stop Journal")
"""


if __name__ == "__main__":
    sys.exit(main())