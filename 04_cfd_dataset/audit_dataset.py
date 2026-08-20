#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only audit of raw Fluent wall-field CSVs produced by 03_fluent_cfd."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

EXPECTED_HEADER = [
    "nodenumber", "x-coordinate", "y-coordinate", "z-coordinate",
    "pressure", "temperature", "y-plus", "wall-shear",
    "heat-flux", "heat-transfer-coef",
]
EXPECTED_SPEEDS = ("05mps", "08mps", "10mps")
FILE_RE = re.compile(r"^(?P<face>face_\d+)_(?P<speed>05mps|08mps|10mps)\.csv$", re.I)


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def update(self, x: float) -> None:
        self.count += 1
        self.min_value = min(self.min_value, x)
        self.max_value = max(self.max_value, x)
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0

    def as_dict(self) -> dict:
        if not self.count:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        return {
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean,
            "std": self.std,
        }


def normalize_header(row):
    return [str(x).strip().strip('"').strip("'").lower() for x in row]


def default_csv_dir() -> Path:
    # Expected location:
    # <project-root>/github/04_cfd_dataset/audit_dataset.py
    # <project-root>/ai-cfd-data/05_cfd_csv
    script = Path(__file__).resolve()
    if len(script.parents) >= 3:
        return script.parents[2] / "ai-cfd-data" / "05_cfd_csv"
    return Path.cwd() / "ai-cfd-data" / "05_cfd_csv"


def fmt(x):
    if x is None:
        return "-"
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-4:
        return f"{x:.6e}"
    return f"{x:.6f}"


def audit_file(path: Path, stats: dict, max_examples: int = 5) -> dict:
    out = {
        "file": path.name,
        "rows": 0,
        "header_ok": False,
        "bad_width_rows": 0,
        "numeric_parse_errors": 0,
        "nan_count": 0,
        "inf_count": 0,
        "issues": [],
    }

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f)
            try:
                header = normalize_header(next(reader))
            except StopIteration:
                out["issues"].append("empty file")
                return out

            out["header"] = header
            out["header_ok"] = header == EXPECTED_HEADER
            if not out["header_ok"]:
                out["issues"].append(f"header mismatch: {header}")
                return out

            for line_no, row in enumerate(reader, start=2):
                if not row or all(not str(v).strip() for v in row):
                    continue

                if len(row) != len(EXPECTED_HEADER):
                    out["bad_width_rows"] += 1
                    if len(out["issues"]) < max_examples:
                        out["issues"].append(
                            f"line {line_no}: expected 10 fields, got {len(row)}"
                        )
                    continue

                values = []
                valid = True
                for name, raw in zip(EXPECTED_HEADER, row):
                    text = str(raw).strip().strip('"').strip("'")
                    try:
                        value = float(text)
                    except ValueError:
                        out["numeric_parse_errors"] += 1
                        valid = False
                        if len(out["issues"]) < max_examples:
                            out["issues"].append(
                                f"line {line_no}: non-numeric {name}={text!r}"
                            )
                        break

                    if math.isnan(value):
                        out["nan_count"] += 1
                        valid = False
                        if len(out["issues"]) < max_examples:
                            out["issues"].append(f"line {line_no}: NaN in {name}")
                        break
                    if math.isinf(value):
                        out["inf_count"] += 1
                        valid = False
                        if len(out["issues"]) < max_examples:
                            out["issues"].append(f"line {line_no}: Inf in {name}")
                        break
                    values.append(value)

                if not valid:
                    continue

                out["rows"] += 1
                for name, value in zip(EXPECTED_HEADER, values):
                    stats[name].update(value)

    except OSError as exc:
        out["issues"].append(f"read error: {exc}")

    return out


def save_reports(output_dir: Path, summary: dict, file_results: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "audit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "file_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "face", "speed", "rows", "header_ok", "bad_width_rows",
            "numeric_parse_errors", "nan_count", "inf_count",
        ])
        for r in file_results:
            writer.writerow([
                r["file"], r["face"], r["speed"], r["rows"], r["header_ok"],
                r["bad_width_rows"], r["numeric_parse_errors"], r["nan_count"], r["inf_count"],
            ])

    print(f"\n[REPORT] {json_path}")
    print(f"[REPORT] {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only audit of raw Fluent CFD CSVs.")
    ap.add_argument("--csv-dir", type=Path, default=default_csv_dir())
    ap.add_argument("--expected-faces", type=int, default=100)
    ap.add_argument("--expected-files", type=int, default=300)
    ap.add_argument("--save-report", action="store_true")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "audit_results",
    )
    args = ap.parse_args()

    csv_dir = args.csv_dir.resolve()

    print("=" * 80)
    print("CFD DATASET AUDIT")
    print("=" * 80)
    print("CSV directory :", csv_dir)
    print("Expected files:", args.expected_files)
    print("Expected faces:", args.expected_faces)
    print("Speeds        :", ", ".join(EXPECTED_SPEEDS))
    print("=" * 80)

    if not csv_dir.is_dir():
        print(f"[FAIL] CSV directory not found: {csv_dir}", file=sys.stderr)
        return 1

    all_csv = sorted(p for p in csv_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    matched = []
    unexpected = []
    face_to_files = defaultdict(dict)

    for path in all_csv:
        m = FILE_RE.match(path.name)
        if not m:
            unexpected.append(path.name)
            continue
        face = m.group("face").lower()
        speed = m.group("speed").lower()
        matched.append((path, face, speed))
        face_to_files[face][speed] = path

    faces = sorted(face_to_files)
    missing = {
        face: [speed for speed in EXPECTED_SPEEDS if speed not in face_to_files[face]]
        for face in faces
    }
    missing = {face: speeds for face, speeds in missing.items() if speeds}
    complete_faces = [face for face in faces if face not in missing]

    print("\n[STRUCTURE]")
    print(f"CSV files found          : {len(all_csv)}")
    print(f"Matching expected names  : {len(matched)}")
    print(f"Unexpected CSV names     : {len(unexpected)}")
    print(f"Unique faces             : {len(faces)}")
    print(f"Complete 3-speed triplets: {len(complete_faces)}")
    print(f"Incomplete faces         : {len(missing)}")

    if unexpected:
        print("\n[UNEXPECTED FILENAMES]")
        for name in unexpected[:20]:
            print(" -", name)

    if missing:
        print("\n[MISSING SPEED FILES]")
        for face, speeds in list(missing.items())[:20]:
            print(f" - {face}: {', '.join(speeds)}")

    stats = {name: RunningStats() for name in EXPECTED_HEADER}
    results = []

    print("\n[READING FILES]")
    total = len(matched)
    for i, (path, face, speed) in enumerate(matched, start=1):
        r = audit_file(path, stats)
        r["face"] = face
        r["speed"] = speed
        results.append(r)
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  {i:>3}/{total}  {path.name}")

    row_counts = [r["rows"] for r in results if r["rows"] > 0]
    bad_headers = sum(not r["header_ok"] for r in results)
    empty_files = [r["file"] for r in results if r["rows"] == 0]
    bad_width = sum(r["bad_width_rows"] for r in results)
    parse_errors = sum(r["numeric_parse_errors"] for r in results)
    nan_count = sum(r["nan_count"] for r in results)
    inf_count = sum(r["inf_count"] for r in results)

    rows_by_face = defaultdict(dict)
    for r in results:
        rows_by_face[r["face"]][r["speed"]] = r["rows"]

    row_mismatches = {}
    for face, speed_rows in rows_by_face.items():
        if all(speed in speed_rows for speed in EXPECTED_SPEEDS):
            counts = [speed_rows[speed] for speed in EXPECTED_SPEEDS]
            if len(set(counts)) != 1:
                row_mismatches[face] = {speed: speed_rows[speed] for speed in EXPECTED_SPEEDS}

    print("\n[ROW COUNTS]")
    if row_counts:
        print(f"Samples with data: {len(row_counts)}")
        print(f"Total data rows  : {sum(row_counts):,}")
        print(f"Min rows/sample  : {min(row_counts):,}")
        print(f"Max rows/sample  : {max(row_counts):,}")
        print(f"Mean rows/sample : {statistics.fmean(row_counts):,.2f}")
        print(f"Median           : {statistics.median(row_counts):,.2f}")
    else:
        print("No valid data rows found.")
    print(f"Faces with 5/8/10 row-count mismatch: {len(row_mismatches)}")

    if row_mismatches:
        print("\n[ROW-COUNT MISMATCHES]")
        for face, counts in list(row_mismatches.items())[:20]:
            print(" -", face, ", ".join(f"{s}={n:,}" for s, n in counts.items()))

    print("\n[DATA QUALITY]")
    print(f"Files with bad header : {bad_headers}")
    print(f"Files with zero rows  : {len(empty_files)}")
    print(f"Bad-width rows        : {bad_width}")
    print(f"Numeric parse errors  : {parse_errors}")
    print(f"NaN values            : {nan_count}")
    print(f"Inf values            : {inf_count}")

    issue_files = [
        r for r in results
        if (not r["header_ok"] or r["rows"] == 0 or r["bad_width_rows"]
            or r["numeric_parse_errors"] or r["nan_count"] or r["inf_count"])
    ]
    if issue_files:
        print("\n[ISSUE EXAMPLES]")
        for r in issue_files[:20]:
            print(" -", r["file"])
            for issue in r["issues"]:
                print("    *", issue)

    print("\n[AGGREGATE COLUMN STATISTICS]")
    print(f"{'column':<22}{'count':>14}{'min':>16}{'max':>16}{'mean':>16}{'std':>16}")
    print("-" * 100)
    stats_dict = {}
    for name in EXPECTED_HEADER:
        s = stats[name]
        stats_dict[name] = s.as_dict()
        print(
            f"{name:<22}{s.count:>14,}"
            f"{fmt(s.min_value if s.count else None):>16}"
            f"{fmt(s.max_value if s.count else None):>16}"
            f"{fmt(s.mean if s.count else None):>16}"
            f"{fmt(s.std if s.count else None):>16}"
        )

    structure_ok = (
        len(all_csv) == args.expected_files
        and len(matched) == args.expected_files
        and len(faces) == args.expected_faces
        and len(complete_faces) == args.expected_faces
        and not unexpected
        and not missing
    )
    data_ok = (
        bad_headers == 0
        and not empty_files
        and bad_width == 0
        and parse_errors == 0
        and nan_count == 0
        and inf_count == 0
        and not row_mismatches
    )
    overall_ok = structure_ok and data_ok

    summary = {
        "csv_directory": str(csv_dir),
        "expected_files": args.expected_files,
        "files_found": len(all_csv),
        "matching_files": len(matched),
        "unexpected_filenames": unexpected,
        "expected_faces": args.expected_faces,
        "unique_faces": len(faces),
        "complete_triplets": len(complete_faces),
        "missing_triplets": missing,
        "row_counts": {
            "samples_with_data": len(row_counts),
            "total": sum(row_counts) if row_counts else 0,
            "min": min(row_counts) if row_counts else None,
            "max": max(row_counts) if row_counts else None,
            "mean": statistics.fmean(row_counts) if row_counts else None,
            "median": statistics.median(row_counts) if row_counts else None,
            "mismatch_faces": row_mismatches,
        },
        "quality": {
            "bad_header_files": bad_headers,
            "empty_data_files": empty_files,
            "bad_width_rows": bad_width,
            "numeric_parse_errors": parse_errors,
            "nan_count": nan_count,
            "inf_count": inf_count,
        },
        "column_stats": stats_dict,
        "overall_ok": overall_ok,
    }

    if args.save_report:
        save_reports(args.output_dir.resolve(), summary, results)

    print("\n" + "=" * 80)
    print("[PASS] Raw CFD dataset audit passed." if overall_ok
          else "[FAIL] Raw CFD dataset audit found one or more issues.")
    print("=" * 80)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
