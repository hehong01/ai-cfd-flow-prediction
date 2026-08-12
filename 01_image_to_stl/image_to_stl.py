"""Convert face images in the local data directory to watertight STL files.

Default local data flow
-----------------------
ai-cfd-data/01_images/<face_id>.jpg
    -> ai-cfd-data/02_stl/<face_id>.stl

The geometric algorithm is reconstructed from ``back_head_v6_2.py`` while
removing historical hard-coded paths, persistent intermediate CSV files, and
enclosure generation.
"""

import argparse
import sys
from pathlib import Path

# Allow this script to be executed directly from 01_image_to_stl/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import IMAGE_DIR, STL_DIR

from face_mesh import MESH_INDEX, extract_scaled_landmarks
from mesh_utils import (
    build_smoothed_face_mesh,
    close_face_mesh,
    save_stl,
)


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def convert_image_to_stl(
    image_path,
    output_path,
    resolution=3,
    iterations=10,
    target_height=200.0,
):
    """Convert one frontal face image into the reconstructed watertight STL."""
    image_path = Path(image_path)
    output_path = Path(output_path)

    landmarks = extract_scaled_landmarks(
        image_path,
        target_height=target_height,
    )

    face_vertices, face_faces = build_smoothed_face_mesh(
        landmarks,
        MESH_INDEX,
        resolution=resolution,
        iterations=iterations,
        alpha=0.3,
        beta=0.5,
    )

    closed_vertices, closed_faces = close_face_mesh(
        face_vertices,
        face_faces,
    )

    save_stl(
        closed_vertices,
        closed_faces,
        output_path,
    )

    return output_path


def resolve_input_path(input_value):
    """Resolve --input as an absolute path or as a file inside IMAGE_DIR."""
    candidate = Path(input_value)
    if not candidate.is_absolute():
        candidate = IMAGE_DIR / candidate
    return candidate.resolve()


def collect_images():
    """Collect supported images from IMAGE_DIR in deterministic name order."""
    if not IMAGE_DIR.exists():
        return []

    return sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def process_one(image_path, resolution, iterations, target_height):
    """Process one image and save <stem>.stl in STL_DIR."""
    output_path = STL_DIR / f"{image_path.stem}.stl"

    print("=" * 60)
    print(f"Input : {image_path}")
    print(f"Output: {output_path}")

    convert_image_to_stl(
        image_path=image_path,
        output_path=output_path,
        resolution=resolution,
        iterations=iterations,
        target_height=target_height,
    )

    print(f"Saved : {output_path}")
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert frontal face images to watertight STL files."
    )
    parser.add_argument(
        "--input",
        help=(
            "Optional image path. A bare filename is resolved inside "
            "ai-cfd-data/01_images. If omitted, every JPG/JPEG/PNG in "
            "01_images is processed."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=3,
        help="Triangle subdivision resolution (original default: 3).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="HC Laplacian smoothing iterations (original default: 10).",
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=200.0,
        help="Uniform target face-landmark height in mm (original: 200).",
    )
    return parser


def main():
    args = build_parser().parse_args()

    STL_DIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        image_paths = [resolve_input_path(args.input)]
    else:
        image_paths = collect_images()

    if not image_paths:
        print(f"No JPG/JPEG/PNG images found in: {IMAGE_DIR}")
        return 0

    failed = []

    for image_path in image_paths:
        if not image_path.is_file():
            print(f"FAILED: input image does not exist: {image_path}")
            failed.append(image_path)
            continue

        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            print(f"FAILED: unsupported image type: {image_path}")
            failed.append(image_path)
            continue

        try:
            process_one(
                image_path=image_path,
                resolution=args.resolution,
                iterations=args.iterations,
                target_height=args.target_height,
            )
        except Exception as exc:
            print(f"FAILED: {image_path.name}: {exc}")
            failed.append(image_path)

    if failed:
        print(f"Completed with {len(failed)} failure(s).")
        return 1

    print(f"Completed successfully: {len(image_paths)} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
