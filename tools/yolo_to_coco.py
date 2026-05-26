#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_classes(class_args: list[str] | None, labels_dir: Path) -> list[str]:
    if class_args:
        if len(class_args) == 1 and "," in class_args[0]:
            return [name.strip() for name in class_args[0].split(",") if name.strip()]
        return class_args

    max_class_id = -1
    for label_path in sorted(labels_dir.rglob("*.txt")):
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                class_id = int(stripped.split()[0])
                max_class_id = max(max_class_id, class_id)

    return [f"class_{idx}" for idx in range(max_class_id + 1)]


def yolo_box_to_coco(box: list[float], image_width: int, image_height: int) -> list[float]:
    x_center, y_center, width, height = box
    box_width = width * image_width
    box_height = height * image_height
    x_min = (x_center * image_width) - (box_width / 2)
    y_min = (y_center * image_height) - (box_height / 2)

    x_min = max(0.0, min(x_min, float(image_width)))
    y_min = max(0.0, min(y_min, float(image_height)))
    box_width = max(0.0, min(box_width, float(image_width) - x_min))
    box_height = max(0.0, min(box_height, float(image_height) - y_min))

    return [x_min, y_min, box_width, box_height]


def convert(input_dir: Path, output_path: Path, class_names: list[str]) -> None:
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")
    if not class_names:
        raise ValueError("No classes found. Pass --classes if labels are empty.")

    categories = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 1
    image_id = 1

    for image_path in iter_images(images_dir):
        relative_image_path = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_image_path.with_suffix(".txt")

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": relative_image_path.as_posix(),
                "width": image_width,
                "height": image_height,
            }
        )

        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    parts = stripped.split()
                    if len(parts) != 5:
                        raise ValueError(
                            f"Invalid YOLO row at {label_path}:{line_number}. "
                            "Expected: class_id x_center y_center width height"
                        )

                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= len(class_names):
                        raise ValueError(
                            f"class_id {class_id} at {label_path}:{line_number} "
                            f"is outside class range 0..{len(class_names) - 1}"
                        )

                    bbox = yolo_box_to_coco(
                        [float(value) for value in parts[1:]],
                        image_width,
                        image_height,
                    )
                    area = bbox[2] * bbox[3]

                    coco["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    annotation_id += 1

        image_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"images: {len(coco['images'])}")
    print(f"annotations: {len(coco['annotations'])}")
    print(f"categories: {len(coco['categories'])}")
    print(f"output: {output_path}")


def main() -> None:
    parser = ArgumentParser(description="Convert YOLO annotations to COCO JSON.")
    parser.add_argument(
        "input_dir",
        help="Directory containing images/ and labels/ subdirectories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output COCO JSON path. Defaults to <input_dir>/annotations.json.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Class names in class_id order. Also accepts a single comma-separated string.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else input_dir / "annotations.json"
    class_names = parse_classes(args.classes, input_dir / "labels")

    convert(input_dir, output_path, class_names)


if __name__ == "__main__":
    main()
