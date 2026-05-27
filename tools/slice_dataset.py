#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_ANNOTATION_KEYS_TO_REWRITE = {"id", "image_id", "bbox", "area", "segmentation"}


@dataclass(frozen=True)
class CropSize:
    width: int
    height: int


@dataclass(frozen=True)
class Window:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class BBox:
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class CocoAnnotation:
    annotation: dict
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def parse_crop_size(value: str) -> CropSize:
    normalized = value.lower().replace(",", "x")
    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if len(parts) == 1:
        width = height = int(parts[0])
    elif len(parts) == 2:
        width, height = (int(part) for part in parts)
    else:
        raise ValueError(f"Invalid crop size: {value}. Use 640 or 640x640.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Crop size must be positive: {value}")

    return CropSize(width=width, height=height)


def parse_crop_sizes(values: list[str]) -> list[CropSize]:
    seen: set[tuple[int, int]] = set()
    crop_sizes: list[CropSize] = []
    for value in values:
        crop_size = parse_crop_size(value)
        key = (crop_size.width, crop_size.height)
        if key not in seen:
            seen.add(key)
            crop_sizes.append(crop_size)
    return crop_sizes


def resize_image(image: Image.Image, resize_size: CropSize | None) -> Image.Image:
    if resize_size is None or image.size == (resize_size.width, resize_size.height):
        return image

    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return image.resize((resize_size.width, resize_size.height), resampling)


def validate_overlap(value: float, name: str) -> float:
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be in the range [0, 1). Got: {value}")
    return value


def iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def make_starts(full_size: int, crop_size: int, overlap_ratio: float) -> list[int]:
    if full_size <= crop_size:
        return [0]

    step = max(1, int(round(crop_size * (1.0 - overlap_ratio))))
    starts = list(range(0, full_size - crop_size + 1, step))
    last_start = full_size - crop_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_windows(
    image_width: int,
    image_height: int,
    crop_size: CropSize,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
) -> list[Window]:
    x_starts = make_starts(image_width, crop_size.width, overlap_width_ratio)
    y_starts = make_starts(image_height, crop_size.height, overlap_height_ratio)

    windows: list[Window] = []
    for y_start in y_starts:
        for x_start in x_starts:
            windows.append(
                Window(
                    left=x_start,
                    top=y_start,
                    right=min(x_start + crop_size.width, image_width),
                    bottom=min(y_start + crop_size.height, image_height),
                )
            )
    return windows


def intersect_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    window: Window,
    min_area_ratio: float,
) -> tuple[float, float, float, float] | None:
    clipped_x_min = max(x_min, float(window.left))
    clipped_y_min = max(y_min, float(window.top))
    clipped_x_max = min(x_max, float(window.right))
    clipped_y_max = min(y_max, float(window.bottom))

    clipped_width = clipped_x_max - clipped_x_min
    clipped_height = clipped_y_max - clipped_y_min
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None

    original_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    clipped_area = clipped_width * clipped_height
    if original_area <= 0.0 or (clipped_area / original_area) < min_area_ratio:
        return None

    return (
        clipped_x_min - window.left,
        clipped_y_min - window.top,
        clipped_x_max - window.left,
        clipped_y_max - window.top,
    )


def yolo_to_bbox(
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> BBox:
    box_width = width * image_width
    box_height = height * image_height
    x_min = (x_center * image_width) - (box_width / 2.0)
    y_min = (y_center * image_height) - (box_height / 2.0)
    x_max = x_min + box_width
    y_max = y_min + box_height

    return BBox(
        class_id=class_id,
        x_min=max(0.0, min(x_min, float(image_width))),
        y_min=max(0.0, min(y_min, float(image_height))),
        x_max=max(0.0, min(x_max, float(image_width))),
        y_max=max(0.0, min(y_max, float(image_height))),
    )


def bbox_to_yolo_line(
    class_id: int,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    x_min, y_min, x_max, y_max = bbox
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + (box_width / 2.0)
    y_center = y_min + (box_height / 2.0)
    return (
        f"{class_id} "
        f"{x_center / width:.6f} "
        f"{y_center / height:.6f} "
        f"{box_width / width:.6f} "
        f"{box_height / height:.6f}"
    )


def read_yolo_labels(label_path: Path, image_width: int, image_height: int) -> list[BBox]:
    boxes: list[BBox] = []
    if not label_path.exists():
        return boxes

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

            boxes.append(
                yolo_to_bbox(
                    class_id=int(parts[0]),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                    image_width=image_width,
                    image_height=image_height,
                )
            )
    return boxes


def make_slice_relative_path(relative_image_path: Path, crop_size: CropSize, window: Window) -> Path:
    suffix = relative_image_path.suffix
    stem = relative_image_path.stem
    parent = relative_image_path.parent
    sliced_name = (
        f"{stem}__slice_{crop_size.width}x{crop_size.height}"
        f"_x{window.left}_y{window.top}{suffix}"
    )
    if parent == Path("."):
        return Path(sliced_name)
    return parent / sliced_name


def save_image_slice(image: Image.Image, output_path: Path, window: Window) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_slice = image.crop((window.left, window.top, window.right, window.bottom))
    if output_path.suffix.lower() in {".jpg", ".jpeg"} and image_slice.mode not in {"RGB", "L"}:
        image_slice = image_slice.convert("RGB")
    image_slice.save(output_path)


def slice_yolo_dataset(
    input_dir: Path,
    output_dir: Path,
    crop_sizes: list[CropSize],
    resize_size: CropSize | None,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
    min_area_ratio: float,
    skip_empty: bool,
) -> None:
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    image_count = 0
    label_count = 0
    box_count = 0

    for image_path in iter_images(images_dir):
        relative_image_path = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_image_path.with_suffix(".txt")

        with Image.open(image_path) as image:
            image = resize_image(image, resize_size)
            image_width, image_height = image.size
            boxes = read_yolo_labels(label_path, image_width, image_height)

            for crop_size in crop_sizes:
                windows = generate_windows(
                    image_width=image_width,
                    image_height=image_height,
                    crop_size=crop_size,
                    overlap_width_ratio=overlap_width_ratio,
                    overlap_height_ratio=overlap_height_ratio,
                )
                for window in windows:
                    sliced_lines: list[str] = []
                    for box in boxes:
                        clipped = intersect_bbox(
                            box.x_min,
                            box.y_min,
                            box.x_max,
                            box.y_max,
                            window,
                            min_area_ratio,
                        )
                        if clipped is None:
                            continue
                        sliced_lines.append(
                            bbox_to_yolo_line(
                                box.class_id,
                                clipped,
                                window.width,
                                window.height,
                            )
                        )

                    if skip_empty and not sliced_lines:
                        continue

                    sliced_relative_path = make_slice_relative_path(relative_image_path, crop_size, window)
                    output_image_path = output_images_dir / sliced_relative_path
                    output_label_path = output_labels_dir / sliced_relative_path.with_suffix(".txt")

                    save_image_slice(image, output_image_path, window)
                    output_label_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_label_path.open("w", encoding="utf-8") as handle:
                        if sliced_lines:
                            handle.write("\n".join(sliced_lines))
                            handle.write("\n")

                    image_count += 1
                    label_count += 1
                    box_count += len(sliced_lines)

    print(f"images: {image_count}")
    print(f"labels: {label_count}")
    print(f"annotations: {box_count}")
    if resize_size is not None:
        print(f"resize_to: {resize_size.width}x{resize_size.height}")
    print(f"output: {output_dir}")


def load_coco_annotations(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    required_keys = {"images", "annotations", "categories"}
    missing_keys = sorted(required_keys - set(coco))
    if missing_keys:
        raise ValueError(f"COCO file is missing required keys: {', '.join(missing_keys)}")

    return coco


def coco_bbox_to_annotation(annotation: dict) -> CocoAnnotation:
    x_min, y_min, width, height = annotation["bbox"]
    return CocoAnnotation(
        annotation=annotation,
        x_min=float(x_min),
        y_min=float(y_min),
        x_max=float(x_min + width),
        y_max=float(y_min + height),
    )


def scale_coco_annotation(
    coco_annotation: CocoAnnotation,
    scale_x: float,
    scale_y: float,
) -> CocoAnnotation:
    return CocoAnnotation(
        annotation=coco_annotation.annotation,
        x_min=coco_annotation.x_min * scale_x,
        y_min=coco_annotation.y_min * scale_y,
        x_max=coco_annotation.x_max * scale_x,
        y_max=coco_annotation.y_max * scale_y,
    )


def slice_coco_dataset(
    images_dir: Path,
    annotation_path: Path,
    output_dir: Path,
    crop_sizes: list[CropSize],
    resize_size: CropSize | None,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
    min_area_ratio: float,
    skip_empty: bool,
    output_annotation_name: str,
) -> None:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

    output_annotation_path = output_dir / output_annotation_name
    coco = load_coco_annotations(annotation_path)

    annotations_by_image_id: dict[int, list[CocoAnnotation]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image_id[int(annotation["image_id"])].append(
            coco_bbox_to_annotation(annotation)
        )

    sliced_coco = {
        "images": [],
        "annotations": [],
        "categories": coco["categories"],
    }
    for optional_key in ("info", "licenses"):
        if optional_key in coco:
            sliced_coco[optional_key] = coco[optional_key]

    next_image_id = 1
    next_annotation_id = 1

    for image_record in sorted(coco["images"], key=lambda item: item["id"]):
        relative_image_path = Path(image_record["file_name"])
        if relative_image_path.is_absolute():
            raise ValueError(
                f"COCO image file_name must be relative, got: {relative_image_path}"
            )
        image_path = images_dir / relative_image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image from COCO file not found: {image_path}")

        with Image.open(image_path) as image:
            source_width, source_height = image.size
            image = resize_image(image, resize_size)
            image_width, image_height = image.size
            image_annotations = annotations_by_image_id.get(int(image_record["id"]), [])
            if (source_width, source_height) != (image_width, image_height):
                scale_x = image_width / source_width
                scale_y = image_height / source_height
                image_annotations = [
                    scale_coco_annotation(annotation, scale_x, scale_y)
                    for annotation in image_annotations
                ]

            for crop_size in crop_sizes:
                windows = generate_windows(
                    image_width=image_width,
                    image_height=image_height,
                    crop_size=crop_size,
                    overlap_width_ratio=overlap_width_ratio,
                    overlap_height_ratio=overlap_height_ratio,
                )
                for window in windows:
                    sliced_annotations: list[dict] = []
                    for coco_annotation in image_annotations:
                        clipped = intersect_bbox(
                            coco_annotation.x_min,
                            coco_annotation.y_min,
                            coco_annotation.x_max,
                            coco_annotation.y_max,
                            window,
                            min_area_ratio,
                        )
                        if clipped is None:
                            continue

                        x_min, y_min, x_max, y_max = clipped
                        width = x_max - x_min
                        height = y_max - y_min
                        sliced_annotation = {
                            key: value
                            for key, value in coco_annotation.annotation.items()
                            if key not in COCO_ANNOTATION_KEYS_TO_REWRITE
                        }
                        sliced_annotation.update(
                            {
                                "id": next_annotation_id,
                                "image_id": next_image_id,
                                "bbox": [x_min, y_min, width, height],
                                "area": width * height,
                                "segmentation": [],
                            }
                        )
                        sliced_annotations.append(sliced_annotation)
                        next_annotation_id += 1

                    if skip_empty and not sliced_annotations:
                        continue

                    sliced_relative_path = make_slice_relative_path(
                        relative_image_path,
                        crop_size,
                        window,
                    )
                    output_image_path = output_dir / sliced_relative_path

                    save_image_slice(image, output_image_path, window)
                    sliced_coco["images"].append(
                        {
                            "id": next_image_id,
                            "file_name": sliced_relative_path.as_posix(),
                            "width": window.width,
                            "height": window.height,
                        }
                    )
                    sliced_coco["annotations"].extend(sliced_annotations)
                    next_image_id += 1

    output_annotation_path.parent.mkdir(parents=True, exist_ok=True)
    with output_annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(sliced_coco, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"images: {len(sliced_coco['images'])}")
    print(f"annotations: {len(sliced_coco['annotations'])}")
    print(f"categories: {len(sliced_coco['categories'])}")
    if resize_size is not None:
        print(f"resize_to: {resize_size.width}x{resize_size.height}")
    print(f"output: {output_dir}")
    print(f"annotations_output: {output_annotation_path}")


def detect_format(input_dir: Path, annotation_path: Path | None) -> Literal["yolo", "coco"]:
    if annotation_path is not None:
        return "coco"
    if (input_dir / "images").is_dir() and (input_dir / "labels").is_dir():
        return "yolo"
    raise ValueError(
        "Could not auto-detect dataset format. Pass --format yolo or --format coco. "
        "YOLO input must contain images/ and labels/. COCO input must use --annotations."
    )


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Slice an object-detection dataset into SAHI-style overlapping crops. "
            "YOLO input keeps YOLO output; COCO input writes a sliced COCO JSON."
        )
    )
    parser.add_argument(
        "input_dir",
        help=(
            "YOLO dataset root containing images/ and labels/, or COCO image root "
            "when --annotations is passed."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "yolo", "coco"],
        default="auto",
        help="Input/output annotation format. Defaults to auto.",
    )
    parser.add_argument(
        "--annotations",
        help="COCO annotation JSON path. Required for --format coco.",
    )
    parser.add_argument(
        "--output-annotations",
        default=None,
        help="Output COCO annotation file name. Defaults to the input JSON file name.",
    )
    parser.add_argument(
        "--crop-size",
        action="append",
        required=True,
        help="Crop size as SIZE or WIDTHxHEIGHT. Can be repeated.",
    )
    parser.add_argument(
        "--resize-to",
        default=None,
        help=(
            "Resize every source image to WIDTHxHEIGHT before slicing. "
            "YOLO labels and COCO annotations are scaled to the resized image."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Overlap ratio for both axes. Defaults to 0.2.",
    )
    parser.add_argument(
        "--overlap-width",
        type=float,
        default=None,
        help="Horizontal overlap ratio. Overrides --overlap for width.",
    )
    parser.add_argument(
        "--overlap-height",
        type=float,
        default=None,
        help="Vertical overlap ratio. Overrides --overlap for height.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.1,
        help=(
            "Minimum visible area ratio needed to keep a clipped box. "
            "SAHI-style default is 0.1."
        ),
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not save crops that contain no annotations after clipping.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    annotation_path = Path(args.annotations).resolve() if args.annotations else None
    crop_sizes = parse_crop_sizes(args.crop_size)
    resize_size = parse_crop_size(args.resize_to) if args.resize_to else None

    overlap_width_ratio = validate_overlap(
        args.overlap_width if args.overlap_width is not None else args.overlap,
        "--overlap-width",
    )
    overlap_height_ratio = validate_overlap(
        args.overlap_height if args.overlap_height is not None else args.overlap,
        "--overlap-height",
    )
    if args.min_area_ratio < 0.0 or args.min_area_ratio > 1.0:
        raise ValueError(
            f"--min-area-ratio must be in the range [0, 1]. Got: {args.min_area_ratio}"
        )

    dataset_format = detect_format(input_dir, annotation_path) if args.format == "auto" else args.format

    if dataset_format == "yolo":
        if annotation_path is not None:
            raise ValueError("--annotations can only be used with --format coco")
        slice_yolo_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            crop_sizes=crop_sizes,
            resize_size=resize_size,
            overlap_width_ratio=overlap_width_ratio,
            overlap_height_ratio=overlap_height_ratio,
            min_area_ratio=args.min_area_ratio,
            skip_empty=args.skip_empty,
        )
        return

    if annotation_path is None:
        raise ValueError("--annotations is required for COCO slicing")
    output_annotation_name = args.output_annotations or annotation_path.name
    if Path(output_annotation_name).is_absolute():
        raise ValueError("--output-annotations must be a file name or relative path")

    slice_coco_dataset(
        images_dir=input_dir,
        annotation_path=annotation_path,
        output_dir=output_dir,
        crop_sizes=crop_sizes,
        resize_size=resize_size,
        overlap_width_ratio=overlap_width_ratio,
        overlap_height_ratio=overlap_height_ratio,
        min_area_ratio=args.min_area_ratio,
        skip_empty=args.skip_empty,
        output_annotation_name=output_annotation_name,
    )


if __name__ == "__main__":
    main()
