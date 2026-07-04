#!/usr/bin/env python3
# Kullanim:
#
#   Standart (full-image) degerlendirme:
#     python scripts/yolov11/test.py \
#         --model runs/detect/train/weights/best.pt \
#         -images /path/to/images -ann /path/to/annotations_coco.json
#
#   SAHI benzeri dilimli (cropped) inference ile degerlendirme:
#     python scripts/yolov11/test.py \
#         --model runs/detect/train/weights/best.pt \
#         -images /path/to/images -ann /path/to/annotations_coco.json \
#         --sliced --slice-size 1080 --overlap 0.2 --nms-iou 0.5 --device 0
#
#   Komutu calistirmadan once gormek icin:
#     python scripts/yolov11/test.py --model ... -images ... -ann ... --dry-run
#
# --model, -images ve -ann her zaman zorunludur; script herhangi bir dataset
# klasor yapisi varsaymaz (annotations_coco.json + duz bir goruntu klasoru
# yeterlidir). --config yalnizca --imgsz icin varsayilan degeri okumakta
# kullanilir.
#
# Cikti metrikleri: standart COCOeval ozeti (AP, AP50, AP75, AR@1/10/100) ve
# sinif bazinda precision, recall, mAP50, mAP50:95 tablosu (faster-coco-eval
# extended_metrics) - hem standart hem sliced modda ayni formatta basilir.
import sys
from argparse import ArgumentParser
from pathlib import Path

MODEL_NAME = "YOLOv11"
DEFAULT_CONFIG = "configs/yolov11/yolo11m_datasetv1.yaml"
ACTION = "test"
DEFAULT_IMGSZ = 640
DEFAULT_SLICE_SIZE = "1080"


def parse_args():
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to YAML config file, used only to source a default --imgsz.",
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO checkpoint (.pt) to evaluate.")
    parser.add_argument("-images", dest="val_images", required=True, help="Image folder to evaluate against.")
    parser.add_argument("-ann", dest="val_ann", required=True, help="COCO annotation file to evaluate against.")
    parser.add_argument("--imgsz", type=int, help="Model input size. Defaults to the config's train.imgsz.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for predictions.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold used by YOLO's own per-inference NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image/slice.")
    parser.add_argument("--device", help="Ultralytics device, for example: 0, 0,1 or cpu.")
    parser.add_argument("--dry-run", action="store_true", help="Print the eval plan without running it.")

    parser.add_argument(
        "--sliced",
        action="store_true",
        help=(
            "Use SAHI-style sliced (cropped) inference: split each image into overlapping "
            "windows, run YOLO per window, merge detections back to full-image coordinates "
            "with NMS, then evaluate the merged result."
        ),
    )
    parser.add_argument(
        "--slice-size",
        default=DEFAULT_SLICE_SIZE,
        help="Slice window size for --sliced, as SIZE or WIDTHxHEIGHT. Defaults to 1080.",
    )
    parser.add_argument("--overlap", type=float, default=0.2, help="Slice overlap ratio for both axes.")
    parser.add_argument("--overlap-width", type=float, help="Horizontal slice overlap ratio. Overrides --overlap.")
    parser.add_argument("--overlap-height", type=float, help="Vertical slice overlap ratio. Overrides --overlap.")
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="IoU threshold used to merge overlapping detections from adjacent slices.",
    )

    return parser.parse_args()


def load_default_imgsz(config_path: Path) -> int:
    if not config_path.exists():
        return DEFAULT_IMGSZ

    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return int(cfg.get("train", {}).get("imgsz", DEFAULT_IMGSZ))


def print_class_wise_metrics(coco_eval) -> None:
    metrics = coco_eval.extended_metrics
    print("Class-wise mAP:")
    print(f"{'class':<15}{'precision':>12}{'recall':>12}{'mAP50':>12}{'mAP50:95':>12}")
    for row in metrics["class_map"]:
        print(
            f"{row['class']:<15}{row['precision']:>12.4f}{row['recall']:>12.4f}"
            f"{row['map@50']:>12.4f}{row['map@50:95']:>12.4f}"
        )


def resolve_dataset_paths(args, workspace_root: Path) -> tuple[Path, Path]:
    img_folder = Path(args.val_images)
    if not img_folder.is_absolute():
        img_folder = workspace_root / img_folder

    ann_file = Path(args.val_ann)
    if not ann_file.is_absolute():
        ann_file = workspace_root / ann_file

    if not img_folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {img_folder}")
    if not ann_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    return img_folder, ann_file


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[2]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    img_folder, ann_file = resolve_dataset_paths(args, workspace_root)
    imgsz = args.imgsz or load_default_imgsz(config_path)
    overlap_width = args.overlap_width if args.overlap_width is not None else args.overlap
    overlap_height = args.overlap_height if args.overlap_height is not None else args.overlap

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}{' (sliced)' if args.sliced else ''}")
    print(f"weights: {args.model}")
    print(f"images: {img_folder}")
    print(f"annotations: {ann_file}")
    print(f"imgsz: {imgsz}")
    print(f"conf: {args.conf}  iou: {args.iou}  max_det: {args.max_det}")
    if args.sliced:
        print(f"slice_size: {args.slice_size}")
        print(f"overlap: width={overlap_width} height={overlap_height}")
        print(f"nms_iou: {args.nms_iou}")
    if args.device:
        print(f"device: {args.device}")

    if args.dry_run:
        return

    import torch
    import torchvision
    from PIL import Image
    from faster_coco_eval import COCO
    from faster_coco_eval.utils.pytorch import FasterCocoEvaluator
    from ultralytics import YOLO

    sys.path.insert(0, str(workspace_root))
    from tools.slice_dataset import Window, generate_windows, parse_crop_size

    model = YOLO(args.model)

    coco_gt = COCO(str(ann_file))
    evaluator = FasterCocoEvaluator(coco_gt, iou_types=["bbox"])

    slice_size = parse_crop_size(args.slice_size) if args.sliced else None

    predict_kwargs = dict(imgsz=imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det, verbose=False)
    if args.device:
        predict_kwargs["device"] = args.device

    image_ids = coco_gt.getImgIds()
    for n, image_id in enumerate(image_ids, start=1):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = img_folder / image_info["file_name"]

        if args.sliced:
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
                width, height = image.size
                windows = generate_windows(
                    image_width=width,
                    image_height=height,
                    crop_size=slice_size,
                    overlap_width_ratio=overlap_width,
                    overlap_height_ratio=overlap_height,
                )
                sources = [image.crop((w.left, w.top, w.right, w.bottom)) for w in windows]
                results = model.predict(sources, **predict_kwargs)
        else:
            windows = [Window(left=0, top=0, right=image_info["width"], bottom=image_info["height"])]
            results = model.predict([str(image_path)], **predict_kwargs)

        all_boxes, all_scores, all_labels = [], [], []
        for window, result in zip(windows, results):
            boxes = result.boxes.xyxy.clone()
            boxes[:, [0, 2]] += window.left
            boxes[:, [1, 3]] += window.top
            all_boxes.append(boxes)
            all_scores.append(result.boxes.conf)
            all_labels.append(result.boxes.cls.long())

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)

        if len(windows) > 1 and boxes.shape[0] > 0:
            keep = torchvision.ops.batched_nms(boxes, scores, labels, args.nms_iou)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        evaluator.update({image_id: {"boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()}})

        if n % 10 == 0 or n == len(image_ids):
            slice_note = f"{len(windows)} slice" + ("s" if len(windows) != 1 else "")
            print(f"[{n}/{len(image_ids)}] {image_info['file_name']} ({slice_note})")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print_class_wise_metrics(evaluator.coco_eval["bbox"])


if __name__ == "__main__":
    main()
