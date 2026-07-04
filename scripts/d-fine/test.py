#!/usr/bin/env python3
# Kullanim:
#
#   Standart degerlendirme (D-FINE'in kendi train.py --test-only akisi):
#     python scripts/d-fine/test.py --resume output/dfine_.../best_stg2.pth \
#         -images datasets/datasetv1/val -ann datasets/datasetv1/val/val.json
#
#   Farkli bir model config'i ile:
#     python scripts/d-fine/test.py --config configs/d-fine/dfine_hgnetv2_m_datasetv1.yml \
#         --resume output/dfine_.../best_stg2.pth \
#         -images datasets/datasetv1/val -ann datasets/datasetv1/val/val.json
#
#   SAHI benzeri dilimli (cropped) inference ile degerlendirme:
#     python scripts/d-fine/test.py --resume output/dfine_.../best_stg2.pth \
#         -images datasets/datasetv1/val -ann datasets/datasetv1/val/val.json \
#         --sliced --slice-size 1080 --overlap 0.2 --nms-iou 0.5
#
#   Komutu calistirmadan once gormek icin:
#     python scripts/d-fine/test.py --resume ... -images ... -ann ... --dry-run
#
#   Belirli GPU(lar) uzerinde calistirmak icin:
#     python scripts/d-fine/test.py --resume ... -images ... -ann ... --devices 0
#
# -images ve -ann her zaman zorunludur; script herhangi bir dataset
# klasor yapisi varsaymaz. Ek D-FINE config override'lari icin --update
# key=value seklinde gecilebilir (orn. --update val_dataloader.total_batch_size=8).
#
# Cikti metrikleri: precision, recall, f1, iou (D-FINE Validator, conf/IoU=0.5)
# ve mAP50:95, mAP50, mAP75, AR@1/10/100 (COCOeval) - hem standart hem sliced
# modda ayni formatta basilir.
import os
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

MODEL_NAME = "D-FINE"
MODEL_DIR = "models/D-FINE"
DEFAULT_CONFIG = "configs/d-fine/dfine_hgnetv2_m_datasetv1.yml"
ACTION = "test"
DEFAULT_EVAL_SIZE = 640


def parse_args():
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--resume", required=True, help="Path to checkpoint (.pth) to evaluate.")
    parser.add_argument("--devices", help="CUDA_VISIBLE_DEVICES value, for example: 0 or 0,1,2,3.")
    parser.add_argument(
        "-images",
        dest="val_images",
        required=True,
        help="Val image folder to evaluate against (absolute or workspace-relative).",
    )
    parser.add_argument(
        "-ann",
        dest="val_ann",
        required=True,
        help="Val COCO annotation file to evaluate against (absolute or workspace-relative).",
    )
    parser.add_argument(
        "--update",
        nargs="*",
        default=[],
        help="Additional D-FINE config overrides, for example: val_dataloader.total_batch_size=8",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the eval plan without running it.")

    parser.add_argument(
        "--sliced",
        action="store_true",
        help=(
            "Use SAHI-style sliced (cropped) inference: split each val image into overlapping "
            "windows, run the model per window, merge detections back to full-image coordinates "
            "with NMS, then evaluate the merged result."
        ),
    )
    parser.add_argument(
        "--slice-size",
        default="1080",
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
    parser.add_argument(
        "--eval-size",
        type=int,
        help="Square model input size for each slice. Defaults to the Resize size in the config's val transforms.",
    )

    return parser.parse_args()


def resolve_dataset_paths(args, workspace_root: Path) -> tuple[Path, Path]:
    img_folder = Path(args.val_images)
    if not img_folder.is_absolute():
        img_folder = workspace_root / img_folder

    ann_file = Path(args.val_ann)
    if not ann_file.is_absolute():
        ann_file = workspace_root / ann_file

    if not img_folder.is_dir():
        raise FileNotFoundError(f"Val image folder not found: {img_folder}")
    if not ann_file.is_file():
        raise FileNotFoundError(f"Val annotation file not found: {ann_file}")

    return img_folder, ann_file


def run_standard_eval(args, workspace_root: Path, config_path: Path, model_path: Path) -> None:
    train_entrypoint = model_path / "train.py"
    if not train_entrypoint.exists():
        raise FileNotFoundError(f"D-FINE train.py not found: {train_entrypoint}")

    img_folder, ann_file = resolve_dataset_paths(args, workspace_root)

    updates = [
        *args.update,
        f"val_dataloader.dataset.img_folder={img_folder}",
        f"val_dataloader.dataset.ann_file={ann_file}",
    ]

    command = [
        sys.executable,
        "train.py",
        "-c",
        str(config_path),
        "-r",
        args.resume,
        "--test-only",
        "--update",
        *updates,
    ]

    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"resume: {args.resume}")
    print(f"workdir: {model_path}")
    print(f"val_images: {img_folder}")
    print(f"val_ann: {ann_file}")
    if args.devices:
        print(f"CUDA_VISIBLE_DEVICES: {args.devices}")
    print(f"command: {shlex.join(command)}")

    if args.dry_run:
        return

    subprocess.run(command, cwd=model_path, env=env, check=True)


def infer_eval_size(yaml_cfg: dict) -> int:
    ops = yaml_cfg.get("val_dataloader", {}).get("dataset", {}).get("transforms", {}).get("ops", [])
    for op in ops:
        if op.get("type") == "Resize":
            size = op.get("size")
            if size:
                return int(size[0])
    return DEFAULT_EVAL_SIZE


def run_sliced_eval(args, workspace_root: Path, config_path: Path, model_path: Path) -> None:
    import numpy as np
    import torch
    import torchvision
    from PIL import Image

    sys.path.insert(0, str(workspace_root))
    from tools.slice_dataset import generate_windows, parse_crop_size

    img_folder, ann_file = resolve_dataset_paths(args, workspace_root)

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION} (sliced)")
    print(f"config: {config_path}")
    print(f"resume: {args.resume}")
    print(f"workdir: {model_path}")
    print(f"slice_size: {args.slice_size}")
    print(f"overlap: width={args.overlap_width or args.overlap} height={args.overlap_height or args.overlap}")
    print(f"nms_iou: {args.nms_iou}")
    print(f"val_images: {img_folder}")
    print(f"val_ann: {ann_file}")

    if args.dry_run:
        return

    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    sys.path.insert(0, str(model_path))
    os.chdir(model_path)

    from src.core import YAMLConfig, yaml_utils
    from src.misc import dist_utils
    from src.solver import TASKS
    from src.solver.validator import Validator

    dist_utils.setup_distributed()

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict = yaml_utils.merge_dict(
        update_dict,
        {"val_dataloader": {"dataset": {"img_folder": str(img_folder), "ann_file": str(ann_file)}}},
    )

    cfg = YAMLConfig(str(config_path), resume=str(args.resume), **update_dict)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()

    device = solver.device
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    postprocessor = solver.postprocessor
    evaluator = solver.evaluator

    eval_size = args.eval_size or infer_eval_size(cfg.yaml_cfg)
    print(f"eval_size: {eval_size}x{eval_size}")

    slice_size = parse_crop_size(args.slice_size)
    overlap_width = args.overlap_width if args.overlap_width is not None else args.overlap
    overlap_height = args.overlap_height if args.overlap_height is not None else args.overlap

    dataset = solver.val_dataloader.dataset
    coco_gt = dataset.coco
    dataset_img_folder = Path(dataset.img_folder)

    gt_list = []
    preds_list = []
    image_ids = list(dataset.ids)

    for n, image_id in enumerate(image_ids, start=1):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = dataset_img_folder / image_info["file_name"]
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

            batch = []
            for window in windows:
                crop = image.crop((window.left, window.top, window.right, window.bottom))
                resized = crop.resize((eval_size, eval_size), Image.Resampling.LANCZOS)
                array = np.asarray(resized, dtype=np.float32) / 255.0
                batch.append(torch.from_numpy(array).permute(2, 0, 1))

            samples = torch.stack(batch, dim=0).to(device)
            orig_sizes = torch.tensor(
                [[window.width, window.height] for window in windows], dtype=torch.float32, device=device
            )

            with torch.no_grad():
                outputs = model(samples)
            window_results = postprocessor(outputs, orig_sizes)

            all_boxes, all_scores, all_labels = [], [], []
            for window, result in zip(windows, window_results):
                boxes = result["boxes"].clone()
                boxes[:, [0, 2]] += window.left
                boxes[:, [1, 3]] += window.top
                all_boxes.append(boxes)
                all_scores.append(result["scores"])
                all_labels.append(result["labels"])

            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)

            if len(windows) > 1 and boxes.shape[0] > 0:
                keep = torchvision.ops.batched_nms(boxes, scores, labels, args.nms_iou)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        boxes, scores, labels = boxes.cpu(), scores.cpu(), labels.cpu()
        evaluator.update({image_id: {"boxes": boxes, "scores": scores, "labels": labels}})

        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[image_id]))
        if anns:
            gt_boxes = torch.tensor(
                [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns],
                dtype=torch.float32,
            )
            gt_labels = torch.tensor([a["category_id"] for a in anns], dtype=torch.long)
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.long)

        gt_list.append({"boxes": gt_boxes, "labels": gt_labels})
        preds_list.append({"boxes": boxes, "labels": labels, "scores": scores})

        if n % 10 == 0 or n == len(image_ids):
            print(f"[{n}/{len(image_ids)}] {image_info['file_name']} ({len(windows)} slices)")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = Validator(gt_list, preds_list).compute_metrics()
    print("Metrics:", metrics)

    dist_utils.cleanup()


def main() -> None:
    args = parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    model_path = workspace_root / MODEL_DIR

    if args.sliced:
        run_sliced_eval(args, workspace_root, config_path, model_path)
    else:
        run_standard_eval(args, workspace_root, config_path, model_path)


if __name__ == "__main__":
    main()
