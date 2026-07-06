#!/usr/bin/env python3
# Kullanim:
#
#   Standart degerlendirme (D-FINE'in kendi tam-goruntu evaluate() akisi):
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
#   TensorRT engine ile degerlendirme (--resume yerine --engine, --config
#   gerekmez; girdi boyutu engine'in "images" binding'inden okunur):
#     python scripts/d-fine/test.py --engine output/dfine_.../best_stg2.engine \
#         -images datasets/datasetv1/val -ann datasets/datasetv1/val/val.json
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
# --resume ve --engine'den tam olarak biri verilmelidir; --engine su anda
# --sliced ile birlikte desteklenmez.
#
# Cikti metrikleri: precision, recall, f1, iou (D-FINE Validator, conf/IoU=0.5),
# mAP50:95, mAP50, mAP75, AR@1/10/100 (COCOeval), sinif bazinda precision,
# recall, mAP50, mAP50:95 (faster-coco-eval extended_metrics) ve saf model/engine
# forward-pass suresine dayali latency/FPS (data loading/postprocessing haric)
# - standart, sliced ve tensorrt modlarinda ayni formatta basilir.
import os
import sys
import time
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
    parser.add_argument("--resume", help="Path to checkpoint (.pth) to evaluate. Required unless --engine is given.")
    parser.add_argument(
        "--engine",
        help="Path to a TensorRT engine (.engine) to evaluate instead of a PyTorch checkpoint.",
    )
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

    args = parser.parse_args()

    if not args.resume and not args.engine:
        parser.error("--resume veya --engine belirtilmeli.")
    if args.resume and args.engine:
        parser.error("--resume ve --engine ayni anda verilemez.")
    if args.engine and args.sliced:
        parser.error("--engine su anda --sliced ile desteklenmiyor.")

    return args


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


def infer_eval_size(yaml_cfg: dict) -> int:
    ops = yaml_cfg.get("val_dataloader", {}).get("dataset", {}).get("transforms", {}).get("ops", [])
    for op in ops:
        if op.get("type") == "Resize":
            size = op.get("size")
            if size:
                return int(size[0])
    return DEFAULT_EVAL_SIZE


def print_class_wise_metrics(coco_eval) -> None:
    metrics = coco_eval.extended_metrics
    print("Class-wise mAP:")
    print(f"{'class':<15}{'precision':>12}{'recall':>12}{'mAP50':>12}{'mAP50:95':>12}")
    for row in metrics["class_map"]:
        print(
            f"{row['class']:<15}{row['precision']:>12.4f}{row['recall']:>12.4f}"
            f"{row['map@50']:>12.4f}{row['map@50:95']:>12.4f}"
        )


def print_latency_stats(total_time: float, total_images: int) -> None:
    if total_images == 0 or total_time <= 0:
        return
    avg_latency_ms = (total_time / total_images) * 1000
    fps = total_images / total_time
    print(f"latency: {avg_latency_ms:.2f} ms/image (model forward pass only)")
    print(f"fps: {fps:.2f}")


class LatencyTimer:
    def __init__(self, module, device) -> None:
        self._module = module
        self._device = device
        self.total_time = 0.0
        self.total_images = 0

    def eval(self):
        self._module.eval()
        return self

    def __call__(self, samples):
        import torch

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = self._module(samples)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self.total_time += time.perf_counter() - start
        self.total_images += samples.shape[0]
        return outputs


def build_solver(args, config_path: Path, model_path: Path, img_folder: Path, ann_file: Path):
    sys.path.insert(0, str(model_path))
    os.chdir(model_path)

    from src.core import YAMLConfig, yaml_utils
    from src.misc import dist_utils
    from src.solver import TASKS

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
    return solver


def run_standard_eval(args, workspace_root: Path, config_path: Path, model_path: Path) -> None:
    img_folder, ann_file = resolve_dataset_paths(args, workspace_root)

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"resume: {args.resume}")
    print(f"workdir: {model_path}")
    print(f"val_images: {img_folder}")
    print(f"val_ann: {ann_file}")

    if args.dry_run:
        return

    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    solver = build_solver(args, config_path, model_path, img_folder, ann_file)

    from src.misc import dist_utils
    from src.solver.det_engine import evaluate

    module = solver.ema.module if solver.ema else solver.model
    timer = LatencyTimer(module, solver.device)
    _, coco_evaluator = evaluate(
        timer,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device,
        epoch=-1,
        use_wandb=False,
    )

    print_class_wise_metrics(coco_evaluator.coco_eval["bbox"])
    print_latency_stats(timer.total_time, timer.total_images)
    dist_utils.cleanup()


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

    solver = build_solver(args, config_path, model_path, img_folder, ann_file)

    from src.misc import dist_utils
    from src.solver.validator import Validator

    device = solver.device
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    postprocessor = solver.postprocessor
    evaluator = solver.evaluator

    eval_size = args.eval_size or infer_eval_size(solver.cfg.yaml_cfg)
    print(f"eval_size: {eval_size}x{eval_size}")

    slice_size = parse_crop_size(args.slice_size)
    overlap_width = args.overlap_width if args.overlap_width is not None else args.overlap
    overlap_height = args.overlap_height if args.overlap_height is not None else args.overlap

    dataset = solver.val_dataloader.dataset
    coco_gt = dataset.coco
    dataset_img_folder = Path(dataset.img_folder)

    gt_list = []
    preds_list = []
    predictions = {}
    image_ids = list(dataset.ids)

    total_time = 0.0
    total_images = 0

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

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(samples)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            total_images += 1

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
        predictions[image_id] = {"boxes": boxes, "scores": scores, "labels": labels}

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

    # A single update() call is required (rather than one per image): FasterCocoEvaluator.update()
    # replaces self.coco_eval[...].cocoDt on every call instead of merging, so extended_metrics()'s
    # per-class precision/recall (which reads cocoDt directly) would otherwise only reflect the last
    # image processed - collapsing to a near-empty table whenever that image had few/no detections.
    evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = Validator(gt_list, preds_list).compute_metrics()
    print("Metrics:", metrics)
    print_class_wise_metrics(evaluator.coco_eval["bbox"])
    print_latency_stats(total_time, total_images)

    dist_utils.cleanup()


def run_trt_eval(args, workspace_root: Path, engine_path: Path, model_path: Path) -> None:
    img_folder, ann_file = resolve_dataset_paths(args, workspace_root)

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION} (tensorrt)")
    print(f"engine: {engine_path}")
    print(f"val_images: {img_folder}")
    print(f"val_ann: {ann_file}")

    if args.dry_run:
        return

    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    import numpy as np
    import torch
    from faster_coco_eval import COCO
    from faster_coco_eval.utils.pytorch import FasterCocoEvaluator
    from PIL import Image

    sys.path.insert(0, str(model_path))
    from src.solver.validator import Validator

    sys.path.insert(0, str(model_path / "tools" / "inference"))
    from trt_inf import TRTInference

    model = TRTInference(str(engine_path), device="cuda:0")
    images_shape = model.bindings["images"].shape
    eval_h, eval_w = int(images_shape[2]), int(images_shape[3])
    print(f"eval_size: {eval_w}x{eval_h}")

    coco_gt = COCO(str(ann_file))
    evaluator = FasterCocoEvaluator(coco_gt, ["bbox"])

    gt_list = []
    preds_list = []
    predictions = {}
    total_time = 0.0
    total_images = 0

    image_ids = coco_gt.getImgIds()
    for n, image_id in enumerate(image_ids, start=1):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = img_folder / image_info["file_name"]

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            width, height = image.size
            resized = image.resize((eval_w, eval_h), Image.Resampling.LANCZOS)

        array = np.asarray(resized, dtype=np.float32) / 255.0
        images = torch.from_numpy(array).permute(2, 0, 1).contiguous()[None].to(model.device)
        orig_sizes = torch.tensor([[width, height]]).to(model.device)

        model.synchronize()
        start = time.perf_counter()
        output = model({"images": images, "orig_target_sizes": orig_sizes})
        model.synchronize()
        total_time += time.perf_counter() - start
        total_images += 1

        boxes = output["boxes"][0].cpu()
        scores = output["scores"][0].cpu()
        labels = output["labels"][0].cpu()

        predictions[image_id] = {"boxes": boxes, "scores": scores, "labels": labels}

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

        if n <= 3:
            print(f"debug[{n}]: image={image_info['file_name']} orig_size(w,h)=({width},{height})")
            print(f"debug[{n}]: pred boxes dtype={boxes.dtype} n={boxes.shape[0]} sample={boxes[:3].tolist()}")
            print(f"debug[{n}]: pred scores dtype={scores.dtype} sample={scores[:3].tolist()}")
            print(f"debug[{n}]: pred labels dtype={labels.dtype} sample={labels[:3].tolist()}")
            print(f"debug[{n}]: gt boxes={gt_boxes.tolist()} gt labels={gt_labels.tolist()}")

        if n % 10 == 0 or n == len(image_ids):
            print(f"[{n}/{len(image_ids)}] {image_info['file_name']}")

    # A single update() call is required (rather than one per image): FasterCocoEvaluator.update()
    # replaces self.coco_eval[...].cocoDt on every call instead of merging, so extended_metrics()'s
    # per-class precision/recall (which reads cocoDt directly) would otherwise only reflect the last
    # image processed - collapsing to a near-empty table whenever that image had few/no detections.
    evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = Validator(gt_list, preds_list).compute_metrics()
    print("Metrics:", metrics)
    print_class_wise_metrics(evaluator.coco_eval["bbox"])
    print_latency_stats(total_time, total_images)


def main() -> None:
    args = parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    model_path = workspace_root / MODEL_DIR

    if args.engine:
        engine_path = Path(args.engine)
        if not engine_path.is_absolute():
            engine_path = workspace_root / engine_path
        run_trt_eval(args, workspace_root, engine_path, model_path)
        return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    resume_path = Path(args.resume)
    if not resume_path.is_absolute():
        resume_path = workspace_root / resume_path
    args.resume = str(resume_path)

    if args.sliced:
        run_sliced_eval(args, workspace_root, config_path, model_path)
    else:
        run_standard_eval(args, workspace_root, config_path, model_path)


if __name__ == "__main__":
    main()
