#!/usr/bin/env python3
from argparse import ArgumentParser
import os
from pathlib import Path
from pprint import pformat


MODEL_NAME = "YOLOv11"
DEFAULT_CONFIG = "configs/yolov11/yolo11m_datasetv1.yaml"
ACTION = "train"


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YOLOv11 config files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(workspace_root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = workspace_root / path
    return str(path)


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--model", help="Override YOLO model path/name.")
    parser.add_argument("--data", help="Override YOLO data YAML path.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--device", help="Override device, for example: 0 or cpu.")
    parser.add_argument("--workers", type=int, help="Override dataloader worker count.")
    parser.add_argument("--project", help="Override output project directory.")
    parser.add_argument("--name", help="Override run name.")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Ultralytics Weights & Biases logging.")
    parser.add_argument("--wandb-project", help="Weights & Biases project name.")
    parser.add_argument("--wandb-name", help="Weights & Biases run name.")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity/team name.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without running.")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_yaml(config_path)
    model_name = args.model or cfg.get("model", "yolo11m.pt")
    data_path = args.data or cfg.get("data")
    train_cfg = dict(cfg.get("train", {}))

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.imgsz is not None:
        train_cfg["imgsz"] = args.imgsz
    if args.batch is not None:
        train_cfg["batch"] = args.batch
    if args.device:
        train_cfg["device"] = args.device
    if args.workers is not None:
        train_cfg["workers"] = args.workers
    if args.project:
        train_cfg["project"] = args.project
    if args.name:
        train_cfg["name"] = args.name

    train_cfg["data"] = resolve_path(workspace_root, data_path)
    if train_cfg.get("project"):
        train_cfg["project"] = resolve_path(workspace_root, train_cfg["project"])

    train_kwargs = {key: value for key, value in train_cfg.items() if value is not None}

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"yolo_model: {model_name}")
    print(f"train_kwargs:\n{pformat(train_kwargs)}")
    if args.use_wandb:
        print(
            "wandb:\n"
            + pformat(
                {
                    "project": args.wandb_project,
                    "name": args.wandb_name or train_cfg.get("name"),
                    "entity": args.wandb_entity or None,
                }
            )
        )

    if args.dry_run:
        return

    if args.use_wandb:
        os.environ.pop("WANDB_DISABLED", None)
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_name:
            os.environ["WANDB_NAME"] = args.wandb_name
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

        from ultralytics.utils import SETTINGS

        SETTINGS.update({"wandb": True})

        import wandb

        if not wandb.run:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or train_cfg.get("name"),
                entity=args.wandb_entity or None,
                config=train_kwargs,
            )

    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
