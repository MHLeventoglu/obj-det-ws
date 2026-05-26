#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat


MODEL_NAME = "RF-DETR"
DEFAULT_CONFIG = "configs/rf-detr/rfdetr_medium_datasetv1.yaml"
ACTION = "train"


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read RF-DETR config files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(workspace_root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = workspace_root / path
    return str(path)


def get_model_class(variant: str):
    import rfdetr

    variants = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "base": "RFDETRBase",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
        "xlarge": "RFDETRXLarge",
    }
    try:
        class_name = variants[variant.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported RF-DETR variant: {variant}") from exc

    try:
        return getattr(rfdetr, class_name)
    except AttributeError as exc:
        raise RuntimeError(f"Installed rfdetr package does not expose {class_name}.") from exc


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--dataset-dir", help="Override dataset directory.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--grad-accum-steps", type=int, help="Override gradient accumulation steps.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--device", help="Override training device if supported by installed RF-DETR.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without running.")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_yaml(config_path)
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    train_cfg = dict(cfg.get("train", {}))

    if args.dataset_dir:
        dataset_cfg["dir"] = args.dataset_dir
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        train_cfg["grad_accum_steps"] = args.grad_accum_steps
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir
    if args.device:
        train_cfg["device"] = args.device

    train_cfg["dataset_dir"] = resolve_path(workspace_root, dataset_cfg.get("dir"))
    train_cfg["output_dir"] = resolve_path(workspace_root, train_cfg.get("output_dir"))

    train_kwargs = {key: value for key, value in train_cfg.items() if value is not None}
    model_kwargs = {}
    pretrain_weights = resolve_path(workspace_root, model_cfg.get("pretrain_weights"))
    if pretrain_weights:
        model_kwargs["pretrain_weights"] = pretrain_weights

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"variant: {model_cfg.get('variant', 'medium')}")
    print(f"classes: {dataset_cfg.get('classes')}")
    print(f"train_kwargs:\n{pformat(train_kwargs)}")

    if args.dry_run:
        return

    model_cls = get_model_class(model_cfg.get("variant", "medium"))
    model = model_cls(**model_kwargs)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
