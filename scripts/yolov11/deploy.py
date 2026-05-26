#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


MODEL_NAME = "YOLOv11"
DEFAULT_CONFIG = "configs/yolov11/yolo11m_datasetv1.yaml"
ACTION = "deploy"


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"workdir: {workspace_root}")

    # TODO: Add YOLOv11 deployment/export command here.


if __name__ == "__main__":
    main()
