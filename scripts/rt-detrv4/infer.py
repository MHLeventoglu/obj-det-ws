#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


MODEL_NAME = "RT-DETRv4"
MODEL_DIR = "models/RT-DETRv4"
DEFAULT_CONFIG = "configs/rt-detrv4/rtv4_hgnetv2_m_datasetv1.yml"
ACTION = "infer"


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    model_path = workspace_root / MODEL_DIR

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"workdir: {model_path}")

    # TODO: Add RT-DETRv4 inference command here.


if __name__ == "__main__":
    main()
