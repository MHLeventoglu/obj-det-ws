#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


MODEL_NAME = "RT-DETRv2"
MODEL_DIR = "models/RT-DETR/rtdetrv2_pytorch"
DEFAULT_CONFIG = "configs/rt-detr/rtdetrv2_r50vd_l_datasetv1_sliced_1080_2crop.yml"
ACTION = "test"


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--resume", required=True, help="Path to checkpoint to evaluate.")
    parser.add_argument("--devices", help="CUDA_VISIBLE_DEVICES value, for example: 0 or 0,1,2,3.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it.")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    model_path = workspace_root / MODEL_DIR
    train_entrypoint = model_path / "tools" / "train.py"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not train_entrypoint.exists():
        raise FileNotFoundError(f"RTDETRv2 tools/train.py not found: {train_entrypoint}")

    command = [sys.executable, "tools/train.py", "-c", str(config_path), "-r", args.resume, "--test-only"]

    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"workdir: {model_path}")
    if args.devices:
        print(f"CUDA_VISIBLE_DEVICES: {args.devices}")
    print(f"command: {shlex.join(command)}")

    if args.dry_run:
        return

    subprocess.run(command, cwd=model_path, env=env, check=True)


if __name__ == "__main__":
    main()
