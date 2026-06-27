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
ACTION = "infer"


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--resume", required=True, help="Path to checkpoint (.pth).")
    parser.add_argument("--im-file", required=True, help="Path to image file.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it.")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    model_path = workspace_root / MODEL_DIR
    infer_script = model_path / "references" / "deploy" / "rtdetrv2_torch.py"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not infer_script.exists():
        raise FileNotFoundError(f"RTDETRv2 inference script not found: {infer_script}")

    command = [
        sys.executable,
        str(infer_script),
        "-c", str(config_path),
        "-r", args.resume,
        "--im-file", args.im_file,
        "--device", args.device,
    ]

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"workdir: {model_path}")
    print(f"command: {shlex.join(command)}")

    if args.dry_run:
        return

    subprocess.run(command, cwd=model_path, check=True)


if __name__ == "__main__":
    main()
