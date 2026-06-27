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
ACTION = "deploy"


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--resume", required=True, help="Path to checkpoint (.pth).")
    parser.add_argument("--output", help="Output ONNX file path (default: <checkpoint>.onnx).")
    parser.add_argument("--check", action="store_true", help="Verify ONNX output after export.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it.")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    model_path = workspace_root / MODEL_DIR
    export_script = model_path / "tools" / "export_onnx.py"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not export_script.exists():
        raise FileNotFoundError(f"RTDETRv2 export script not found: {export_script}")

    command = [
        sys.executable,
        "tools/export_onnx.py",
        "-c", str(config_path),
        "-r", args.resume,
    ]
    if args.output:
        command.extend(["--file-name", args.output])
    if args.check:
        command.append("--check")

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
