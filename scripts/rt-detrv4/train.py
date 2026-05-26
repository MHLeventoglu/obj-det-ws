#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


MODEL_NAME = "RT-DETRv4"
MODEL_DIR = "models/RT-DETRv4"
DEFAULT_CONFIG = "configs/rt-detrv4/rtv4_hgnetv2_m_datasetv1.yml"
ACTION = "train"


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--devices", help="CUDA_VISIBLE_DEVICES value, for example: 0 or 0,1,2,3.")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Number of training processes.")
    parser.add_argument("--master-port", type=int, default=7777, help="torchrun master port.")
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument("--resume", help="Resume training from a checkpoint.")
    parser.add_argument("--tuning", help="Fine-tune from a checkpoint.")
    parser.add_argument("--output-dir", help="Override RT-DETRv4 output_dir.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it.")
    parser.add_argument(
        "--update",
        nargs="*",
        default=[],
        help="RT-DETRv4 config overrides, for example: train_dataloader.total_batch_size=16",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path

    model_path = workspace_root / MODEL_DIR
    train_entrypoint = model_path / "train.py"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not train_entrypoint.exists():
        raise FileNotFoundError(f"RT-DETRv4 train.py not found: {train_entrypoint}")

    if args.nproc_per_node > 1:
        command = [
            "torchrun",
            f"--master_port={args.master_port}",
            f"--nproc_per_node={args.nproc_per_node}",
            "train.py",
        ]
    else:
        command = [sys.executable, "train.py"]

    command.extend(["-c", str(config_path), "--seed", str(args.seed)])

    if not args.no_amp:
        command.append("--use-amp")
    if args.resume:
        command.extend(["--resume", args.resume])
    if args.tuning:
        command.extend(["--tuning", args.tuning])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.update:
        command.extend(["--update", *args.update])

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
