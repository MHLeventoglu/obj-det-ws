#!/usr/bin/env python3
# Kullanim:
#
#   Checkpoint'ten TensorRT engine'e (onnx export + trtexec, varsayilan FP16):
#     python scripts/d-fine/deploy.py --resume output/dfine_.../best_stg2.pth
#
#   Farkli bir model config'i ile:
#     python scripts/d-fine/deploy.py --config configs/d-fine/dfine_hgnetv2_n_plane_44k.yml \
#         --resume output/dfine_hgnetv2_n_plane_44k/best_stg2.pth
#
#   Var olan bir ONNX dosyasindan dogrudan TensorRT engine'e (onnx export atlanir):
#     python scripts/d-fine/deploy.py --onnx model.onnx --output model.engine
#
#   FP32 engine (varsayilan FP16'yi kapatmak icin):
#     python scripts/d-fine/deploy.py --resume ... --no-fp16
#
#   trtexec'e ekstra argumanlar gecmek icin:
#     python scripts/d-fine/deploy.py --resume ... --trtexec-args --workspace=4096
#
#   Komutlari calistirmadan once gormek icin:
#     python scripts/d-fine/deploy.py --resume ... --dry-run
#
# Akis D-FINE'in kendi export/deploy talimatlarini izler: once
# tools/deployment/export_onnx.py ile ONNX'e, sonra trtexec ile TensorRT
# engine'e cevirir. trtexec, TensorRT kurulumunun bir parcasi olarak PATH'te
# bulunmalidir (--trtexec ile farkli bir binary yolu verilebilir).
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

MODEL_NAME = "D-FINE"
MODEL_DIR = "models/D-FINE"
DEFAULT_CONFIG = "configs/d-fine/dfine_hgnetv2_m_datasetv1.yml"
ACTION = "deploy"


def resolve_path(path: str, workspace_root: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = workspace_root / resolved
    return resolved


def export_onnx(args, workspace_root: Path, config_path: Path, model_path: Path, resume_path: Path) -> Path:
    export_script = model_path / "tools" / "deployment" / "export_onnx.py"
    if not export_script.exists():
        raise FileNotFoundError(f"D-FINE export_onnx.py not found: {export_script}")
    if not args.dry_run and not resume_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    onnx_path = resume_path.with_suffix(".onnx")

    command = [
        sys.executable,
        "tools/deployment/export_onnx.py",
        "-c", str(config_path),
        "-r", str(resume_path),
        "--check",
    ]

    print(f"command: {shlex.join(command)}")
    if not args.dry_run:
        subprocess.run(command, cwd=model_path, check=True)

    return onnx_path


def build_trt_engine(args, onnx_path: Path, output_path: Path) -> None:
    if not args.dry_run and not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    command = [args.trtexec, f"--onnx={onnx_path}", f"--saveEngine={output_path}"]
    if not args.no_fp16:
        command.append("--fp16")
    command.extend(args.trtexec_args)

    env = None
    if args.devices:
        import os

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    print(f"command: {shlex.join(command)}")
    if not args.dry_run:
        subprocess.run(command, env=env, check=True)


def main() -> None:
    parser = ArgumentParser(description=f"{MODEL_NAME} {ACTION} script")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--resume", help="Path to checkpoint (.pth) to export. Required unless --onnx is given.")
    parser.add_argument("--onnx", help="Use an existing ONNX file instead of exporting from --resume.")
    parser.add_argument("--output", help="Output TensorRT engine path (default: <onnx>.engine).")
    parser.add_argument("--no-fp16", action="store_true", help="Build an FP32 engine instead of the default FP16.")
    parser.add_argument("--trtexec", default="trtexec", help="Path to the trtexec binary.")
    parser.add_argument(
        "--trtexec-args",
        nargs="*",
        default=[],
        help="Extra raw trtexec arguments, for example: --workspace=4096 --minShapes=images:1x3x640x640",
    )
    parser.add_argument("--devices", help="CUDA_VISIBLE_DEVICES value for the trtexec build step.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    if not args.resume and not args.onnx:
        parser.error("--resume veya --onnx belirtilmeli.")

    workspace_root = Path(__file__).resolve().parents[2]
    config_path = resolve_path(args.config, workspace_root)
    model_path = workspace_root / MODEL_DIR

    print(f"model: {MODEL_NAME}")
    print(f"action: {ACTION}")
    print(f"config: {config_path}")
    print(f"workdir: {model_path}")

    if args.onnx:
        onnx_path = resolve_path(args.onnx, workspace_root)
        if not args.dry_run and not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    else:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        resume_path = resolve_path(args.resume, workspace_root)
        onnx_path = export_onnx(args, workspace_root, config_path, model_path, resume_path)

    output_path = resolve_path(args.output, workspace_root) if args.output else onnx_path.with_suffix(".engine")

    build_trt_engine(args, onnx_path, output_path)

    if not args.dry_run:
        print(f"engine: {output_path}")


if __name__ == "__main__":
    main()
