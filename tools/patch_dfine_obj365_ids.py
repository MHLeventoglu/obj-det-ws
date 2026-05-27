#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


def patch_solver(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    patch = (
        '        custom_obj365_ids = getattr(cfg, "yaml_cfg", {}).get("obj365_ids")\n'
        "        if custom_obj365_ids is not None:\n"
        "            self.obj365_ids = [int(item) for item in custom_obj365_ids]\n"
    )
    if patch in text:
        print(f"already patched: {path}")
        return

    needle = "        ]\n\n    def _setup(self):"
    replacement = f"        ]\n{patch}\n    def _setup(self):"
    if needle not in text:
        raise ValueError(f"Could not find obj365_ids insertion point in {path}")

    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print(f"patched: {path}")


def main() -> None:
    parser = ArgumentParser(
        description="Patch D-FINE BaseSolver to read obj365_ids from YAML config."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="models/D-FINE/src/solver/_solver.py",
        help="Path to D-FINE src/solver/_solver.py.",
    )
    args = parser.parse_args()
    patch_solver(Path(args.path))


if __name__ == "__main__":
    main()
