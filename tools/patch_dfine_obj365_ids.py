#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


def patch_solver(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    init_patch = (
        '        yaml_cfg = getattr(cfg, "yaml_cfg", {})\n'
        '        custom_obj365_ids = yaml_cfg.get("obj365_ids")\n'
        "        if custom_obj365_ids is not None:\n"
        "            self.obj365_ids = [int(item) for item in custom_obj365_ids]\n"
        "\n"
        '        custom_pretrain_class_ids = yaml_cfg.get("pretrain_class_ids")\n'
        "        self.pretrain_class_ids = None\n"
        "        self.pretrain_class_id_offset = 1\n"
        "        if custom_pretrain_class_ids is not None:\n"
        "            self.pretrain_class_ids = [int(item) for item in custom_pretrain_class_ids]\n"
        '            self.pretrain_class_id_offset = int(yaml_cfg.get("pretrain_class_id_offset", 0))\n'
    )
    old_init_patch = (
        '        custom_obj365_ids = getattr(cfg, "yaml_cfg", {}).get("obj365_ids")\n'
        "        if custom_obj365_ids is not None:\n"
        "            self.obj365_ids = [int(item) for item in custom_obj365_ids]\n"
    )
    map_class_weights = (
        "    def map_class_weights(self, cur_tensor, pretrain_tensor):\n"
        '        """Map class weights from a pretrain model to the current class order."""\n'
        "        if pretrain_tensor.size() == cur_tensor.size():\n"
        "            return pretrain_tensor\n"
        "\n"
        "        adjusted_tensor = cur_tensor.clone()\n"
        "        adjusted_tensor.requires_grad = False\n"
        "\n"
        '        class_ids = getattr(self, "pretrain_class_ids", None)\n'
        '        class_id_offset = getattr(self, "pretrain_class_id_offset", 1)\n'
        "        if class_ids is None:\n"
        "            class_ids = self.obj365_ids\n"
        "            class_id_offset = 1\n"
        "\n"
        "        if pretrain_tensor.size(0) > cur_tensor.size(0):\n"
        "            if len(class_ids) < cur_tensor.size(0):\n"
        "                return None\n"
        "            for current_id, pretrain_id in enumerate(class_ids[: cur_tensor.size(0)]):\n"
        "                source_id = int(pretrain_id) + int(class_id_offset)\n"
        "                if source_id < 0 or source_id >= pretrain_tensor.size(0):\n"
        "                    return None\n"
        "                adjusted_tensor[current_id] = pretrain_tensor[source_id]\n"
        "        else:\n"
        "            if len(class_ids) < pretrain_tensor.size(0):\n"
        "                return None\n"
        "            for source_id, current_id in enumerate(class_ids[: pretrain_tensor.size(0)]):\n"
        "                target_id = int(current_id) + int(class_id_offset)\n"
        "                if target_id < 0 or target_id >= adjusted_tensor.size(0):\n"
        "                    return None\n"
        "                adjusted_tensor[target_id] = pretrain_tensor[source_id]\n"
        "\n"
        "        return adjusted_tensor\n"
    )

    changed = False
    if init_patch not in text:
        if old_init_patch in text:
            text = text.replace(old_init_patch, init_patch, 1)
        else:
            needle = "        ]\n\n    def _setup(self):"
            replacement = f"        ]\n{init_patch}\n    def _setup(self):"
            if needle not in text:
                raise ValueError(f"Could not find obj365_ids insertion point in {path}")
            text = text.replace(needle, replacement, 1)
        changed = True

    if map_class_weights not in text:
        start = text.find("    def map_class_weights(self, cur_tensor, pretrain_tensor):")
        end = text.find("\n\n    def fit(self):", start)
        if start == -1 or end == -1:
            raise ValueError(f"Could not find map_class_weights in {path}")
        text = f"{text[:start]}{map_class_weights}{text[end:]}"
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"patched: {path}")
    else:
        print(f"already patched: {path}")


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
