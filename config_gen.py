"""Generate llama-server preset INI from GGUF models.

Supports two modes (set via `mode` key in group_params.yaml):
  - cpu_only: one preset per model [{stem}] using "cpu" params (no suffix)
  - split:    two presets per model [{stem}-gpu] and [{stem}-cpu]

group_params.yaml defines arbitrary llama-server preset keys per group.
Any key valid in a --models-preset INI section can be used (corresponds
to CLI args without leading dashes, e.g. "batch-size", "threads",
"flash-attn", "ctx-size", "cache-type-k", etc.).

Outputs:
  - presets.ini for llama-server's --models-preset flag
  - model_sizes.json mapping stem -> file size in bytes (used by the router)

Split GGUF models (``-NNNNN-of-MMMMM`` filename parts) are grouped by their
shared stem: the set is exposed as ONE preset whose model path is part 00001
(llama.cpp loads the whole set from the first part), and model_sizes.json
records the sum of every part. The parts are never concatenated.

Usage:
    python config_gen.py /models /config/presets.ini [/path/to/group_params.yaml]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

DEFAULT_MODE = "split"

DEFAULT_GROUP_PARAMS = {
    "gpu": {"n-gpu-layers": 999},
    "cpu": {"n-gpu-layers": 0},
}


# e.g. "tiny-7b-00001-of-00003.gguf" -> ("tiny-7b", 1, 3)
SPLIT_PART_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d+)-of-(?P<total>\d+)\.gguf$")


def group_split_parts(gguf_files: list[Path]) -> list[list[Path]]:
    """Group split parts (-NNNNN-of-MMMMM) into ordered parts lists.

    Files matching the split pattern are grouped by shared stem and declared
    total, ordered by part number. Groups with a missing or duplicated part
    number raise ValueError. A group with declared total 1 is a regular model
    and is returned on its own. Files not matching the pattern are returned
    one per group, untouched.
    """
    groups: dict[tuple[str, int], dict[int, Path]] = defaultdict(dict)
    singletons: list[Path] = []

    for gguf in gguf_files:
        match = SPLIT_PART_RE.match(gguf.name)
        if not match:
            singletons.append([gguf])
            continue
        stem = match.group("stem")
        index = int(match.group("index"))
        total = int(match.group("total"))
        part_map = groups[(stem, total)]
        if index in part_map:
            raise ValueError(
                f"Duplicate split part {gguf.name}: part {index} of "
                f"{stem}-{index:05d}-of-{total:05d} conflicts with "
                f"{part_map[index].name}"
            )
        part_map[index] = gguf

    result: list[list[Path]] = []
    for (stem, total), part_map in sorted(groups.items()):
        missing = [i for i in range(1, total + 1) if i not in part_map]
        if missing:
            raise ValueError(
                f"Incomplete split model {stem}: declared {total} parts but "
                f"missing part number(s) {', '.join(f'{i:05d}' for i in missing)}"
            )
        result.append([part_map[i] for i in range(1, total + 1)])
    result.extend(singletons)
    # Preserve the historical filename-sorted output order, keyed by each
    # group's first part (which is the part the preset points at).
    result.sort(key=lambda group: group[0].name)
    return result


def load_group_params(path: Path | None) -> dict[str, dict[str, str | int]]:
    """Load group parameters from YAML, falling back to defaults."""
    if path and path.exists():
        params = yaml.safe_load(path.read_text())
        print(f"Loaded group params from {path}")
        return params

    # Check for group_params.yaml next to this script
    local = Path(__file__).parent / "group_params.yaml"
    if local.exists():
        params = yaml.safe_load(local.read_text())
        print(f"Loaded group params from {local}")
        return params

    print("Using default group params (gpu: ngl=999, cpu: ngl=0)")
    return DEFAULT_GROUP_PARAMS


def _append_section(
    lines: list[str], name: str, model_path: str, params: dict[str, str | int],
) -> None:
    """Append a single INI section."""
    lines.append(f"[{name}]")
    lines.append(f"model = {model_path}")
    for key, value in params.items():
        lines.append(f"{key} = {value}")
    lines.append("")


def generate_config(
    models_dir: Path,
    output_path: Path,
    group_params_path: Path | None = None,
) -> None:
    models_dir = models_dir.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    group_params = load_group_params(group_params_path)
    mode = group_params.get("mode", DEFAULT_MODE)
    gpu_params = group_params.get("gpu", DEFAULT_GROUP_PARAMS["gpu"])
    cpu_params = group_params.get("cpu", DEFAULT_GROUP_PARAMS["cpu"])

    print(f"Mode: {mode}")

    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        print(f"No .gguf files found in {models_dir}", file=sys.stderr)
        sys.exit(1)

    lines: list[str] = ["version = 1", ""]
    model_sizes: dict[str, int] = {}
    count = 0

    # Split parts (-NNNNN-of-MMMMM) form one group per model; everything
    # else is a one-file group, so a single loop handles both.
    for group in group_split_parts(gguf_files):
        # A split set is loaded by llama.cpp from part 00001, which holds a
        # pointer table for the whole set. A regular file is its own part one.
        first_part = group[0]
        stem = first_part.stem
        model_path = str(first_part)
        model_sizes[stem] = sum(part.stat().st_size for part in group)
        print(
            f"Grouped {len(group)} split part(s) into model {stem} "
            if len(group) > 1 else f"Model {stem}"
        )

        if mode == "cpu_only":
            _append_section(lines, stem, model_path, cpu_params)
            count += 1
        else:
            _append_section(lines, f"{stem}-gpu", model_path, gpu_params)
            count += 1
            _append_section(lines, f"{stem}-cpu", model_path, cpu_params)
            count += 1

    output_path.write_text("\n".join(lines))
    print(f"Wrote {count} model presets to {output_path}")

    # Print a sample section for verification
    sample_start = 2  # skip "version = 1\n"
    sample_end = lines.index("", sample_start + 1) if "" in lines[sample_start + 1:] else sample_start + 5
    print(f"Sample section:\n  " + "\n  ".join(lines[sample_start:sample_end + 1]))

    sizes_path = output_path.parent / "model_sizes.json"
    sizes_path.write_text(json.dumps(model_sizes, indent=2) + "\n")
    print(f"Wrote model sizes for {len(model_sizes)} models to {sizes_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            f"Usage: {sys.argv[0]} <models_dir> <output_preset_path> [group_params.yaml]",
            file=sys.stderr,
        )
        sys.exit(1)

    gp_path = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    generate_config(Path(sys.argv[1]), Path(sys.argv[2]), gp_path)
