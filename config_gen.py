"""Generate llama-server preset INI with dual GPU/CPU entries per GGUF model.

For each .gguf file in the models directory, creates two preset sections:
  - [{stem}-gpu]  (n-gpu-layers=999, fully offloaded)
  - [{stem}-cpu]  (n-gpu-layers=0, pure CPU)

Both point to the same model file path via explicit `model = /path`.
Uses --models-preset only (no --models-dir), so these are the only models
llama-server knows about.

Outputs:
  - presets.ini for llama-server's --models-preset flag
  - model_sizes.json mapping stem -> file size in bytes (used by the router)

Usage:
    python config_gen.py /models /config/presets.ini
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def generate_config(models_dir: Path, output_path: Path) -> None:
    models_dir = models_dir.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        print(f"No .gguf files found in {models_dir}", file=sys.stderr)
        sys.exit(1)

    lines: list[str] = ["version = 1", ""]
    model_sizes: dict[str, int] = {}
    count = 0

    for gguf in gguf_files:
        stem = gguf.stem
        model_path = str(gguf)
        file_size = gguf.stat().st_size
        model_sizes[stem] = file_size

        # GPU variant — full offload
        lines.append(f"[{stem}-gpu]")
        lines.append(f"model = {model_path}")
        lines.append("n-gpu-layers = 999")
        lines.append("")
        count += 1

        # CPU variant — no GPU layers
        lines.append(f"[{stem}-cpu]")
        lines.append(f"model = {model_path}")
        lines.append("n-gpu-layers = 0")
        lines.append("")
        count += 1

    output_path.write_text("\n".join(lines))
    print(f"Wrote {count} model presets to {output_path}")

    sizes_path = output_path.parent / "model_sizes.json"
    sizes_path.write_text(json.dumps(model_sizes, indent=2) + "\n")
    print(f"Wrote model sizes to {sizes_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <models_dir> <output_preset_path>", file=sys.stderr)
        sys.exit(1)

    generate_config(Path(sys.argv[1]), Path(sys.argv[2]))
