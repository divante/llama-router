"""Generate llama-server config with dual GPU/CPU entries per GGUF model.

For each .gguf file in the models directory, creates two aliases:
  - {stem}-gpu  (n_gpu_layers=999, fully offloaded)
  - {stem}-cpu  (n_gpu_layers=0, pure CPU)

Outputs:
  - config.json for llama-server's --config flag
  - model_sizes.json mapping stem -> file size in bytes (used by the router)

Usage:
    python config_gen.py /models /config/config.json
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

    slots: list[dict] = []
    model_sizes: dict[str, int] = {}

    for gguf in gguf_files:
        stem = gguf.stem
        model_path = str(gguf)
        file_size = gguf.stat().st_size
        model_sizes[stem] = file_size

        # GPU variant — full offload
        slots.append({
            "model": model_path,
            "model_alias": f"{stem}-gpu",
            "n_gpu_layers": 999,
        })

        # CPU variant — no GPU layers
        slots.append({
            "model": model_path,
            "model_alias": f"{stem}-cpu",
            "n_gpu_layers": 0,
        })

    config = {"slots": slots}
    output_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(slots)} model slots to {output_path}")

    sizes_path = output_path.parent / "model_sizes.json"
    sizes_path.write_text(json.dumps(model_sizes, indent=2) + "\n")
    print(f"Wrote model sizes to {sizes_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <models_dir> <output_config_path>", file=sys.stderr)
        sys.exit(1)

    generate_config(Path(sys.argv[1]), Path(sys.argv[2]))
