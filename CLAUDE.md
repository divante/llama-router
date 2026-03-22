# Project: llama-router

**LLM routing proxy** for llama.cpp's multi-model server. Translates OpenAI-compatible API requests into llama.cpp slot management.

## Files

- `router.py` — Main proxy server (FastAPI). Routes `/v1/chat/completions` and `/api/chat` to llama.cpp slots.
- `sanitizer.py` — Request/response sanitization
- `config_gen.py` — Generates llama.cpp `presets.ini` from model directory + group params
- `group_params.yaml` — Model group definitions (context size, GPU layers per group)
- `Dockerfile` — Container build
- `requirements.txt` — Python dependencies

## Running

Built and run via Docker Compose in `normandy-sr2/docker-composes/llm-compose.yaml`.

Key env vars:
- `LLAMA_SERVER_URL` — llama.cpp server address
- `MODEL_DIR` — Path to GGUF model files
- `VRAM_HEADROOM_MB` — Reserved VRAM buffer
- `FORCE_CPU_MODELS` — Comma-separated models to force CPU inference

## Sibling Repos

- `normandy-sr2/` — Compose files, agent configs
- `normandy-dispatcher/` — Task dispatcher
- `sr2/` — SR2 library
