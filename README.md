# llama-router

VRAM-aware routing proxy for llama-server. Drop-in replacement for Ollama on port 11434.

## How It Works

Each GGUF model is registered in llama-server twice — a `-gpu` variant (fully offloaded) and a `-cpu` variant (no GPU layers). On every request, the router:

1. Reads free VRAM from AMD sysfs (`/sys/class/drm/card0/device/mem_info_vram_*`)
2. Compares against `model_file_size + headroom`
3. Routes to `-gpu` if it fits, `-cpu` otherwise
4. Embeddings always go to CPU (preserves VRAM for generation)

Clients send plain model names (e.g. `qwen3.5-35b`). The router appends `-gpu` or `-cpu` transparently.

## Services

| Service | Role | Notes |
|---------|------|-------|
| `llama-config-gen` | Init container | Scans `/models/*.gguf`, writes config JSON + model sizes, exits |
| `llama-server` | llama.cpp server | ROCm GPU, reads generated config |
| `llama-router` | FastAPI proxy | Port 11434, VRAM-aware routing |

## Quick Start

```bash
# From repo root, on the server
docker compose -f docker-composes/llm-compose.yaml up -d

# Check models are loaded
curl http://localhost:11434/v1/models | jq

# Send a request (router picks GPU or CPU automatically)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Check health
curl http://localhost:11434/health

# View routing decisions
docker compose -f docker-composes/llm-compose.yaml logs -f llama-router
```

## API Endpoints

All OpenAI-compatible — same paths agents already use.

| Endpoint | Method | Routing |
|----------|--------|---------|
| `/v1/chat/completions` | POST | VRAM-aware GPU/CPU |
| `/v1/completions` | POST | VRAM-aware GPU/CPU |
| `/v1/embeddings` | POST | Always CPU |
| `/v1/models` | GET | Deduplicated list (no `-gpu`/`-cpu` suffixes) |
| `/health` | GET | Passthrough to llama-server |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVER_URL` | `http://llama-server:8080` | llama-server backend URL |
| `MODEL_DIR` | `/models` | Directory containing GGUF files |
| `VRAM_HEADROOM_MB` | `1024` | Extra VRAM buffer (MB) beyond model size before choosing GPU |
| `MODEL_SIZES_PATH` | `/config/model_sizes.json` | Path to generated model sizes file |
| `VRAM_TOTAL_PATH` | `/sys/class/drm/card0/device/mem_info_vram_total` | sysfs path for total VRAM |
| `VRAM_USED_PATH` | `/sys/class/drm/card0/device/mem_info_vram_used` | sysfs path for used VRAM |

## Adding Models

1. Place the `.gguf` file in `/data/ai-stack/llm/models/` on the server
2. Restart the stack (config-gen re-scans on startup):
   ```bash
   docker compose -f docker-composes/llm-compose.yaml up -d --force-recreate
   ```
3. The model is immediately available by its filename stem (e.g. `my-model-7b.gguf` → `my-model-7b`)
