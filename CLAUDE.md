# Project: llama-router

**Multi-backend LLM routing proxy.** Routes OpenAI-compatible API requests to vLLM and/or llama.cpp backends based on model name.

## Architecture

```
Agents → llama-router (:11434)
           ├── VLLM_ROUTES match? → vLLM container(s)
           ├── FORCE_CPU_MODELS?  → llama-server (CPU)
           └── default            → llama-server (LLAMA_SERVER_URL)
```

Each vLLM container serves one model. The router maps model names to backend URLs via `VLLM_ROUTES`. Models not in the routing table fall through to llama-server.

## Files

- `router.py` — Main proxy server (FastAPI). Multi-backend routing, model-to-URL mapping, `/v1/capacity/check` for pre-spawn memory checks.
- `sanitizer.py` — Request/response sanitization
- `config_gen.py` — Generates llama.cpp `presets.ini` from model directory + group params. Respects `mode` from group_params.yaml.
- `group_params.yaml` — Model group definitions and routing mode (`cpu_only` or `split`)
- `Dockerfile` — Container build
- `requirements.txt` — Python dependencies

## Endpoints

- `POST /v1/chat/completions` — Proxied chat completions with model routing
- `POST /v1/completions` — Proxied completions
- `POST /v1/embeddings` — Always CPU-routed embeddings
- `GET /v1/models` — Merged model list from all backends (llama-server + all vLLM instances)
- `GET /v1/capacity/check?model={name}` — Memory-aware capacity check. Returns `can_serve`, `routing`, `reason`, `memory` info, and `loaded_models`. Used by Harbinger for spawn decisions.
- `GET /health` — Backend health proxy

## Running

Built and run via Docker Compose in `normandy-sr2/docker-composes/llm-compose.yaml`.

Key env vars:
- `ROUTING_MODE` — `cpu_only` (no suffixes, no VRAM checks) or `split` (GPU/CPU routing with -gpu/-cpu suffixes)
- `LLAMA_SERVER_URL` — llama.cpp server address (fallback backend)
- `VLLM_ROUTES` — Model-to-vLLM-backend routing table. Format: `model1=http://host1:port,model2=http://host2:port`. Models listed here are routed to the specified vLLM backend; all others go to `LLAMA_SERVER_URL`. One httpx client is created per unique URL.
- `MODEL_DIR` — Path to GGUF model files
- `VRAM_HEADROOM_MB` — Reserved VRAM buffer (split mode only)
- `FORCE_CPU_MODELS` — Comma-separated models to force CPU inference (split mode only)

Example `VLLM_ROUTES`:
```
VLLM_ROUTES=qwen3.5:27b=http://vllm-primary:8000,glm-4.7-flash=http://vllm-glm:8000
```

## Sibling Repos

- `normandy-sr2/` — Compose files, agent configs
- `harbinger/` — Task dispatcher
- `sr2/` — SR2 library

## V2 Product-Owner Build Guardrail

For a spec-defined, buildable task bead, follow the canonical v2 workflow
at `/data/obsidian/workflow/README.md`. Diego is product owner, not a
midpoint reviewer: run A → B → C → D and stop only for the five canonical
escalation triggers. `ACCEPTED` is Agent D automated acceptance, not a
claim that Diego reviewed the work.

Use v2 only when the project `CLAUDE.md` documents an exact full-suite
command and its mandates, and Agent D is independent. If any precondition
is missing, state the fallback and use per-step human review. Do not write
smoke-test runbooks for v2 work. Push only after acceptance; if integration
changes the accepted commit range, rerun acceptance.
