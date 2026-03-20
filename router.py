"""VRAM-aware LLM router for llama-server.

Drop-in replacement for Ollama on port 11434. Routes requests to GPU or CPU
model variants based on available VRAM, read from AMD sysfs.

Routing priority:
1. FORCE_CPU_MODELS — always CPU
2. GPU variant already loaded in llama-server — route to GPU (skip VRAM check)
3. Enough free VRAM for the model + headroom — route to GPU
4. Otherwise — route to CPU
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

from sanitizer import sanitize_response
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

logger = logging.getLogger("llama-router")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
VRAM_HEADROOM_MB = int(os.environ.get("VRAM_HEADROOM_MB", "1024"))
VRAM_TOTAL_PATH = os.environ.get(
    "VRAM_TOTAL_PATH", "/sys/class/drm/card0/device/mem_info_vram_total"
)
VRAM_USED_PATH = os.environ.get(
    "VRAM_USED_PATH", "/sys/class/drm/card0/device/mem_info_vram_used"
)
MODEL_SIZES_PATH = Path(os.environ.get("MODEL_SIZES_PATH", "/config/model_sizes.json"))
FORCE_CPU_MODELS: set[str] = set(
    m.strip() for m in os.environ.get("FORCE_CPU_MODELS", "").split(",") if m.strip()
)
LOADED_MODELS_TTL_SECONDS = int(os.environ.get("LOADED_MODELS_TTL_SECONDS", "5"))

# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def _read_sysfs(path: str) -> int | None:
    """Read an integer value from a sysfs file. Returns None if unavailable."""
    try:
        return int(Path(path).read_text().strip())
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def get_vram_free_bytes() -> int | None:
    """Return free VRAM in bytes, or None if sysfs is unavailable."""
    total = _read_sysfs(VRAM_TOTAL_PATH)
    used = _read_sysfs(VRAM_USED_PATH)
    if total is None or used is None:
        return None
    return total - used


# ---------------------------------------------------------------------------
# Model size lookup
# ---------------------------------------------------------------------------

_model_sizes: dict[str, int] = {}


def _load_model_sizes() -> None:
    global _model_sizes
    if MODEL_SIZES_PATH.exists():
        _model_sizes = json.loads(MODEL_SIZES_PATH.read_text())
        logger.info("Loaded model sizes for %d models", len(_model_sizes))
    else:
        logger.warning("Model sizes file not found at %s — falling back to filesystem", MODEL_SIZES_PATH)
        for gguf in MODEL_DIR.glob("*.gguf"):
            _model_sizes[gguf.stem] = gguf.stat().st_size
        if _model_sizes:
            logger.info("Scanned %d models from %s", len(_model_sizes), MODEL_DIR)


def get_model_size(stem: str) -> int | None:
    """Return model file size in bytes, or None if unknown."""
    return _model_sizes.get(stem)


# ---------------------------------------------------------------------------
# Loaded model tracking
# ---------------------------------------------------------------------------

_loaded_gpu_models: set[str] = set()
_loaded_models_updated_at: float = 0.0


async def _refresh_loaded_models() -> None:
    """Query llama-server /models endpoint and cache loaded GPU model stems."""
    global _loaded_gpu_models, _loaded_models_updated_at

    now = time.monotonic()
    if now - _loaded_models_updated_at < LOADED_MODELS_TTL_SECONDS:
        return

    try:
        resp = await _client.get("/models", timeout=5.0)
        data = resp.json()
    except Exception as exc:
        logger.warning("Failed to query /models for loaded status: %s", exc)
        return

    loaded: set[str] = set()
    for model in data.get("data", []):
        model_id = model.get("id", "")
        status = model.get("status", {}).get("value", "")
        if status == "loaded" and model_id.endswith("-gpu"):
            loaded.add(_strip_suffix(model_id))

    _loaded_gpu_models = loaded
    _loaded_models_updated_at = now

    if loaded:
        logger.debug("Loaded GPU models: %s", loaded)


def is_gpu_model_loaded(stem: str) -> bool:
    """Check if the GPU variant of a model is currently loaded."""
    return stem in _loaded_gpu_models


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _strip_suffix(model: str) -> str:
    """Strip -gpu or -cpu suffix to get the base model stem."""
    if model.endswith("-gpu") or model.endswith("-cpu"):
        return model.rsplit("-", 1)[0]
    return model


def _normalize_model(model: str) -> str:
    if model.startswith("openai/"):
        return model[7:]
    return model


async def route_model(requested_model: str) -> str:
    """Decide GPU or CPU variant for the requested model.

    Priority:
    1. FORCE_CPU_MODELS — always CPU
    2. GPU variant already loaded — route to GPU (no VRAM check needed)
    3. Enough free VRAM — route to GPU
    4. Otherwise — route to CPU
    """
    stem = _strip_suffix(requested_model)

    # 1. Forced CPU
    if stem in FORCE_CPU_MODELS:
        logger.info("Forced CPU (config): %s", stem)
        return f"{stem}-cpu"

    # 2. GPU variant already loaded — skip VRAM check
    await _refresh_loaded_models()
    if is_gpu_model_loaded(stem):
        logger.info("GPU (already loaded): %s", stem)
        return f"{stem}-gpu"

    # 3. VRAM-based routing for models not currently loaded
    model_size = get_model_size(stem)
    vram_free = get_vram_free_bytes()
    headroom = VRAM_HEADROOM_MB * 1024 * 1024

    if vram_free is None:
        logger.warning(
            "VRAM sysfs unavailable — routing %s to CPU", stem
        )
        return f"{stem}-cpu"

    if model_size is None:
        logger.warning(
            "Unknown model size for %s — routing to CPU (vram_free=%dMB)",
            stem, vram_free // (1024 * 1024),
        )
        return f"{stem}-cpu"

    vram_free_mb = vram_free // (1024 * 1024)
    model_size_mb = model_size // (1024 * 1024)
    needed_mb = (model_size + headroom) // (1024 * 1024)

    if vram_free >= model_size + headroom:
        logger.info(
            "GPU: %s (model=%dMB, vram_free=%dMB, headroom=%dMB)",
            stem, model_size_mb, vram_free_mb, VRAM_HEADROOM_MB,
        )
        return f"{stem}-gpu"
    else:
        logger.info(
            "CPU: %s (model=%dMB, need=%dMB, vram_free=%dMB)",
            stem, model_size_mb, needed_mb, vram_free_mb,
        )
        return f"{stem}-cpu"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="llama-router")
_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(base_url=LLAMA_SERVER_URL, timeout=httpx.Timeout(600.0))
    _load_model_sizes()
    logger.info(
        "Router started — backend=%s, headroom=%dMB, models=%d, force_cpu=%s, loaded_ttl=%ds",
        LLAMA_SERVER_URL, VRAM_HEADROOM_MB, len(_model_sizes),
        FORCE_CPU_MODELS or "(none)", LOADED_MODELS_TTL_SECONDS,
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client:
        await _client.aclose()


# -- Proxy helpers ----------------------------------------------------------

async def _proxy_stream(upstream: httpx.Response):
    """Yield chunks from an upstream streaming response."""
    async for chunk in upstream.aiter_bytes():
        yield chunk


async def _proxy_request(
    request: Request,
    path: str,
    route: bool = True,
    force_cpu: bool = False,
    strip_nulls: bool = False,
) -> StreamingResponse | JSONResponse:
    """Forward a request to llama-server, optionally routing the model."""
    body = await request.json()

    if strip_nulls:
        body = {k: v for k, v in body.items() if v is not None}

    if route and "model" in body:
        original = body["model"]
        normalized = _normalize_model(original)
        if force_cpu:
            stem = _strip_suffix(normalized)
            body["model"] = f"{stem}-cpu"
            logger.info("Forced CPU: %s -> %s", original, body["model"])
        else:
            body["model"] = await route_model(normalized)

    is_stream = body.get("stream", False)

    try:
        upstream = await _client.send(
            _client.build_request(
                "POST",
                path,
                json=body,
                headers={"content-type": "application/json"},
            ),
            stream=is_stream,
        )
    except httpx.ConnectError as exc:
        logger.error("Backend connection failed for %s: %s", body.get("model", "?"), exc)
        return JSONResponse(
            content={"error": {"message": f"Backend unavailable: {exc}", "type": "server_error"}},
            status_code=502,
        )

    if is_stream:
        if upstream.status_code >= 500:
            body_bytes = b""
            async for chunk in upstream.aiter_bytes():
                body_bytes += chunk
            model_name = body.get("model", "unknown")
            logger.error(
                "Backend stream error for %s: status=%d, body=%s",
                model_name, upstream.status_code, body_bytes[:200],
            )
            return JSONResponse(
                content={
                    "error": {
                        "message": f"Backend returned {upstream.status_code} for model {model_name}",
                        "type": "server_error",
                    }
                },
                status_code=upstream.status_code,
            )
        return StreamingResponse(
            _proxy_stream(upstream),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    # Handle non-JSON or empty responses from upstream
    raw = upstream.content
    if upstream.status_code >= 500 or not raw or not raw.strip():
        model_name = body.get("model", "unknown")
        logger.error(
            "Backend error for %s: status=%d, body=%s",
            model_name, upstream.status_code, raw[:200] if raw else "(empty)",
        )
        return JSONResponse(
            content={
                "error": {
                    "message": f"Backend returned {upstream.status_code} for model {model_name}",
                    "type": "server_error",
                }
            },
            status_code=upstream.status_code if upstream.status_code >= 400 else 502,
        )

    try:
        data = upstream.json()
        data = sanitize_response(data)
    except (json.JSONDecodeError, ValueError):
        logger.error(
            "Invalid JSON from backend for %s: %s", body.get("model", "?"), raw[:200]
        )
        return JSONResponse(
            content={"error": {"message": "Backend returned invalid JSON", "type": "server_error"}},
            status_code=502,
        )
    return JSONResponse(content=data, status_code=upstream.status_code)


# -- Endpoints --------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy_request(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy_request(request, "/v1/completions")


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Embeddings always go to CPU to preserve VRAM for generation."""
    return await _proxy_request(request, "/v1/embeddings", force_cpu=True, strip_nulls=True)


@app.get("/v1/models")
async def list_models():
    """Return deduplicated model list (strips -gpu/-cpu suffixes)."""
    resp = await _client.get("/v1/models")
    data = resp.json()

    if "data" in data:
        seen: set[str] = set()
        deduped: list[dict] = []
        for model in data["data"]:
            stem = _strip_suffix(model.get("id", ""))
            if stem not in seen:
                seen.add(stem)
                model["id"] = stem
                deduped.append(model)
        data["data"] = deduped

    return JSONResponse(content=data)


@app.get("/health")
async def health():
    resp = await _client.get("/health")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/")
async def root():
    return {"status": "ok", "service": "llama-router"}
