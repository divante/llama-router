"""CPU-first load-balancing proxy for llama-server with GPU/CPU model variants.

Accepts OpenAI-compatible requests with plain model names (e.g. "qwen3.5:35b"),
routes to the :cpu variant by default, falls back to :gpu when CPU returns 503.
"""

import logging
import os
import re
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("llama-router")

BACKEND = os.getenv("BACKEND_URL", "http://llama-server:8080").rstrip("/")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config/config.ini")
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))

ROUTED_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
})

HOP_HEADERS = frozenset({
    "host", "content-length", "transfer-encoding", "connection",
    "keep-alive", "upgrade",
})


def load_model_registry(config_path: str) -> dict[str, dict[str, bool]]:
    """Parse config.ini section headers to discover model variants.

    Returns {base_name: {"gpu": bool, "cpu": bool}}.
    """
    registry: dict[str, dict[str, bool]] = {}
    section_re = re.compile(r"^\[(.+)\]$")

    try:
        with open(config_path) as f:
            for line in f:
                m = section_re.match(line.strip())
                if not m:
                    continue
                section = m.group(1)
                for suffix in (":gpu", ":cpu"):
                    if section.endswith(suffix):
                        base = section[: -len(suffix)]
                        registry.setdefault(base, {"gpu": False, "cpu": False})
                        registry[base][suffix[1:]] = True
    except FileNotFoundError:
        logger.warning("Config %s not found, will discover from backend", config_path)

    return registry


class Router:
    def __init__(self) -> None:
        self.registry: dict[str, dict[str, bool]] = {}
        self.client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        self.registry = load_model_registry(CONFIG_PATH)
        if not self.registry:
            await self._discover()
        logger.info("Models loaded: %s", list(self.registry.keys()))

    async def stop(self) -> None:
        if self.client:
            await self.client.aclose()

    async def _discover(self) -> None:
        """Discover models from llama-server /v1/models."""
        try:
            r = await self.client.get(f"{BACKEND}/v1/models")
            r.raise_for_status()
            for m in r.json().get("data", []):
                mid = m["id"]
                for suffix in (":gpu", ":cpu"):
                    if mid.endswith(suffix):
                        base = mid[: -len(suffix)]
                        self.registry.setdefault(base, {"gpu": False, "cpu": False})
                        self.registry[base][suffix[1:]] = True
        except Exception as e:
            logger.error("Backend discovery failed: %s", e)

    def _variants(self, model: str) -> list[str]:
        """Return ordered variant list. CPU first, GPU fallback."""
        if model.endswith((":gpu", ":cpu")):
            return [model]

        info = self.registry.get(model)
        if not info:
            return [model]

        out = []
        if info["cpu"]:
            out.append(f"{model}:cpu")
        if info["gpu"]:
            out.append(f"{model}:gpu")
        return out or [model]

    async def route_completion(self, request: Request) -> Response:
        """Route a completion request with CPU-first, GPU-fallback."""
        body = await request.json()
        model = body.get("model", "")
        to_try = self._variants(model)
        headers = _proxy_headers(request)
        url = f"{BACKEND}{request.url.path}"
        is_stream = body.get("stream", False)

        last_err: Response | None = None
        for variant in to_try:
            body["model"] = variant

            try:
                req = self.client.build_request("POST", url, json=body, headers=headers)
                resp = await self.client.send(req, stream=True)
            except httpx.ConnectError:
                logger.warning("Backend unreachable for %s", variant)
                last_err = JSONResponse(
                    {"error": {"message": "Backend unreachable", "type": "server_error"}},
                    status_code=502,
                )
                continue
            except httpx.TimeoutException:
                logger.warning("Timeout connecting to %s", variant)
                last_err = JSONResponse(
                    {"error": {"message": "Backend timeout", "type": "server_error"}},
                    status_code=504,
                )
                continue

            if resp.status_code == 503:
                await resp.aclose()
                logger.info("%s -> 503, trying next variant", variant)
                last_err = JSONResponse(
                    {"error": {"message": f"All slots busy for {model}", "type": "server_error"}},
                    status_code=503,
                )
                continue

            logger.info("%s -> %s%s", model, variant, " (stream)" if is_stream else "")

            if is_stream:
                async def stream_body(r=resp):
                    try:
                        async for chunk in r.aiter_raw():
                            yield chunk
                    finally:
                        await r.aclose()

                return StreamingResponse(
                    stream_body(),
                    status_code=resp.status_code,
                    headers=_filter_headers(resp.headers),
                    media_type=resp.headers.get("content-type", "text/event-stream"),
                )
            else:
                content = await resp.aread()
                await resp.aclose()
                return Response(
                    content=content,
                    status_code=resp.status_code,
                    headers=_filter_headers(resp.headers),
                )

        return last_err or JSONResponse(
            {"error": {"message": f"No variant available for {model}", "type": "server_error"}},
            status_code=503,
        )

    async def proxy_passthrough(self, request: Request) -> Response:
        """Proxy a request directly without model routing."""
        body = await request.body()
        headers = _proxy_headers(request)
        url = f"{BACKEND}{request.url.path}"

        try:
            resp = await self.client.request(
                request.method, url, content=body, headers=headers
            )
        except httpx.ConnectError:
            return JSONResponse({"error": "Backend unreachable"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse({"error": "Backend timeout"}, status_code=504)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=_filter_headers(resp.headers),
        )

    async def list_models(self) -> JSONResponse:
        """Return deduplicated model list with base names."""
        try:
            r = await self.client.get(f"{BACKEND}/v1/models")
        except (httpx.ConnectError, httpx.TimeoutException):
            return JSONResponse({"error": "Backend unavailable"}, status_code=502)

        if r.status_code != 200:
            return JSONResponse(r.json(), status_code=r.status_code)

        seen: set[str] = set()
        deduped = []
        for m in r.json().get("data", []):
            mid = m["id"]
            base = mid
            for suffix in (":gpu", ":cpu"):
                if mid.endswith(suffix):
                    base = mid[: -len(suffix)]
                    break
            if base not in seen:
                seen.add(base)
                entry = dict(m)
                entry["id"] = base
                deduped.append(entry)

        return JSONResponse({"object": "list", "data": deduped})


def _proxy_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}


def _filter_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS}


# --- App setup ---

router = Router()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await router.start()
    yield
    await router.stop()


app = FastAPI(title="llama-router", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "models": len(router.registry)}


@app.get("/v1/models")
async def models():
    return await router.list_models()


async def _route_completion(request: Request):
    return await router.route_completion(request)


for _path in ROUTED_PATHS:
    app.add_api_route(_path, _route_completion, methods=["POST"])


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(request: Request):
    return await router.proxy_passthrough(request)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
