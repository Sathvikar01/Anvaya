"""FastAPI demo surface for the frozen Voice-RAG harness."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import tempfile
import threading
import time
import wave
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hh_goa_rag.harness import VoiceRAGHarness

LOGGER = logging.getLogger(__name__)

APP_TITLE = "Anvaya Voice-RAG"
MAX_AUDIO_BYTES = 2 * 1024 * 1024
MAX_AUDIO_SECONDS = 30.0
MAX_TEXT_CHARS = 2_000
MAX_CONCURRENT_REQUESTS = 1
MAX_REQUEST_BODY_BYTES = MAX_AUDIO_BYTES + 64 * 1024
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
STATIC_DIR = Path(__file__).with_name("static")
SUPPORTED_LANGUAGES = {
    "hi-IN": "हिन्दी",
    "en-IN": "English",
    "bn-IN": "বাংলা",
    "gu-IN": "ગુજરાતી",
    "kn-IN": "ಕನ್ನಡ",
    "ml-IN": "മലയാളം",
    "mr-IN": "मराठी",
    "od-IN": "ଓଡ଼ିଆ",
    "pa-IN": "ਪੰਜਾਬੀ",
    "ta-IN": "தமிழ்",
    "te-IN": "తెలుగు",
    "as-IN": "অসমীয়া",
    "ur-IN": "اردو",
    "ne-IN": "नेपाली",
    "kok-IN": "कोंकणी",
    "ks-IN": "कॉशुर / कश्मीरी",
    "sd-IN": "सिन्धी",
    "sa-IN": "संस्कृतम्",
    "sat-IN": "संताली",
    "mni-IN": "মৈতৈলোন্",
    "brx-IN": "बड़ो",
    "mai-IN": "मैथिली",
    "doi-IN": "डोगरी",
}


@dataclass(frozen=True)
class AppSettings:
    retriever_config: Path
    env_file: Path
    device: str
    api_token: str | None = None

    @classmethod
    def from_env(cls) -> AppSettings:
        load_dotenv(Path(".env"), override=False)
        selected_env_file = Path(os.getenv("HH_RAG_ENV_FILE", ".env"))
        load_dotenv(selected_env_file, override=False)
        return cls(
            retriever_config=Path(
                os.getenv("HH_RAG_RETRIEVER_CONFIG", "results/final_retriever_config.json")
            ),
            env_file=selected_env_file,
            device=os.getenv("HH_RAG_DEVICE", "auto"),
            api_token=os.getenv("HH_RAG_API_TOKEN") or None,
        )


class ProgressRegistry:
    """Small in-memory progress registry for the single model-serving worker."""

    def __init__(self, *, max_entries: int = 128) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def start(self, request_id: str, *, initial_stage: str = "Preparing audio") -> None:
        with self._lock:
            self._prune()
            self._entries[request_id] = {
                "stage": initial_stage,
                "history": [],
                "complete": False,
                "error": False,
                "updated_at": time.time(),
            }

    def update(self, request_id: str, stage_name: str) -> None:
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            entry["stage"] = stage_name
            if not entry["history"] or entry["history"][-1] != stage_name:
                entry["history"].append(stage_name)
            entry["updated_at"] = time.time()

    def finish(self, request_id: str, *, error: bool = False) -> None:
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            entry["stage"] = "Complete" if not error else "System error"
            if not error and (not entry["history"] or entry["history"][-1] != "Complete"):
                entry["history"].append("Complete")
            entry["complete"] = True
            entry["error"] = error
            entry["updated_at"] = time.time()

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(request_id)
            return dict(entry) if entry is not None else None

    def _prune(self) -> None:
        if len(self._entries) < self._max_entries:
            return
        oldest = sorted(self._entries, key=lambda key: self._entries[key]["updated_at"])
        for key in oldest[: max(1, len(oldest) // 4)]:
            self._entries.pop(key, None)


def validate_environment(settings: AppSettings) -> dict[str, Any]:
    """Validate secrets and frozen artifacts without revealing secret values."""
    load_dotenv(settings.env_file, override=False)
    errors: list[str] = []
    if not os.getenv("SARVAM_API_KEY", "").strip():
        errors.append("SARVAM_API_KEY is missing")
    if not settings.retriever_config.is_file():
        errors.append(f"retriever config is missing: {settings.retriever_config}")
    else:
        try:
            config = json.loads(settings.retriever_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"retriever config is unreadable: {type(error).__name__}")
        else:
            for key in ("model_cache_path", "index_artifact", "chunk_artifact"):
                path = Path(str(config.get(key, "")).replace("\\", "/"))
                if not path.exists():
                    errors.append(f"required artifact is missing: {key}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "sarvam_api_key": "configured",
        "generation": "local-qwen-gguf",
        "retriever_config": "available",
        "frozen_artifacts": "available",
    }


def _default_harness_factory(settings: AppSettings) -> VoiceRAGHarness:
    validate_environment(settings)
    return VoiceRAGHarness.from_frozen_artifacts(
        retriever_config_path=settings.retriever_config,
        env_path=settings.env_file,
        device=settings.device,
        include_stt=True,
    )


def create_app(
    *,
    harness_factory: Callable[[AppSettings], Any] | None = None,
    settings: AppSettings | None = None,
    include_ui: bool = True,
    defer_startup: bool = False,
) -> FastAPI:
    configured_settings = settings or AppSettings.from_env()
    build_harness = harness_factory or _default_harness_factory
    progress = ProgressRegistry()
    request_slots = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    active_jobs: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        app.state.harness = None
        app.state.startup_error = None
        app.state.environment = {}
        def initialize_pipeline() -> None:
            try:
                if harness_factory is None:
                    app.state.environment = validate_environment(configured_settings)
                app.state.harness = build_harness(configured_settings)
            except Exception as error:  # health must remain reachable on startup failure
                LOGGER.exception("Voice-RAG startup failed")
                app.state.startup_error = f"{type(error).__name__}"

        if defer_startup:
            # Some hosted demos need to expose their UI while a large model is
            # downloading. Requests remain explicitly unavailable until the
            # pipeline is ready, while the health endpoint reports 503.
            app.state.startup_thread = threading.Thread(
                target=initialize_pipeline,
                name="voice-rag-startup",
                daemon=True,
            )
            app.state.startup_thread.start()
        else:
            initialize_pipeline()
        yield
        if active_jobs:
            await asyncio.gather(*active_jobs, return_exceptions=True)
        harness = app.state.harness
        if harness is not None:
            close = getattr(harness, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title=APP_TITLE,
        description="Live demo for the frozen HH Goa Voice-RAG pipeline.",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.progress = progress

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[..., Any]) -> Response:
        content_length = request.headers.get("content-length")
        if request.url.path in {"/api/query/audio", "/api/query/text"} and content_length:
            try:
                max_body_bytes = (
                    MAX_REQUEST_BODY_BYTES
                    if request.url.path == "/api/query/audio"
                    else MAX_TEXT_CHARS + 64 * 1024
                )
                if int(content_length) > max_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                "Audio upload is too large"
                                if request.url.path == "/api/query/audio"
                                else "Text query is too large"
                            )
                        },
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid content length"})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Hugging Face renders a Space app inside an iframe on the Space page.
        # Keep the strict local default, but allow the hosted wrapper to opt in
        # so the unchanged UI can be displayed from the HF app tab.
        if os.getenv("HH_RAG_ALLOW_EMBED", "").strip().lower() not in {"1", "true", "yes"}:
            response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "microphone=(self)"
        return response

    if include_ui:

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        if request.app.state.harness is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unavailable",
                    "pipeline": "frozen-voice-rag",
                    "detail": "pipeline is not ready",
                },
            )
        return JSONResponse(
            {
                "status": "ok",
                "pipeline": "frozen-voice-rag",
                "checks": request.app.state.environment,
            }
        )

    @app.get("/api/query/status/{request_id}")
    async def query_status(request_id: str) -> dict[str, Any]:
        entry = progress.get(request_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Unknown request ID")
        entry.pop("updated_at", None)
        return entry

    @app.post("/api/query/audio")
    async def query_audio(
        request: Request,
        response: Response,
        audio: UploadFile,
        language_code: str = Form(default="hi-IN"),
        x_request_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        request_id = _validated_request_id(x_request_id)
        language_code = _validated_language_code(language_code)
        response.headers["X-Request-ID"] = request_id
        _require_api_token(request, configured_settings.api_token)
        progress.start(request_id)
        harness = request.app.state.harness
        if harness is None:
            progress.finish(request_id, error=True)
            raise HTTPException(status_code=503, detail="Voice-RAG pipeline is unavailable")

        temporary_path: Path | None = None
        try:
            temporary_path = await _save_upload(audio)
            _validate_pcm16_wav(temporary_path)
            async with request_slots:
                job = asyncio.create_task(
                    asyncio.to_thread(
                        harness.handle_audio,
                        temporary_path,
                        language_code=language_code,
                        on_stage=lambda stage_name: progress.update(request_id, stage_name),
                    )
                )
                active_jobs.add(job)
                try:
                    result = await asyncio.shield(job)
                except asyncio.CancelledError:
                    await asyncio.shield(job)
                    raise
                finally:
                    active_jobs.discard(job)
            progress.finish(request_id)
            return result.to_dict()
        except HTTPException:
            progress.finish(request_id, error=True)
            raise
        except Exception:
            LOGGER.exception("Audio request failed", extra={"request_id": request_id})
            progress.finish(request_id, error=True)
            raise HTTPException(
                status_code=500, detail="The request could not be processed"
            ) from None
        finally:
            await audio.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @app.post("/api/query/text")
    async def query_text(
        request: Request,
        response: Response,
        x_request_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        request_id = _validated_request_id(x_request_id)
        response.headers["X-Request-ID"] = request_id
        _require_api_token(request, configured_settings.api_token)
        progress.start(request_id, initial_stage="Preparing text")
        harness = request.app.state.harness
        if harness is None:
            progress.finish(request_id, error=True)
            raise HTTPException(status_code=503, detail="Voice-RAG pipeline is unavailable")

        try:
            try:
                payload = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise HTTPException(
                    status_code=400, detail="Request body must be valid JSON"
                ) from None
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                raise HTTPException(status_code=422, detail="Text query must be a string")
            text = payload["text"]
            language_code = _validated_language_code(payload.get("language_code", "hi-IN"))
            if len(text) > MAX_TEXT_CHARS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Text query must be {MAX_TEXT_CHARS:,} characters or fewer",
                )
            async with request_slots:
                job = asyncio.create_task(
                    asyncio.to_thread(
                        harness.handle_text,
                        text,
                        language_code=language_code,
                        on_stage=lambda stage_name: progress.update(request_id, stage_name),
                    )
                )
                active_jobs.add(job)
                try:
                    result = await asyncio.shield(job)
                except asyncio.CancelledError:
                    await asyncio.shield(job)
                    raise
                finally:
                    active_jobs.discard(job)
            progress.finish(request_id)
            return result.to_dict()
        except HTTPException:
            progress.finish(request_id, error=True)
            raise
        except Exception:
            LOGGER.exception("Text request failed", extra={"request_id": request_id})
            progress.finish(request_id, error=True)
            raise HTTPException(
                status_code=500, detail="The request could not be processed"
            ) from None

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _validated_request_id(value: str | None) -> str:
    # Hosting proxies may append their own non-conforming x-request-id; be
    # tolerant of foreign formats by sanitizing instead of rejecting, so plain
    # HTTP clients without the header are never blocked.
    if value:
        cleaned = re.sub(r"[^A-Za-z0-9_-]", "-", value.strip()).strip("-")[:80]
        if len(cleaned) >= 8:
            return cleaned
    import secrets

    return secrets.token_urlsafe(16)


def _validated_language_code(value: object) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail="Unsupported language_code; choose one of the listed Saaras languages",
        )
    return value


async def _save_upload(upload: UploadFile) -> Path:
    suffix = ".wav"
    descriptor, raw_path = tempfile.mkstemp(prefix="hh-goa-audio-", suffix=suffix)
    os.close(descriptor)
    path = Path(raw_path)
    total = 0
    try:
        with path.open("wb") as handle:
            while chunk := await upload.read(64 * 1024):
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="Audio upload is too large")
                handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="The uploaded recording is empty")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _validate_pcm16_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
    except (OSError, EOFError, wave.Error):
        raise HTTPException(status_code=422, detail="Audio must be a valid PCM WAV file") from None
    if (sample_rate, channels, sample_width) != (16_000, 1, 2):
        raise HTTPException(
            status_code=422,
            detail="Audio must be 16 kHz, mono, PCM16 WAV",
        )
    if frames <= 0:
        raise HTTPException(status_code=400, detail="The recording contains no audio frames")
    if frames / sample_rate > MAX_AUDIO_SECONDS + 0.01:
        raise HTTPException(status_code=413, detail="Audio exceeds the 30-second limit")


def _require_api_token(request: Request, expected: str | None) -> None:
    if expected is None:
        return
    supplied = request.headers.get("authorization", "")
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()
    else:
        supplied = request.headers.get("x-api-key", "")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Authentication required")


app = create_app()


def main() -> None:
    uvicorn.run(
        "hh_goa_rag.web:app",
        host=os.getenv("HH_RAG_BIND_HOST", "127.0.0.1"),
        port=int(os.getenv("HH_RAG_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
