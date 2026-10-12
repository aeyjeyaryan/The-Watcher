import base64
import binascii
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import tts
from .config import ROOT, load_settings
from .ollama_client import OllamaClient
from .pipeline import NarratorPipeline

logger = logging.getLogger("narrator")

SETTINGS = load_settings()

FRONTEND_DIR = ROOT / "frontend"

state = {
    "client": None,
    "pipeline": None,
    "tts": None,
    "models": [],
    "models_missing": [],
    "ollama_ok": False,
}


class AnalyzeRequest(BaseModel):
    image: str
    min_gap: float | None = None
    context: str | None = None


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None


class ResetRequest(BaseModel):
    pass


def _strip_data_url(image):
    if not image:
        return ""
    if image.startswith("data:"):
        _, _, payload = image.partition(",")
        return payload
    return image


async def _refresh_models():
    client = state["client"]
    try:
        models = await client.list_models()
        state["ollama_ok"] = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list Ollama models: %s", exc)
        state["models"] = []
        state["models_missing"] = [SETTINGS.vision_model, SETTINGS.narrator_model]
        state["ollama_ok"] = False
        return
    state["models"] = models
    installed = {name.split(":")[0] for name in models}
    installed.update(models)
    missing = []
    for model in {SETTINGS.vision_model, SETTINGS.narrator_model}:
        if model not in installed and model.split(":")[0] not in installed:
            missing.append(model)
    state["models_missing"] = missing
    if missing:
        logger.warning(
            "Missing Ollama models: %s. Pull them with: ollama pull " + " && ollama pull ".join(missing),
            ", ".join(missing),
        )


@asynccontextmanager
async def lifespan(app):
    logging.basicConfig(
        level=SETTINGS.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    state["client"] = OllamaClient(SETTINGS.ollama_host, max(
        SETTINGS.vision_timeout, SETTINGS.narrator_timeout
    ))
    state["pipeline"] = NarratorPipeline(SETTINGS, state["client"])
    state["tts"] = tts.TTSManager(SETTINGS)
    await _refresh_models()
    logger.info(
        "Narrator ready. vision=%s narrator=%s ollama=%s tts=%s",
        SETTINGS.vision_model, SETTINGS.narrator_model, SETTINGS.ollama_host,
        state["tts"].engine_name,
    )
    try:
        yield
    finally:
        await state["client"].close()


app = FastAPI(title="AI Life Documentary", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health():
    client = state["client"]
    ollama_ok = await client.ping() if client else False
    if ollama_ok:
        await _refresh_models()
    ready = ollama_ok and not state["models_missing"]
    return {
        "ollama_ok": ollama_ok,
        "ready": ready,
        "vision_model": SETTINGS.vision_model,
        "narrator_model": SETTINGS.narrator_model,
        "models": state["models"],
        "models_missing": state["models_missing"],
        "busy": state["pipeline"].busy if state["pipeline"] else False,
        "tts_available": state["tts"].available() if state["tts"] else False,
        "tts_engine": state["tts"].engine_name if state["tts"] else "none",
    }


@app.get("/api/config")
async def config():
    manager = state["tts"]
    return {
        "vision_model": SETTINGS.vision_model,
        "narrator_model": SETTINGS.narrator_model,
        "default_min_gap": SETTINGS.default_min_gap,
        "narrator_context": SETTINGS.narrator_context,
        "tts_engine": manager.engine_name if manager else "none",
        "tts_voice": manager.default_voice() if manager else "",
        "tts_available": manager.available() if manager else False,
        "voices": manager.list_voices() if manager else [],
    }


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    pipeline = state["pipeline"]
    image_b64 = _strip_data_url(request.image)
    if not image_b64:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "Missing image."},
        )
    try:
        base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "Image is not valid base64."},
        )

    if SETTINGS.save_debug_frames:
        try:
            SETTINGS.debug_frames_dir.mkdir(parents=True, exist_ok=True)
            path = SETTINGS.debug_frames_dir / f"frame_{int(time.time())}.jpg"
            path.write_bytes(base64.b64decode(image_b64))
        except OSError as exc:
            logger.warning("Could not save debug frame: %s", exc)

    result = await pipeline.analyze(
        image_b64, min_gap=request.min_gap, context=request.context
    )
    return JSONResponse(content=result)


@app.post("/api/tts")
async def synthesize(request: TTSRequest):
    voice = request.voice or None
    try:
        audio, media_type = await state["tts"].synthesize(request.text, voice)
    except tts.TTSError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(exc)},
        )
    return Response(content=audio, media_type=media_type)


@app.post("/api/reset")
async def reset(_: ResetRequest | None = None):
    state["pipeline"].reset()
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
