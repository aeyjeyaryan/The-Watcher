import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"

DEFAULT_VISION_PROMPT = (
    "Look at the webcam image of a person. Describe only what is visibly "
    "supported. One or two short factual sentences, or NO_OBSERVATION."
)
DEFAULT_NARRATOR_PROMPT = (
    "You are narrating a nature documentary. Your only subject: a software "
    "engineering intern sitting at a desk. You refer to him as master. You "
    "act like you are narrating the daily life of a main character. If you "
    "see him using a mobile phone, say exactly: Master has 3 assignments "
    "pending and yet he chooses to scroll reels. Master's manager should "
    "whoop his ass. Write new lines every time. Third person, present tense, "
    "simple words. Funny and dry. Never mean, never insulting. Never describe "
    "anything not visible on screen. No emojis, no hashtags, no quotation "
    "marks, no stage directions. Never address the viewer, explain the joke, "
    "or give advice. Never open with I see, Here is, It appears, or As an AI. "
    "If the scene has not meaningfully changed since your last caption, reply "
    "with exactly: NO_CAPTION."
)
VISION_SYSTEM_PROMPT = (
    "You are a precise visual observer. You report only visible facts about "
    "people and objects. You never speculate about thoughts or emotions."
)


def _env(name, default):
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _env_float(name, default):
    try:
        return float(_env(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(_env(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _load_prompt(filename, default):
    override = os.environ.get(f"{filename.split('.')[0].upper()}_PROMPT")
    if override:
        return override
    path = PROMPTS_DIR / filename
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return default


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    ollama_host: str
    vision_model: str
    narrator_model: str
    vision_timeout: float
    narrator_timeout: float
    vision_temperature: float
    narrator_temperature: float
    vision_num_predict: int
    narrator_num_predict: int
    vision_seed: int
    frame_diff_threshold: float
    similarity_threshold: float
    default_min_gap: float
    context_history: int
    keep_alive: str
    narrator_context: str
    tts_engine: str
    tts_voice: str
    tts_speed: float
    tts_timeout: float
    kokoro_model_dir: str
    log_level: str
    vision_prompt: str
    narrator_prompt: str
    vision_system_prompt: str
    save_debug_frames: bool
    debug_frames_dir: Path


def load_settings():
    return Settings(
        host=_env("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8765),
        ollama_host=_env("OLLAMA_HOST", "http://127.0.0.1:11434"),
        vision_model=_env("VISION_MODEL", "gemma3:4b"),
        narrator_model=_env("NARRATOR_MODEL", "gemma3:4b"),
        vision_timeout=_env_float("VISION_TIMEOUT", 120.0),
        narrator_timeout=_env_float("NARRATION_TIMEOUT", 120.0),
        vision_temperature=_env_float("VISION_TEMPERATURE", 0.0),
        narrator_temperature=_env_float("NARRATOR_TEMPERATURE", 0.85),
        vision_num_predict=_env_int("VISION_NUM_PREDICT", 160),
        narrator_num_predict=_env_int("NARRATOR_NUM_PREDICT", 90),
        vision_seed=_env_int("VISION_SEED", 42),
        frame_diff_threshold=_env_float("FRAME_DIFF_THRESHOLD", 3.5),
        similarity_threshold=_env_float("SIMILARITY_THRESHOLD", 0.72),
        default_min_gap=_env_float("DEFAULT_MIN_GAP", 4.0),
        context_history=_env_int("CONTEXT_HISTORY", 6),
        keep_alive=_env("OLLAMA_KEEP_ALIVE", "5m"),
        narrator_context=_env(
            "NARRATOR_CONTEXT",
            "a software engineering intern buried in deadlines, tickets and "
            "unanswered messages, who keeps scrolling instead of working",
        ),
        tts_engine=_env("TTS_ENGINE", "auto"),
        tts_voice=_env("TTS_VOICE", ""),
        tts_speed=_env_float("TTS_SPEED", 0.92),
        tts_timeout=_env_float("TTS_TIMEOUT", 60.0),
        kokoro_model_dir=_env("KOKORO_MODEL_DIR", str(ROOT / "models" / "tts")),
        log_level=_env("LOG_LEVEL", "INFO"),
        vision_prompt=_load_prompt("vision_prompt.txt", DEFAULT_VISION_PROMPT),
        narrator_prompt=_load_prompt("narrator_prompt.txt", DEFAULT_NARRATOR_PROMPT),
        vision_system_prompt=_env("VISION_SYSTEM_PROMPT", VISION_SYSTEM_PROMPT),
        save_debug_frames=_env_bool("SAVE_DEBUG_FRAMES", False),
        debug_frames_dir=ROOT / "debug_frames",
    )
