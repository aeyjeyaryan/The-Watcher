import asyncio
import base64
import io
import json
import re
import time
from collections import deque

from PIL import Image

from .ollama_client import OllamaError, OllamaUnavailable

NO_CAPTION = "NO_CAPTION"
NO_OBSERVATION = "NO_OBSERVATION"

_WORD_RE = re.compile(r"[a-z0-9']+")
_WHITESPACE_RE = re.compile(r"\s+")
_ABSENCE_RE = re.compile(
    r"\b(no (?:person|people|human|individual|subject)\b"
    r"|nobody\b"
    r"|no one (?:is|in|present|visible|appears|elsewhere)\b"
    r"|empty (?:room|scene|frame|chair|space)\b"
    r"|\b(?:is|are|appears|remains) empty\b"
    r"|not a person\b)",
    re.IGNORECASE,
)
_NO_CAPTION_RE = re.compile(r"\bNO[ _-]?CAPTION\b", re.IGNORECASE)

VISION_FORMAT = {
    "type": "object",
    "properties": {
        "person_visible": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["person_visible", "description"],
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "of", "to", "in", "on", "at", "with", "for", "his", "her",
    "he", "she", "it", "they", "them", "that", "this", "as", "by", "from",
}


def tokenize(text):
    return {token for token in _WORD_RE.findall((text or "").lower()) if token not in STOPWORDS}


def similarity(left, right):
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def clean_line(text):
    text = (text or "").strip()
    text = text.strip('"').strip("'").strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_no_caption(text):
    cleaned = clean_line(text)
    if not _NO_CAPTION_RE.search(cleaned):
        return cleaned
    return _NO_CAPTION_RE.sub("", cleaned).strip(" .,;:-").strip()


def declares_no_caption(text):
    return strip_no_caption(text) == ""


def declares_no_observation(text):
    normalized = clean_line(text).upper()
    if normalized == NO_OBSERVATION or NO_OBSERVATION in normalized:
        return True
    return bool(_ABSENCE_RE.search(text))


def parse_vision_response(text):
    cleaned = clean_line(text)
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict) and "description" in data:
        description = clean_line(str(data.get("description", "")))
        visible = bool(data.get("person_visible", True)) and len(description) > 0
        return visible, description
    visible = bool(cleaned) and not declares_no_observation(cleaned)
    return visible, cleaned


class NarratorPipeline:
    def __init__(self, settings, client):
        self.settings = settings
        self.client = client
        self._lock = asyncio.Lock()
        self._recent_observations = deque(maxlen=settings.context_history)
        self._recent_captions = deque(maxlen=settings.context_history)
        self._last_caption_ts = 0.0
        self._last_captioned_signature = None

    def reset(self):
        self._recent_observations.clear()
        self._recent_captions.clear()
        self._last_caption_ts = 0.0
        self._last_captioned_signature = None

    def snapshot(self):
        return {
            "recent_captions": list(self._recent_captions),
            "recent_observations": list(self._recent_observations),
            "last_caption_ts": self._last_caption_ts,
        }

    @property
    def busy(self):
        return self._lock.locked()

    def _fingerprint(self, image_b64):
        try:
            raw = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(raw)).convert("L").resize((32, 32))
            return image.tobytes()
        except Exception:  # noqa: BLE001
            return None

    def _frame_is_new(self, signature):
        previous = self._last_captioned_signature
        if signature is None or previous is None:
            return True
        total = sum(abs(a - b) for a, b in zip(signature, previous))
        mean_diff = total / len(signature)
        return mean_diff >= self.settings.frame_diff_threshold

    def _too_similar_to_recent(self, text, recent):
        threshold = self.settings.similarity_threshold
        return any(similarity(text, previous) >= threshold for previous in recent)

    def _build_narration_prompt(self, observation, context=None):
        if self._recent_captions:
            recent = "\n".join(f"- {caption}" for caption in self._recent_captions)
        else:
            recent = "(none yet)"
        situation = (context or self.settings.narrator_context or "").strip()
        context_block = (
            f"Off-screen background, never describe it: {situation}\n\n"
            if situation else ""
        )
        return (
            f"{context_block}"
            f"Recent narration:\n{recent}\n\n"
            f"Current visual observation:\n{observation}\n\n"
            "Write the next narration line, staying exactly in your role and "
            "obeying every rule you were given. If it would repeat recent "
            "narration or nothing meaningful changed, reply with exactly "
            "NO_CAPTION."
        )

    async def _observe(self, image_b64):
        raw = await self.client.chat(
            model=self.settings.vision_model,
            system=self.settings.vision_system_prompt,
            user=self.settings.vision_prompt,
            images=[image_b64],
            temperature=self.settings.vision_temperature,
            num_predict=self.settings.vision_num_predict,
            seed=self.settings.vision_seed,
            response_format=VISION_FORMAT,
            timeout=self.settings.vision_timeout,
            keep_alive=self.settings.keep_alive,
        )
        return parse_vision_response(raw)

    async def _narrate(self, observation, context=None):
        caption = await self.client.chat(
            model=self.settings.narrator_model,
            system=self.settings.narrator_prompt,
            user=self._build_narration_prompt(observation, context),
            temperature=self.settings.narrator_temperature,
            num_predict=self.settings.narrator_num_predict,
            timeout=self.settings.narrator_timeout,
            keep_alive=self.settings.keep_alive,
        )
        return clean_line(caption)

    async def analyze(self, image_b64, min_gap=None, context=None):
        if not image_b64:
            return {"status": "error", "error": "Empty image payload."}

        if self._lock.locked():
            return {"status": "busy"}

        async with self._lock:
            return await self._analyze_locked(image_b64, min_gap, context)

    async def _analyze_locked(self, image_b64, min_gap, context=None):
        settings = self.settings
        gap = settings.default_min_gap if min_gap is None else max(0.0, float(min_gap))
        started = time.monotonic()
        result = {
            "status": "no_caption",
            "caption": None,
            "observation": None,
            "reason": None,
            "timings": {},
        }

        try:
            signature = self._fingerprint(image_b64)
            if not self._frame_is_new(signature):
                result["reason"] = "frame_unchanged"
                result["timings"]["total_s"] = round(time.monotonic() - started, 2)
                return result

            observe_start = time.monotonic()
            visible, observation = await self._observe(image_b64)
            result["timings"]["vision_s"] = round(time.monotonic() - observe_start, 2)

            if not visible or not observation:
                result["reason"] = "no_observation"
                result["timings"]["total_s"] = round(time.monotonic() - started, 2)
                return result

            result["observation"] = observation

            if self._too_similar_to_recent(observation, self._recent_observations):
                result["reason"] = "no_change"
                result["timings"]["total_s"] = round(time.monotonic() - started, 2)
                return result

            elapsed = time.monotonic() - self._last_caption_ts
            if self._last_caption_ts and elapsed < gap:
                result["reason"] = "cooldown"
                result["timings"]["total_s"] = round(time.monotonic() - started, 2)
                return result

            narrate_start = time.monotonic()
            caption = await self._narrate(observation, context)
            result["timings"]["narration_s"] = round(time.monotonic() - narrate_start, 2)
            caption = strip_no_caption(caption)

            if len(caption) < 3:
                result["reason"] = "model_declined"
                result["timings"]["total_s"] = round(time.monotonic() - started, 2)
                return result

            if self._too_similar_to_recent(caption, self._recent_captions):
                result["reason"] = "repeat"
                result["timings"]["total_s"] = round(time.monotonic() - started, 2)
                return result

            self._recent_observations.append(observation)
            self._recent_captions.append(caption)
            self._last_caption_ts = time.monotonic()
            self._last_captioned_signature = signature

            result["status"] = "ok"
            result["caption"] = caption
            result["timings"]["total_s"] = round(time.monotonic() - started, 2)
            return result

        except OllamaUnavailable as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            result["timings"]["total_s"] = round(time.monotonic() - started, 2)
            return result
        except OllamaError as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            result["timings"]["total_s"] = round(time.monotonic() - started, 2)
            return result
        except asyncio.TimeoutError:
            result["status"] = "error"
            result["error"] = "Inference timed out."
            result["timings"]["total_s"] = round(time.monotonic() - started, 2)
            return result
        except Exception as exc:  # noqa: BLE001
            result["status"] = "error"
            result["error"] = f"Unexpected error: {exc}"
            result["timings"]["total_s"] = round(time.monotonic() - started, 2)
            return result
