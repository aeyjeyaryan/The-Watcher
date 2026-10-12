import asyncio
import io
import os
import shutil
import tempfile
import threading
import wave


class TTSError(Exception):
    pass


def _voice_lang(voice):
    if voice.startswith(("bf_", "bm_")):
        return "en-gb"
    if voice.startswith(("af_", "am_")):
        return "en-us"
    return "en-us"


def _to_wav(samples, sample_rate):
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class KokoroEngine:
    name = "kokoro"
    default_voice = "bm_george"

    def __init__(self, model_dir, speed):
        self.model_dir = model_dir
        self.speed = speed
        self._model = None
        self._lock = threading.Lock()
        self._voices = None

    @property
    def model_path(self):
        return os.path.join(self.model_dir, "kokoro-v1.0.onnx")

    @property
    def voices_path(self):
        return os.path.join(self.model_dir, "voices-v1.0.bin")

    def available(self):
        if not (os.path.exists(self.model_path) and os.path.exists(self.voices_path)):
            return False
        try:
            import kokoro_onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        with self._lock:
            if self._model is None:
                from kokoro_onnx import Kokoro

                self._model = Kokoro(self.model_path, self.voices_path)
            return self._model

    def list_voices(self):
        if self._voices is None:
            try:
                self._voices = sorted(self._load().get_voices())
            except Exception as exc:  # noqa: BLE001
                raise TTSError(f"Kokoro voice list failed: {exc}") from exc
        return self._voices

    def synthesize(self, text, voice):
        model = self._load()
        voice = voice or self.default_voice
        samples, sample_rate = model.create(
            text, voice=voice, speed=self.speed, lang=_voice_lang(voice)
        )
        return _to_wav(samples, sample_rate), "audio/wav"


class SayEngine:
    name = "say"
    default_voice = "Daniel"

    def __init__(self, speed):
        self.rate = max(90, min(300, int(180 * speed)))

    def available(self):
        return shutil.which("say") is not None

    def list_voices(self):
        if not self.available():
            return []
        import re
        import subprocess

        try:
            output = subprocess.run(
                ["say", "-v", "?"], capture_output=True, text=True, timeout=10
            ).stdout
        except Exception:  # noqa: BLE001
            return []
        pattern = re.compile(r"^(?P<name>.+?)\s{2,}(?P<lang>[a-z]{2}_[A-Z]{2})\b")
        voices = []
        for line in output.splitlines():
            match = pattern.match(line)
            if match and match.group("lang").startswith("en"):
                voices.append(match.group("name").strip())
        return sorted(set(voices))

    async def _run(self, *args, timeout):
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            raise TTSError("Text-to-speech timed out.") from exc
        if process.returncode != 0:
            message = (stderr or b"").decode("utf-8", "ignore").strip()
            raise TTSError(message or "Text-to-speech failed.")
        return stderr

    async def synthesize(self, text, voice):
        if not self.available():
            raise TTSError("macOS `say` command not found.")
        voice = voice or self.default_voice
        with tempfile.TemporaryDirectory() as workdir:
            aiff_path = os.path.join(workdir, "speech.aiff")
            wav_path = os.path.join(workdir, "speech.wav")
            try:
                await self._run(
                    "say", "-v", voice, "-r", str(self.rate), "-o", aiff_path, text,
                    timeout=30.0,
                )
            except TTSError:
                if voice == self.default_voice:
                    raise
                await self._run(
                    "say", "-v", self.default_voice, "-r", str(self.rate),
                    "-o", aiff_path, text, timeout=30.0,
                )
            if shutil.which("afconvert"):
                await self._run(
                    "afconvert", "-f", "WAVE", "-d", "LEI16", aiff_path, wav_path,
                    timeout=30.0,
                )
                with open(wav_path, "rb") as handle:
                    return handle.read(), "audio/wav"
            with open(aiff_path, "rb") as handle:
                return handle.read(), "audio/aiff"


class TTSManager:
    def __init__(self, settings):
        self.settings = settings
        self.engine = self._select(settings)
        self._voices = None

    @staticmethod
    def _select(settings):
        requested = (settings.tts_engine or "auto").lower()
        kokoro = KokoroEngine(settings.kokoro_model_dir, settings.tts_speed)
        say = SayEngine(settings.tts_speed)
        if requested in ("auto", "kokoro") and kokoro.available():
            return kokoro
        if requested == "kokoro":
            import logging

            logging.getLogger("narrator").warning(
                "Kokoro model not found in %s; falling back to macOS `say`.",
                settings.kokoro_model_dir,
            )
        if say.available():
            return say
        if kokoro.available():
            return kokoro
        return None

    @property
    def engine_name(self):
        return self.engine.name if self.engine else "none"

    def available(self):
        return self.engine is not None

    def default_voice(self):
        if not self.engine:
            return ""
        configured = getattr(self.settings, "tts_voice", "") or ""
        if configured:
            return configured
        return self.engine.default_voice

    def list_voices(self):
        if not self.engine:
            return []
        if self._voices is None:
            try:
                self._voices = self.engine.list_voices()
            except TTSError:
                self._voices = []
        return self._voices

    async def synthesize(self, text, voice=None):
        if not self.engine:
            raise TTSError("No text-to-speech engine is available.")
        text = (text or "").strip()
        if not text:
            raise TTSError("No text to speak.")
        if len(text) > 600:
            text = text[:600]
        voice = voice or self.default_voice()
        if isinstance(self.engine, KokoroEngine):
            return await asyncio.to_thread(self.engine.synthesize, text, voice)
        return await self.engine.synthesize(text, voice)
