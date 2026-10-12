import asyncio
import base64
import io
import time
import unittest

from PIL import Image

from backend.config import load_settings
from backend.pipeline import (
    NarratorPipeline,
    clean_line,
    declares_no_caption,
    declares_no_observation,
    parse_vision_response,
    similarity,
    strip_no_caption,
    tokenize,
)


class FakeClient:
    def __init__(self, observations=None, captions=None):
        self.observations = list(observations or [])
        self.captions = list(captions or [])

    async def chat(
        self,
        model,
        user,
        system=None,
        images=None,
        temperature=0.2,
        num_predict=128,
        seed=None,
        response_format=None,
        timeout=120.0,
        keep_alive="5m",
    ):
        if images:
            if not self.observations:
                raise AssertionError("unexpected vision call")
            return self.observations.pop(0)
        if not self.captions:
            raise AssertionError("unexpected narration call")
        return self.captions.pop(0)


def make_pipeline(observations, captions, **overrides):
    settings = load_settings()
    if overrides:
        settings = settings.__class__(**{**settings.__dict__, **overrides})
    return NarratorPipeline(settings, FakeClient(observations, captions))


class TextHelpersTest(unittest.TestCase):
    def test_tokenize_drops_stopwords(self):
        self.assertNotIn("the", tokenize("the cat sat on the mat"))
        self.assertIn("cat", tokenize("the cat sat on the mat"))

    def test_similarity_identical(self):
        self.assertEqual(similarity("he drinks water", "he drinks water"), 1.0)

    def test_similarity_unrelated(self):
        self.assertLess(similarity("he drinks water", "she reads a book"), 0.3)

    def test_clean_line_strips_quotes_and_spacing(self):
        self.assertEqual(clean_line('  "He  sits."  '), "He sits.")

    def test_declares_no_caption(self):
        self.assertTrue(declares_no_caption("NO_CAPTION"))
        self.assertTrue(declares_no_caption("no_caption."))
        self.assertFalse(declares_no_caption("He watches the screen."))

    def test_strip_no_caption_removes_trailing_token(self):
        self.assertEqual(strip_no_caption("He waits. NO_CAPTION"), "He waits")
        self.assertEqual(strip_no_caption("NO_CAPTION"), "")
        self.assertEqual(strip_no_caption("The feast continues."), "The feast continues.")

    def test_declares_no_observation(self):
        self.assertTrue(declares_no_observation("NO_OBSERVATION"))
        self.assertTrue(declares_no_observation("There is no person in the image."))
        self.assertTrue(declares_no_observation("The frame is empty."))
        self.assertFalse(declares_no_observation("A man sits at a desk holding a cup."))
        self.assertFalse(declares_no_observation("No one else is in the frame but him."))

    def test_parse_vision_response_json(self):
        visible, description = parse_vision_response(
            '{"person_visible": true, "description": "A man sits at a desk."}'
        )
        self.assertTrue(visible)
        self.assertEqual(description, "A man sits at a desk.")

    def test_parse_vision_response_json_no_person(self):
        visible, description = parse_vision_response(
            '{"person_visible": false, "description": ""}'
        )
        self.assertFalse(visible)
        self.assertEqual(description, "")

    def test_parse_vision_response_plain_text(self):
        visible, description = parse_vision_response("He sips from a cup.")
        self.assertTrue(visible)
        self.assertEqual(description, "He sips from a cup.")


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_successful_caption(self):
        pipeline = make_pipeline(
            observations=["a person sitting at a desk looking at a laptop"],
            captions=["He returns to the glowing rectangle of his ambition."],
        )
        result = await pipeline.analyze("ZmFrZQ==")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["caption"], "He returns to the glowing rectangle of his ambition.")
        self.assertIn("vision_s", result["timings"])
        self.assertIn("narration_s", result["timings"])

    async def test_no_observation_skips_narration(self):
        pipeline = make_pipeline(observations=["NO_OBSERVATION"], captions=[])
        result = await pipeline.analyze("ZmFrZQ==")
        self.assertEqual(result["status"], "no_caption")
        self.assertEqual(result["reason"], "no_observation")

    async def test_repeated_observation_is_suppressed(self):
        observation = "a person sitting at a desk looking at a laptop"
        pipeline = make_pipeline(
            observations=[observation, observation],
            captions=["He stares into the abyss of the screen."],
        )
        first = await pipeline.analyze("ZmFrZQ==", min_gap=0)
        self.assertEqual(first["status"], "ok")
        second = await pipeline.analyze("ZmFrZQ==", min_gap=0)
        self.assertEqual(second["status"], "no_caption")
        self.assertEqual(second["reason"], "no_change")

    async def test_cooldown_blocks_rapid_caption(self):
        pipeline = make_pipeline(
            observations=[
                "a person sitting at a desk looking at a laptop",
                "a person lifting a glass of water to his mouth",
            ],
            captions=["He stares into the abyss of the screen."],
        )
        first = await pipeline.analyze("ZmFrZQ==", min_gap=0)
        self.assertEqual(first["status"], "ok")
        second = await pipeline.analyze("ZmFrZQ==", min_gap=100)
        self.assertEqual(second["status"], "no_caption")
        self.assertEqual(second["reason"], "cooldown")

    async def test_model_declined(self):
        pipeline = make_pipeline(
            observations=["a person sitting at a desk looking at a laptop"],
            captions=["NO_CAPTION"],
        )
        result = await pipeline.analyze("ZmFrZQ==", min_gap=0)
        self.assertEqual(result["status"], "no_caption")
        self.assertEqual(result["reason"], "model_declined")

    async def test_busy_when_lock_held(self):
        pipeline = make_pipeline(observations=[], captions=[])
        await pipeline._lock.acquire()
        try:
            result = await pipeline.analyze("ZmFrZQ==")
        finally:
            pipeline._lock.release()
        self.assertEqual(result["status"], "busy")

    async def test_unavailable_client_reports_error(self):
        class BrokenClient:
            async def chat(self, *args, **kwargs):
                from backend.ollama_client import OllamaUnavailable
                raise OllamaUnavailable("ollama down")

        settings = load_settings()
        pipeline = NarratorPipeline(settings, BrokenClient())
        result = await pipeline.analyze("ZmFrZQ==")
        self.assertEqual(result["status"], "error")
        self.assertIn("ollama down", result["error"])

    async def test_frame_gate_suppresses_identical_frame(self):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (30, 60, 90)).save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        pipeline = make_pipeline(
            observations=["a person sitting at a desk"],
            captions=["He waits for purpose to arrive."],
        )
        first = await pipeline.analyze(encoded, min_gap=0)
        self.assertEqual(first["status"], "ok")
        second = await pipeline.analyze(encoded, min_gap=0)
        self.assertEqual(second["status"], "no_caption")
        self.assertEqual(second["reason"], "frame_unchanged")

    async def test_reset_clears_memory(self):
        pipeline = make_pipeline(
            observations=["a person at a desk"],
            captions=["He waits."],
        )
        await pipeline.analyze("ZmFrZQ==", min_gap=0)
        self.assertTrue(pipeline.snapshot()["recent_captions"])
        pipeline.reset()
        self.assertEqual(pipeline.snapshot()["recent_captions"], [])


if __name__ == "__main__":
    unittest.main()
