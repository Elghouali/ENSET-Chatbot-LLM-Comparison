"""Offline logic checks: no Ollama server needed, dependencies must be installed."""
import unittest

from core import GenerationOptions, generate, installed_models


class DummySample:
    def start(self):
        pass

    def stop(self):
        from monitor import ResourceSummary
        return ResourceSummary(samples=1)


class FakeClient:
    def list(self):
        return {"models": [{"name": "qwen2.5:0.5b"}, {"model": "llama3.2:latest"}]}

    def chat(self, **kwargs):
        assert kwargs["stream"] is True
        yield {"message": {"content": "Bon"}}
        yield {"message": {"content": "jour"}}
        yield {"message": {"content": ""}, "eval_count": 20, "eval_duration": 1_000_000_000}


class CoreTests(unittest.TestCase):
    def test_actual_installed_model_names(self):
        self.assertEqual(installed_models(FakeClient()), ["llama3.2:latest", "qwen2.5:0.5b"])

    def test_streamed_text_and_real_generation_metadata(self):
        calls = []
        result = generate(FakeClient(), "llama3.2:latest", [{"role": "user", "content": "Salut"}],
                          GenerationOptions(), calls.append, sampler_factory=DummySample)
        self.assertEqual(result.response, "Bonjour")
        self.assertEqual(result.generated_tokens, 20)
        self.assertAlmostEqual(result.tokens_s, 20.0)
        self.assertEqual(calls, ["Bon", "Bonjour"])
        self.assertIsNotNone(result.ttft_s)


if __name__ == "__main__":
    unittest.main()
