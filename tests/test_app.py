import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from any_guardrail import GuardrailOutput
from fastapi.testclient import TestClient

import app


class FakeGuardrail:
    def __init__(self) -> None:
        self.calls = []

    def validate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return GuardrailOutput(valid=True, explanation="accepted", score=0.9)


class AppTestCase(TestCase):
    def test_app_handlers_are_async_except_sync_service_config(self) -> None:
        self.assertFalse(inspect.iscoroutinefunction(app.get_service_config))
        self.assertTrue(inspect.iscoroutinefunction(app.healthcheck))
        self.assertTrue(inspect.iscoroutinefunction(app.list_profiles))
        self.assertTrue(inspect.iscoroutinefunction(app.validate))

    def test_load_service_config_merges_multiple_yaml_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.yaml"
            second_path = Path(temp_dir) / "second.yaml"

            first_path.write_text(
                "profiles:\n"
                "  prompt-safety:\n"
                "    guardrail_name: harm_guard\n",
                encoding="utf-8",
            )
            second_path.write_text(
                "profiles:\n"
                "  llm-policy:\n"
                "    guardrail_name: any_llm\n"
                "    validate_kwargs:\n"
                "      policy: no unsafe content\n"
                "threadpool:\n"
                "  max_workers: 8\n",
                encoding="utf-8",
            )

            config = app.load_service_config([first_path, second_path])

        self.assertEqual(sorted(config.profiles), ["llm-policy", "prompt-safety"])
        self.assertEqual(config.profiles["prompt-safety"].guardrail_name.value, "harm_guard")
        self.assertEqual(config.threadpool.max_workers, 8)

    def test_apply_threadpool_settings_sets_default_thread_limiter(self) -> None:
        class FakeLimiter:
            total_tokens = 40

        limiter = FakeLimiter()
        config = app.ServiceConfig.model_validate({"profiles": {}, "threadpool": {"max_workers": 12}})

        with patch("app.to_thread.current_default_thread_limiter", return_value=limiter):
            app.apply_threadpool_settings(config)

        self.assertEqual(limiter.total_tokens, 12)

    def test_validate_endpoint_uses_profile_configuration_and_request_overrides(self) -> None:
        fake_guardrail = FakeGuardrail()
        config = app.ServiceConfig.model_validate(
            {
                "profiles": {
                    "llm-policy": {
                        "guardrail_name": "any_llm",
                        "validate_kwargs": {"policy": "stay safe"},
                    }
                }
            }
        )

        app.app.dependency_overrides[app.get_service_config] = lambda: config
        with patch("app.AnyGuardrail.create", return_value=fake_guardrail) as create_mock:
            client = TestClient(app.app)
            response = client.post(
                "/validate",
                json={
                    "profile": "llm-policy",
                    "input_text": "hello",
                    "validate_kwargs": {"model_id": "openai:gpt-5-nano"},
                },
            )
        app.app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "profile": "llm-policy",
                "result": {"valid": True, "explanation": "accepted", "score": 0.9},
            },
        )
        create_mock.assert_called_once()
        self.assertEqual(
            fake_guardrail.calls,
            [(("hello",), {"policy": "stay safe", "model_id": "openai:gpt-5-nano"})],
        )

    def test_profiles_endpoint_lists_available_profiles(self) -> None:
        config = app.ServiceConfig.model_validate(
            {
                "profiles": {
                    "policy-a": {"guardrail_name": "any_llm"},
                    "policy-b": {"guardrail_name": "harm_guard", "model_id": "hbseong/HarmAug-Guard"},
                }
            }
        )

        app.app.dependency_overrides[app.get_service_config] = lambda: config
        client = TestClient(app.app)
        response = client.get("/profiles")
        app.app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [
                {"name": "policy-a", "guardrail_name": "any_llm", "model_id": None},
                {"name": "policy-b", "guardrail_name": "harm_guard", "model_id": "hbseong/HarmAug-Guard"},
            ],
        )
