import inspect
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Generator

import numpy as np
import pytest
from any_guardrail import GuardrailName, GuardrailOutput
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

import app
import models


class FakeGuardrail:
    def __init__(self) -> None:
        self.calls = []

    def validate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return GuardrailOutput(valid=True, explanation="accepted", score=0.9)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.app.dependency_overrides.clear()
    yield
    app.app.dependency_overrides.clear()


def test_app_handlers_are_async_except_sync_service_config() -> None:
    assert not inspect.iscoroutinefunction(app.get_service_config)
    assert inspect.iscoroutinefunction(app.healthcheck)
    assert inspect.iscoroutinefunction(app.list_profiles)
    assert inspect.iscoroutinefunction(app.validate)


def test_load_service_config_merges_multiple_yaml_files() -> None:
    with TemporaryDirectory() as temp_dir:
        first_path = Path(temp_dir) / "first.yaml"
        second_path = Path(temp_dir) / "second.yaml"

        first_path.write_text(
            "threadpool:\n"
            "  max_workers: 4\n"
            "providers:\n"
            "  default:\n"
            "    type: huggingface\n"
            "guardrails:\n"
            "  prompt-safety:\n"
            "    guardrail_name: harm_guard\n",
            encoding="utf-8",
        )
        second_path.write_text(
            "providers:\n"
            "  encoder:\n"
            "    type: encoderfile\n"
            "    binary_path: /tmp/fake.encoderfile\n"
            "guardrails:\n"
            "  llm-policy:\n"
            "    guardrail_name: any_llm\n"
            "    validate_kwargs:\n"
            "      policy: no unsafe content\n"
            "threadpool:\n"
            "  max_workers: 8\n",
            encoding="utf-8",
        )

        config = app.load_service_config([first_path, second_path])

    assert sorted(config.providers) == ["default", "encoder"]
    assert sorted(config.guardrails) == ["llm-policy", "prompt-safety"]
    assert config.guardrails["prompt-safety"].guardrail_name.value == "harm_guard"
    assert config.threadpool.max_workers == 8


def test_load_service_config_requires_explicit_threadpool_settings() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "service.yaml"
        path.write_text(
            "guardrails:\n  prompt-safety:\n    guardrail_name: harm_guard\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            app.load_service_config([path])


def test_apply_threadpool_settings_sets_default_thread_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLimiter:
        total_tokens = 40

    limiter = FakeLimiter()
    config = app.ServiceConfig.model_validate(
        {"providers": {}, "guardrails": {}, "threadpool": {"max_workers": 12}}
    )

    monkeypatch.setattr(
        app.to_thread, "current_default_thread_limiter", lambda: limiter
    )
    app.apply_threadpool_settings(config)

    assert limiter.total_tokens == 12


def test_initialize_logging_uses_default_info_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}

    def fake_basic_config(*, level: int) -> None:
        captured["level"] = level

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setattr(app.logging, "basicConfig", fake_basic_config)

    app.initialize_logging()

    assert captured["level"] == logging.INFO


def test_initialize_logging_uses_log_level_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}

    def fake_basic_config(*, level: int) -> None:
        captured["level"] = level

    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setattr(app.logging, "basicConfig", fake_basic_config)

    app.initialize_logging()

    assert captured["level"] == logging.WARNING


@pytest.mark.anyio
async def test_validate_endpoint_uses_profile_configuration_and_request_overrides() -> (
    None
):
    fake_guardrail = FakeGuardrail()
    config = app.ServiceConfig.model_validate(
        {
            "providers": {},
            "guardrails": {
                "llm-policy": {
                    "guardrail_name": "any_llm",
                    "validate_kwargs": {"policy": "stay safe"},
                }
            },
            "threadpool": {"max_workers": 10},
        }
    )

    app.app.dependency_overrides[app.get_service_config] = lambda: config
    app.app.dependency_overrides[app.get_guardrail_instances] = lambda: {
        "llm-policy": fake_guardrail
    }

    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/validate",
            json={
                "profile": "llm-policy",
                "input_text": "hello",
                "validate_kwargs": {"model_id": "openai:gpt-5-nano"},
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "profile": "llm-policy",
        "result": {"valid": True, "explanation": "accepted", "score": 0.9},
    }
    assert fake_guardrail.calls == [
        (("hello",), {"policy": "stay safe", "model_id": "openai:gpt-5-nano"})
    ]


@pytest.mark.anyio
async def test_profiles_endpoint_lists_available_profiles() -> None:
    config = app.ServiceConfig.model_validate(
        {
            "providers": {},
            "guardrails": {
                "policy-a": {"guardrail_name": "any_llm"},
                "policy-b": {
                    "guardrail_name": "harm_guard",
                    "model_id": "hbseong/HarmAug-Guard",
                },
            },
            "threadpool": {"max_workers": 10},
        }
    )

    app.app.dependency_overrides[app.get_service_config] = lambda: config

    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://testserver"
    ) as client:
        response = await client.get("/profiles")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "policy-a", "guardrail_name": "any_llm", "model_id": None},
        {
            "name": "policy-b",
            "guardrail_name": "harm_guard",
            "model_id": "hbseong/HarmAug-Guard",
        },
    ]


@pytest.mark.anyio
async def test_healthcheck_endpoint_returns_ok() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://testserver"
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_validate_endpoint_supports_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_guardrail = FakeGuardrail()
    config = app.ServiceConfig.model_validate(
        {
            "providers": {},
            "guardrails": {
                "llm-policy": {
                    "guardrail_name": "any_llm",
                    "validate_kwargs": {"policy": "stay safe"},
                }
            },
            "threadpool": {"max_workers": 10},
        }
    )

    app.app.dependency_overrides[app.get_service_config] = lambda: config
    app.app.dependency_overrides[app.get_guardrail_instances] = lambda: {
        "llm-policy": fake_guardrail
    }

    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/validate",
            json={
                "profile": "llm-policy",
                "input_text": "hello",
                "validate_kwargs": {"model_id": "openai:gpt-5-nano"},
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "profile": "llm-policy",
        "result": {"valid": True, "explanation": "accepted", "score": 0.9},
    }


@pytest.mark.anyio
async def test_validate_endpoint_serializes_numpy_bool_to_python_bool() -> None:
    class FakeNumpyBoolGuardrail:
        def validate(self, *args, **kwargs):
            return GuardrailOutput(
                valid=np.bool_(True), explanation="accepted", score=0.9
            )

    fake_guardrail = FakeNumpyBoolGuardrail()
    config = app.ServiceConfig.model_validate(
        {
            "providers": {},
            "guardrails": {
                "llm-policy": {
                    "guardrail_name": "any_llm",
                    "validate_kwargs": {"policy": "stay safe"},
                }
            },
            "threadpool": {"max_workers": 10},
        }
    )

    app.app.dependency_overrides[app.get_service_config] = lambda: config
    app.app.dependency_overrides[app.get_guardrail_instances] = lambda: {
        "llm-policy": fake_guardrail
    }

    async with AsyncClient(
        transport=ASGITransport(app=app.app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/validate",
            json={
                "profile": "llm-policy",
                "input_text": "hello",
                "validate_kwargs": {"model_id": "openai:gpt-5-nano"},
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "profile": "llm-policy",
        "result": {"valid": True, "explanation": "accepted", "score": 0.9},
    }


def test_default_service_config_includes_huggingface_model_matrix() -> None:
    config_path = Path(__file__).resolve().parent.parent / "config" / "service.yaml"
    config = app.load_service_config([config_path])

    assert "llamafile" in config.guardrails
    assert "encoderfile" in config.guardrails
    assert sorted(config.providers) == ["encoderfile", "huggingface", "llamafile"]
    configured_guardrails = {
        profile.guardrail_name for profile in config.guardrails.values()
    }
    assert configured_guardrails == set(GuardrailName)
    assert len(config.guardrails) == len(GuardrailName) + 2
    assert config.guardrails["llamafile"].provider == "llamafile"
    assert config.guardrails["encoderfile"].provider == "encoderfile"


def test_get_config_paths_requires_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANY_GUARDRAILS_CONFIG_PATHS", raising=False)

    with pytest.raises(ValueError, match="ANY_GUARDRAILS_CONFIG_PATHS must be set"):
        app.get_config_paths()


def test_provider_configs_allow_base_url_and_forward_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_kwargs: dict[str, dict[str, object]] = {}

    class FakeEncoderfileProvider:
        def __init__(self, **kwargs: object) -> None:
            created_kwargs["encoderfile"] = kwargs

    class FakeLlamafileProvider:
        def __init__(self, **kwargs: object) -> None:
            created_kwargs["llamafile"] = kwargs

    def fake_import_module(module_name: str) -> SimpleNamespace:
        if module_name == "any_guardrail.providers.encoderfile":
            return SimpleNamespace(EncoderfileProvider=FakeEncoderfileProvider)
        if module_name == "any_guardrail.providers.llamafile":
            return SimpleNamespace(LlamafileProvider=FakeLlamafileProvider)
        msg = f"Unexpected module requested: {module_name}"
        raise AssertionError(msg)

    monkeypatch.setattr(models, "import_module", fake_import_module)

    config = app.ServiceConfig.model_validate(
        {
            "providers": {
                "encoder": {
                    "type": "encoderfile",
                    "base_url": "http://encoder.example:8080",
                },
                "llama": {
                    "type": "llamafile",
                    "base_url": "http://llama.example:8081",
                },
            },
            "guardrails": {},
            "threadpool": {"max_workers": 2},
        }
    )

    provider_instances, guardrail_instances = app.build_runtime_instances(config)

    assert guardrail_instances == {}
    assert sorted(provider_instances) == ["encoder", "llama"]
    assert created_kwargs["encoderfile"]["base_url"] == "http://encoder.example:8080"
    assert created_kwargs["llamafile"]["base_url"] == "http://llama.example:8081"
