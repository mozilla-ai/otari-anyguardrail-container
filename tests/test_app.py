import inspect
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from any_guardrail import GuardrailName, GuardrailOutput
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

import app


class FakeGuardrail:
    def __init__(self) -> None:
        self.calls = []

    def validate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return GuardrailOutput(valid=True, explanation="accepted", score=0.9)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> None:
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

    assert sorted(config.profiles) == ["llm-policy", "prompt-safety"]
    assert config.profiles["prompt-safety"].guardrail_name.value == "harm_guard"
    assert config.threadpool.max_workers == 8


def test_load_service_config_requires_explicit_threadpool_settings() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "service.yaml"
        path.write_text(
            "profiles:\n"
            "  prompt-safety:\n"
            "    guardrail_name: harm_guard\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            app.load_service_config([path])


def test_apply_threadpool_settings_sets_default_thread_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLimiter:
        total_tokens = 40

    limiter = FakeLimiter()
    config = app.ServiceConfig.model_validate({"profiles": {}, "threadpool": {"max_workers": 12}})

    monkeypatch.setattr(app.to_thread, "current_default_thread_limiter", lambda: limiter)
    app.apply_threadpool_settings(config)

    assert limiter.total_tokens == 12


@pytest.mark.anyio
async def test_validate_endpoint_uses_profile_configuration_and_request_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_guardrail = FakeGuardrail()
    config = app.ServiceConfig.model_validate(
        {
            "profiles": {
                "llm-policy": {
                    "guardrail_name": "any_llm",
                    "validate_kwargs": {"policy": "stay safe"},
                }
            },
            "threadpool": {"max_workers": 10},
        }
    )

    create_calls: list[tuple[tuple, dict]] = []

    def fake_create(*args, **kwargs):
        create_calls.append((args, kwargs))
        return fake_guardrail

    app.app.dependency_overrides[app.get_service_config] = lambda: config
    monkeypatch.setattr(app.AnyGuardrail, "create", fake_create)

    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://testserver") as client:
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
    assert len(create_calls) == 1
    assert fake_guardrail.calls == [(("hello",), {"policy": "stay safe", "model_id": "openai:gpt-5-nano"})]


@pytest.mark.anyio
async def test_profiles_endpoint_lists_available_profiles() -> None:
    config = app.ServiceConfig.model_validate(
        {
            "profiles": {
                "policy-a": {"guardrail_name": "any_llm"},
                "policy-b": {"guardrail_name": "harm_guard", "model_id": "hbseong/HarmAug-Guard"},
            },
            "threadpool": {"max_workers": 10},
        }
    )

    app.app.dependency_overrides[app.get_service_config] = lambda: config

    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://testserver") as client:
        response = await client.get("/profiles")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "policy-a", "guardrail_name": "any_llm", "model_id": None},
        {"name": "policy-b", "guardrail_name": "harm_guard", "model_id": "hbseong/HarmAug-Guard"},
    ]


@pytest.mark.anyio
async def test_healthcheck_endpoint_returns_ok() -> None:
    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://testserver") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_validate_endpoint_supports_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_guardrail = FakeGuardrail()
    config = app.ServiceConfig.model_validate(
        {
            "profiles": {
                "llm-policy": {
                    "guardrail_name": "any_llm",
                    "validate_kwargs": {"policy": "stay safe"},
                }
            },
            "threadpool": {"max_workers": 10},
        }
    )

    app.app.dependency_overrides[app.get_service_config] = lambda: config
    monkeypatch.setattr(app.AnyGuardrail, "create", lambda *args, **kwargs: fake_guardrail)

    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://testserver") as client:
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
    config = app.load_service_config([app.DEFAULT_CONFIG_PATH])

    assert "llamafile" in config.profiles
    assert "encoderfile" in config.profiles
    configured_guardrails = {profile.guardrail_name for profile in config.profiles.values()}
    assert configured_guardrails == set(GuardrailName)
    assert len(config.profiles) == len(GuardrailName) + 2
    assert config.profiles["llamafile"].validate_kwargs["api_base"] == "http://llamafile.guardrails.example.com:8080/v1"
    assert config.profiles["encoderfile"].validate_kwargs["api_base"] == "http://encoderfile.guardrails.example.com:8080/v1"
