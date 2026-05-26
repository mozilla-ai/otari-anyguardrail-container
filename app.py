from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from anyio import to_thread
from any_guardrail import AnyGuardrail, GuardrailName, GuardrailOutput, HuggingFaceProvider
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "service.yaml"


class HuggingFaceProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "huggingface"
    tokenizer_id: str | None = None
    trust_remote_code: bool = False
    device: str | None = None
    cache_dir: str | None = None
    revision: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self) -> HuggingFaceProvider:
        if self.type != "huggingface":
            msg = f"Unsupported provider type: {self.type}"
            raise ValueError(msg)

        return HuggingFaceProvider(
            tokenizer_id=self.tokenizer_id,
            trust_remote_code=self.trust_remote_code,
            device=self.device,
            cache_dir=self.cache_dir,
            revision=self.revision,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )


class GuardrailProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guardrail_name: GuardrailName
    model_id: str | None = None
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    validate_kwargs: dict[str, Any] = Field(default_factory=dict)
    provider: HuggingFaceProviderConfig | None = None

    def build_guardrail(self) -> Any:
        create_kwargs = dict(self.init_kwargs)
        if self.model_id is not None and "model_id" not in create_kwargs:
            create_kwargs["model_id"] = self.model_id

        provider = self.provider.build() if self.provider else None
        return AnyGuardrail.create(self.guardrail_name, provider=provider, **create_kwargs)


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class ThreadpoolConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        max_workers: int = Field(ge=1)

    profiles: dict[str, GuardrailProfileConfig] = Field(default_factory=dict)
    threadpool: ThreadpoolConfig


class GuardrailProfileSummary(BaseModel):
    name: str
    guardrail_name: GuardrailName
    model_id: str | None = None


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    input_text: str | list[str] | None = None
    validate_kwargs: dict[str, Any] = Field(default_factory=dict)


class ValidateResponse(BaseModel):
    profile: str
    result: dict[str, Any] | list[dict[str, Any]]


def get_config_paths() -> list[Path]:
    raw_value = os.getenv("ANY_GUARDRAILS_CONFIG_PATHS", str(DEFAULT_CONFIG_PATH))
    return [Path(path.strip()) for path in raw_value.split(",") if path.strip()]


_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")


def resolve_env_vars(data: Any) -> Any:
    """Recursively resolve ``${VAR_NAME}`` references in *data* to their environment variable values.

    Any string value that exactly matches the pattern ``${VAR_NAME}`` is replaced with the value of
    the ``VAR_NAME`` environment variable.  A :exc:`ValueError` is raised if the variable is not set,
    so configuration errors are surfaced early rather than at runtime.  All other values (non-string
    scalars, dicts, lists) are traversed or returned unchanged.
    """
    if isinstance(data, str):
        match = _ENV_VAR_PATTERN.match(data)
        if match:
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                msg = f"Environment variable referenced in configuration is not set: {var_name}"
                raise ValueError(msg)
            return value
    elif isinstance(data, dict):
        return {key: resolve_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    return data


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        msg = f"Configuration file must contain a YAML mapping: {path}"
        raise ValueError(msg)
    return resolve_env_vars(loaded)


def load_service_config(paths: list[Path] | None = None) -> ServiceConfig:
    merged_profiles: dict[str, GuardrailProfileConfig] = {}
    merged_threadpool: ServiceConfig.ThreadpoolConfig | None = None

    for path in paths or get_config_paths():
        config = ServiceConfig.model_validate(load_yaml(path))
        duplicate_profiles = set(merged_profiles).intersection(config.profiles)
        if duplicate_profiles:
            duplicates = ", ".join(sorted(duplicate_profiles))
            msg = f"Duplicate profile names found across configuration files: {duplicates}"
            raise ValueError(msg)
        merged_profiles.update(config.profiles)
        merged_threadpool = config.threadpool

    if merged_threadpool is None:
        msg = "Threadpool configuration must be explicitly set."
        raise ValueError(msg)

    return ServiceConfig(profiles=merged_profiles, threadpool=merged_threadpool)


def apply_threadpool_settings(config: ServiceConfig) -> None:
    limiter = to_thread.current_default_thread_limiter()
    limiter.total_tokens = config.threadpool.max_workers


def get_service_config() -> ServiceConfig:
    try:
        return load_service_config()
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def serialize_result(result: GuardrailOutput[Any, Any, Any] | list[GuardrailOutput[Any, Any, Any]]) -> Any:
    if isinstance(result, list):
        return [item.model_dump(mode="json") for item in result]
    return result.model_dump(mode="json")


app = FastAPI(title="Any Guardrail Service", version="0.1.0")


@app.on_event("startup")
async def configure_threadpool() -> None:
    apply_threadpool_settings(load_service_config())


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/profiles", response_model=list[GuardrailProfileSummary])
async def list_profiles(config: ServiceConfig = Depends(get_service_config)) -> list[GuardrailProfileSummary]:
    return [
        GuardrailProfileSummary(
            name=name,
            guardrail_name=profile.guardrail_name,
            model_id=profile.model_id,
        )
        for name, profile in sorted(config.profiles.items())
    ]


@app.post("/validate", response_model=ValidateResponse)
async def validate(request: ValidateRequest, config: ServiceConfig = Depends(get_service_config)) -> ValidateResponse:
    profile = config.profiles.get(request.profile)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown profile: {request.profile}")

    guardrail = await run_in_threadpool(profile.build_guardrail)
    validate_kwargs = {**profile.validate_kwargs, **request.validate_kwargs}

    try:
        if request.input_text is None:
            result = await run_in_threadpool(guardrail.validate, **validate_kwargs)
        else:
            result = await run_in_threadpool(guardrail.validate, request.input_text, **validate_kwargs)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ValidateResponse(profile=request.profile, result=serialize_result(result))
