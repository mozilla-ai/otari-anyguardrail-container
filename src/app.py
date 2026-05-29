from __future__ import annotations

import logging
import os
from asyncio import Lock
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from anyio import to_thread
from any_guardrail import GuardrailOutput
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import ValidationError
from models import (
    GuardrailProfileConfig,
    GuardrailProfileSummary,
    ProviderConfig,
    ServiceConfig,
    ValidateRequest,
    ValidateResponse,
)
from starlette.concurrency import run_in_threadpool


def get_config_paths() -> list[Path]:
    raw_value = os.getenv("ANY_GUARDRAILS_CONFIG_PATHS")
    if raw_value is None:
        msg = "ANY_GUARDRAILS_CONFIG_PATHS must be set."
        raise ValueError(msg)
    return [Path(path.strip()) for path in raw_value.split(",") if path.strip()]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        msg = f"Configuration file must contain a YAML mapping: {path}"
        raise ValueError(msg)
    return loaded


def load_service_config(paths: list[Path] | None = None) -> ServiceConfig:
    merged_providers: dict[str, ProviderConfig] = {}
    merged_guardrails: dict[str, GuardrailProfileConfig] = {}
    merged_threadpool: ServiceConfig.ThreadpoolConfig | None = None

    for path in paths or get_config_paths():
        config = ServiceConfig.model_validate(load_yaml(path))
        duplicate_providers = set(merged_providers).intersection(config.providers)
        if duplicate_providers:
            duplicates = ", ".join(sorted(duplicate_providers))
            msg = f"Duplicate provider names found across configuration files: {duplicates}"
            raise ValueError(msg)
        duplicate_guardrails = set(merged_guardrails).intersection(config.guardrails)
        if duplicate_guardrails:
            duplicates = ", ".join(sorted(duplicate_guardrails))
            msg = f"Duplicate guardrail names found across configuration files: {duplicates}"
            raise ValueError(msg)
        merged_providers.update(config.providers)
        merged_guardrails.update(config.guardrails)
        merged_threadpool = config.threadpool

    if merged_threadpool is None:
        msg = "Threadpool configuration must be explicitly set."
        raise ValueError(msg)

    return ServiceConfig(providers=merged_providers, guardrails=merged_guardrails, threadpool=merged_threadpool)


def close_provider_instances(provider_instances: dict[str, Any]) -> None:
    for provider_name, provider_instance in provider_instances.items():
        close_method = getattr(provider_instance, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                logging.exception("Failed to close provider %s", provider_name)


def build_runtime_instances(config: ServiceConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    provider_instances: dict[str, Any] = {}
    guardrail_instances: dict[str, Any] = {}

    try:
        for provider_name, provider_config in config.providers.items():
            provider_instances[provider_name] = provider_config.build()

        for guardrail_name, guardrail_config in config.guardrails.items():
            guardrail_instances[guardrail_name] = guardrail_config.build_guardrail(provider_instances)
    except Exception:
        close_provider_instances(provider_instances)
        raise

    return provider_instances, guardrail_instances


def apply_threadpool_settings(config: ServiceConfig) -> None:
    limiter = to_thread.current_default_thread_limiter()
    limiter.total_tokens = config.threadpool.max_workers


def initialize_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name, logging.INFO)
    logging.basicConfig(level=level)


def get_service_config(request: Request) -> ServiceConfig:
    config = getattr(request.app.state, "service_config", None)
    if config is None:
        msg = "Service configuration is not initialized."
        raise HTTPException(status_code=500, detail=msg)
    return config


def get_guardrail_instances(request: Request) -> dict[str, Any]:
    guardrail_instances = getattr(request.app.state, "guardrail_instances", None)
    if guardrail_instances is None:
        msg = "Guardrails are not initialized."
        raise HTTPException(status_code=500, detail=msg)
    return guardrail_instances


async def reload_service_state(app: FastAPI, paths: list[Path] | None = None) -> ServiceConfig:
    config = load_service_config(paths)
    provider_instances, guardrail_instances = await run_in_threadpool(build_runtime_instances, config)
    logging.info(f"Loaded configuration with providers: {provider_instances}")
    logging.info(f"Loaded configuration with guardrails: {guardrail_instances}")
    [logging.info(f"Guardrail {name} using provider: {instance.provider}") for name, instance in guardrail_instances.items()]
    apply_threadpool_settings(config)

    reload_lock: Lock | None = getattr(app.state, "reload_lock", None)
    if reload_lock is None:
        reload_lock = Lock()
        app.state.reload_lock = reload_lock

    async with reload_lock:
        old_providers = getattr(app.state, "provider_instances", {})
        app.state.service_config = config
        app.state.provider_instances = provider_instances
        app.state.guardrail_instances = guardrail_instances

    await run_in_threadpool(close_provider_instances, old_providers)
    return config


def serialize_result(result: GuardrailOutput[Any, Any, Any] | list[GuardrailOutput[Any, Any, Any]]) -> Any:
    if isinstance(result, list):
        return [item.model_dump(mode="json") for item in result]
    return result.model_dump(mode="json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_logging()
    try:
        await reload_service_state(app)
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    yield
    await run_in_threadpool(close_provider_instances, getattr(app.state, "provider_instances", {}))


app = FastAPI(title="Any Guardrail Service", version="0.1.0", lifespan=lifespan)


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
        for name, profile in sorted(config.guardrails.items())
    ]


@app.post("/validate", response_model=ValidateResponse)
async def validate(
    request: ValidateRequest,
    config: ServiceConfig = Depends(get_service_config),
    guardrail_instances: dict[str, Any] = Depends(get_guardrail_instances),
) -> ValidateResponse:
    profile = config.guardrails.get(request.profile)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown profile: {request.profile}")

    guardrail = guardrail_instances.get(request.profile)
    if guardrail is None:
        raise HTTPException(status_code=500, detail=f"Guardrail not initialized for profile: {request.profile}")

    validate_kwargs = {**profile.validate_kwargs, **request.validate_kwargs}

    try:
        if request.input_text is None:
            result = await run_in_threadpool(guardrail.validate, **validate_kwargs)
        else:
            result = await run_in_threadpool(guardrail.validate, request.input_text, **validate_kwargs)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ValidateResponse(profile=request.profile, result=serialize_result(result))


@app.post("/reload")
async def reload(request: Request) -> dict[str, Any]:
    try:
        config = await reload_service_state(request.app)
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "providers": len(config.providers),
        "guardrails": len(config.guardrails),
    }
