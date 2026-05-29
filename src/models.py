from __future__ import annotations

from importlib import import_module
from typing import Any, Literal

from any_guardrail import AnyGuardrail, GuardrailName, HuggingFaceProvider
from pydantic import BaseModel, ConfigDict, Field


class HuggingFaceProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["huggingface", "HuggingFaceProvider", "HuggingfaceProvider"] = "huggingface"
    tokenizer_id: str | None = None
    trust_remote_code: bool = False
    device: str | None = None
    torch_dtype: Any | None = None
    cache_dir: str | None = None
    revision: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self) -> HuggingFaceProvider:
        return HuggingFaceProvider(
            tokenizer_id=self.tokenizer_id,
            trust_remote_code=self.trust_remote_code,
            device=self.device,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
            revision=self.revision,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )


class EncoderfileProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["encoderfile", "EncoderfileProvider"] = "encoderfile"
    base_url: str | None = None
    binary_path: str | None = None
    port: int | None = None
    host: str = "127.0.0.1"
    startup_timeout: float = 60.0
    request_timeout: float = 60.0
    cache_dir: str | None = None
    encoderfile_repo: str = "mozilla-ai/encoderfile"

    def build(self) -> Any:
        try:
            encoderfile_module = import_module("any_guardrail.providers.encoderfile")
            encoderfile_provider_class = getattr(encoderfile_module, "EncoderfileProvider")
        except (ImportError, AttributeError) as exc:
            msg = (
                "EncoderfileProvider is not available in the installed any-guardrail package. "
                "Install a version that provides it (or any-guardrail[encoderfile])."
            )
            raise ValueError(msg) from exc

        return encoderfile_provider_class(
            base_url=self.base_url,
            binary_path=self.binary_path,
            port=self.port,
            host=self.host,
            startup_timeout=self.startup_timeout,
            request_timeout=self.request_timeout,
            cache_dir=self.cache_dir,
            encoderfile_repo=self.encoderfile_repo,
        )


class LlamafileProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["llamafile", "LlamafileProvider"] = "llamafile"
    base_url: str | None = None
    binary_path: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    port: int | None = None
    host: str = "127.0.0.1"
    startup_timeout: float = 120.0
    request_timeout: float = 120.0
    cache_dir: str | None = None
    n_gpu_layers: int | None = None
    context_size: int | None = None
    extra_args: list[str] | None = None

    def build(self) -> Any:
        try:
            llamafile_module = import_module("any_guardrail.providers.llamafile")
            llamafile_provider_class = getattr(llamafile_module, "LlamafileProvider")
        except (ImportError, AttributeError) as exc:
            msg = (
                "LlamafileProvider is not available in the installed any-guardrail package. "
                "Install a version that provides it (or any-guardrail[llamafile])."
            )
            raise ValueError(msg) from exc

        return llamafile_provider_class(
            base_url=self.base_url,
            binary_path=self.binary_path,
            repo_id=self.repo_id,
            filename=self.filename,
            port=self.port,
            host=self.host,
            startup_timeout=self.startup_timeout,
            request_timeout=self.request_timeout,
            cache_dir=self.cache_dir,
            n_gpu_layers=self.n_gpu_layers,
            context_size=self.context_size,
            extra_args=self.extra_args,
        )


ProviderConfig = HuggingFaceProviderConfig | EncoderfileProviderConfig | LlamafileProviderConfig


class GuardrailProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guardrail_name: GuardrailName
    model_id: str | None = None
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    validate_kwargs: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None

    def build_guardrail(self, providers: dict[str, Any]) -> Any:
        create_kwargs = dict(self.init_kwargs)
        if self.model_id is not None and "model_id" not in create_kwargs:
            create_kwargs["model_id"] = self.model_id

        provider_instance = None
        if self.provider is not None:
            provider_instance = providers.get(self.provider)
            if provider_instance is None:
                msg = f"Unknown provider reference: {self.provider}"
                raise ValueError(msg)

        try:
            if provider_instance is None:
                return AnyGuardrail.create(self.guardrail_name, **create_kwargs)
            return AnyGuardrail.create(self.guardrail_name, provider=provider_instance, **create_kwargs)
        except TypeError as exc:
            msg = f"Invalid guardrail initialization for {self.guardrail_name.value}: {exc}"
            raise ValueError(msg) from exc


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class ThreadpoolConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        max_workers: int = Field(ge=1)

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    guardrails: dict[str, GuardrailProfileConfig] = Field(default_factory=dict)
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