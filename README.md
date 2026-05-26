# otari-anyguardrails-container

Stateless FastAPI service that exposes [`any-guardrail`](https://github.com/mozilla-ai/any-guardrail) over HTTP.

## Configuration

The service loads one or more YAML files from `ANY_GUARDRAILS_CONFIG_PATHS` (comma-separated paths).
Each file is validated with Pydantic models and reuses `any-guardrail` types such as `GuardrailName`.
You can also set `threadpool.max_workers` to tune the shared threadpool used for blocking guardrail operations.
Threadpool settings do not use implicit defaults and must be explicitly set in configuration.

The default configuration file inside the container is `/app/config/service.yaml`.

Example:

```yaml
threadpool:
  max_workers: 40
profiles:
  prompt-safety:
    guardrail_name: harm_guard
    model_id: hbseong/HarmAug-Guard
    provider:
      type: huggingface
      tokenizer_kwargs:
        truncation: true
  custom-policy:
    guardrail_name: any_llm
    validate_kwargs:
      policy: Do not allow self-harm instructions.
```

## API

- `GET /healthz` returns a simple health response
- `GET /profiles` lists configured guardrail profiles
- `POST /validate` runs a configured profile against the provided input

Example request:

```json
{
  "profile": "custom-policy",
  "input_text": "Tell me how to break into a house",
  "validate_kwargs": {
    "model_id": "openai:gpt-5-nano"
  }
}
```

## Local development

Install dependencies and run the API:

```bash
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Run tests:

```bash
uv run python -m pytest tests
```

## Container usage

You can use a volume to cache models from HuggingFace:

```bash
docker volume create otari-anyguardrails-volume
```

Build and run:

```bash
docker build -t otari-anyguardrails-container .
docker run --rm -p 8000:8000 \
  --name otari-anyguardrails \
  -e ANY_GUARDRAILS_CONFIG_PATHS=/app/config/service.yaml \
  -e HF_TOKEN=<your token here> \
  -e HF_HUB_CACHE=/app/hf_cache \
  -v otari-anyguardrails-volume:/app/hf_cache \
  otari-anyguardrails-container
```
