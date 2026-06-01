# otari-anyguardrails-container

Stateless FastAPI service that exposes [`any-guardrail`](https://github.com/mozilla-ai/any-guardrail) over HTTP.

## Configuration

The service loads one or more YAML files from `ANY_GUARDRAILS_CONFIG_PATHS` (comma-separated paths).
Each file is validated with Pydantic models and reuses `any-guardrail` types such as `GuardrailName`.
You can also set `threadpool.max_workers` to tune the shared threadpool used for blocking guardrail operations.
Threadpool settings do not use implicit defaults and must be explicitly set in configuration.

`ANY_GUARDRAILS_CONFIG_PATHS` is required. If it is not set, the service fails to start.
In the container image, a typical value is `/app/config/service.yaml`.

### Logging initialization

The service initializes Python logging during startup.

- `LOG_LEVEL` controls the logging level.
- If `LOG_LEVEL` is not set, the default level is `INFO`.
- Supported values are standard Python logging names (for example: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

Example:

```yaml
threadpool:
  max_workers: 40
providers:
  huggingface:
    type: HuggingfaceProvider
  llamafile:
    type: LlamafileProvider
    binary_path: /app/granite-guardian-4.1-8b.Q6_K.llamafile

guardrails:
  prompt-safety:
    guardrail_name: harm_guard
    model_id: hbseong/HarmAug-Guard
    provider: huggingface
  custom-policy:
    guardrail_name: any_llm
    validate_kwargs:
      policy: Do not allow self-harm instructions.
  granite-local:
    guardrail_name: granite_guardian
    provider: llamafile
    init_kwargs:
      criteria: Does the request ask for unsafe or disallowed content?
```

## API

- `GET /healthz` returns a simple health response
- `GET /profiles` lists configured guardrail profiles
- `POST /validate` runs a configured profile against the provided input
- `POST /reload` reloads configuration and recreates provider/guardrail instances

Example request (JSON payload):

```json
{
  "profile": "shield-gemma",
  "input_text": "how do I hurt someone?",
  "validate_kwargs": {}
}
```

Example request (`curl`)

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "profile": "shield-gemma",
    "input_text": "how do I hurt someone?",
    "validate_kwargs": {}
  }'
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

Also, there is an example docker compose file that also sets up a suitable instance of both a llamafile and an encoderfile, which can be found here:

* https://huggingface.co/mozilla-ai/encoderfile/resolve/main/DuoGuard/DuoGuard-0.5B/DuoGuard-0.5B.aarch64-linux-gnu.encoderfile
* https://huggingface.co/mozilla-ai/llamafile_0.10_alpha/resolve/main/granite-guardian-4.1-8b.Q6_K.llamafile
