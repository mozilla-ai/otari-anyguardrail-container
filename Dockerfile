FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Warning: this uses the first interpreter found.
RUN UV_SYSTEM_PYTHON=1 uv sync --frozen --no-dev

COPY src/* ./
COPY config ./config

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
