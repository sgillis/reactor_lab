FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN groupadd --system reactor \
    && useradd --system --gid reactor --create-home reactor \
    && mkdir -p /tmp/matplotlib \
    && chown reactor:reactor /tmp/matplotlib

COPY --from=builder /app/.venv ./.venv
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
COPY --chmod=0755 docker/api-entrypoint.sh /usr/local/bin/reactor-lab-api

USER reactor

EXPOSE 8000

CMD ["reactor-lab-api"]
