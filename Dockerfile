FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src
COPY . .

ENV UV_NO_DEV=1

RUN uv sync --locked --no-install-project --no-editable
