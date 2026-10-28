# syntax=docker/dockerfile:1
# Multi-stage build: dependencies are installed into a virtualenv in the build stage and copied
# into a slim runtime stage that runs as a non-root user.
# (A virtualenv rather than `pip install --user`: /root/.local is not readable by a non-root user.)

FROM python:3.13-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt
# tiktoken downloads its BPE file on first use; bake it in so cold starts don't fetch it from the internet.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

FROM python:3.13-slim
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    PORT=8080
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/tiktoken /opt/tiktoken

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
RUN mkdir -p /app/data && chown appuser:appuser /app/data
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser static ./static
# Structured data for the Data agent (233 KB, read-only). The vector index lives in Pinecone, not the image.
COPY --chown=appuser:appuser data/agentlab.db ./data/agentlab.db

USER appuser
EXPOSE 8080
# Cloud Run injects PORT. --proxy-headers: requests arrive through Google's front end.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75"]
