# =============================================================
# Stage 1 — Builder
# Install dependencies in a separate image to avoid loading compilers and headers in the final image.
# =============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =============================================================
# Stage 2 — Runtime
# =============================================================
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Staffing BI Assistant"
LABEL org.opencontainers.image.description="Text-to-SQL + RAG + RLS + Semantic Cache"

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy compiled dependencies from builder
COPY --from=builder /install /usr/local

# Copy only the source code — data/ comes via volume in docker-compose
COPY src/ ./src/

# ChromaDB persistence directory (overlaid by named volume)
RUN mkdir -p /app/.chroma /app/data && chown -R appuser:appgroup /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Names aligned with docker-compose (DATABASE_PATH, PERMISSIONS_PATH)
    DATABASE_PATH=/app/data/staffing.db \
    PERMISSIONS_PATH=/app/data/user_permissions.json \
    DOCUMENTS_PATH=/app/data/documents \
    CHROMA_PERSIST_PATH=/app/.chroma \
    API_HOST=0.0.0.0 \
    API_PORT=8000

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "src.main", "--mode", "web"]