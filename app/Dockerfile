FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    RAIL_MAX_FILE_BYTES=31457280 PORT=8080
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 fleet
COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist
USER fleet
EXPOSE 8080
CMD ["sh", "-c", "exec python -m uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
