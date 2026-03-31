FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source (root-level modules are imported by `src.*` compatibility shims)
COPY src/     ./src/
COPY app/     ./app/
COPY dags/    ./dags/
COPY data/    ./data/
COPY *.py     ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1