FROM python:3.11-slim

# Set working directory
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE = 1
ENV PYTHONUNBUFFERED = 1

# Get UV binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install needed system dependencies
COPY requirements.txt .

# Install dependencies into the system environment using uv pip
# --system installs globally inside the container (no virtualenv needed in Docker)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy dataset, cached joblib models, and source code
COPY cache/ ./cache/
COPY data/ ./data/
COPY . .

# Exposing PORT 8000 for FastAPI
EXPOSE 8000

# Command to launch server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]