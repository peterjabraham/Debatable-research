FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dev]"

# Copy remaining source
COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
