FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dev]"

# Copy remaining source
COPY . .

EXPOSE 8000

# Listen on a fixed port (8000) to match Railway's service-domain target.
# Bind to :: (IPv6 any) — dual-stack on Linux also accepts IPv4, and Railway's
# internal proxy uses IPv6, so 0.0.0.0 alone causes a 502 Bad Gateway.
CMD ["uvicorn", "api:app", "--host", "::", "--port", "8000"]
