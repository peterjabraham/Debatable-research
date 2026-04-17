FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dev]"

# Copy remaining source
COPY . .

EXPOSE 8000

# Listen on a fixed port (8000). Railway's service-domain form asks which
# port the app listens on and routes edge traffic there, so this must match.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
