# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps for requests/ssl if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY moderation_server.py .

ENV PORT=8080

# Use gunicorn for production serving
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "moderation_server:app", "--workers", "2", "--threads", "4", "--timeout", "120"]
