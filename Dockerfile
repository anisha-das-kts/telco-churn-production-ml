# FROM python:3.12-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PIP_NO_CACHE_DIR=1

# WORKDIR /app

# COPY requirements.txt .

# RUN python -m pip install --upgrade pip \
#     && python -m pip install -r requirements.txt

# COPY src ./src
# COPY models/production/model.joblib ./models/production/model.joblib

# RUN mkdir -p /app/artifacts/logs \
#     && adduser --disabled-password --gecos "" appuser \
#     && chown -R appuser:appuser /app

# USER appuser

# EXPOSE 8000

# CMD [
#     "uvicorn",
#     "src.serving.app:app",
#     "--host",
#     "0.0.0.0",
#     "--port",
#     "8000"
# ]

# Docker configuration was prepared as an optional deployment artifact but was not locally executed because Docker Desktop was unavailable.