# Stage 1 — Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Avoid Python buffering logs
ENV PYTHONUNBUFFERED=1

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Stage 2 — Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3 — Copy application code
COPY main.py .
COPY iforest_model.pkl .
COPY iforest_scaler.pkl .
COPY xgboost_model.pkl .
COPY lstm_model.h5 .

# Stage 4 — Expose & Run
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
