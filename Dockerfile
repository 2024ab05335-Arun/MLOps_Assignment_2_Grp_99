FROM python:3.13.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libglib2.0-0 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service and a local model (if present)
COPY inference_service.py ./
COPY download_model.py ./
COPY entrypoint.sh ./
# optionally copy a local model if present (will be used if MODEL_URI not set)
COPY baseline_cnn_best.h5 ./ || true

# Expose port and use entrypoint which will download model if MODEL_URI is set
EXPOSE 8000
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
