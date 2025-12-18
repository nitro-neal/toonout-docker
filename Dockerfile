# ToonOut Background Removal API
# Supports both CPU and GPU inference

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/BiRefNet \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone BiRefNet repository
RUN git clone --depth 1 https://github.com/ZhengPeng7/BiRefNet.git /app/BiRefNet

# Download ToonOut weights from HuggingFace
RUN mkdir -p /app/weights && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='joelseytre/toonout', filename='birefnet_finetuned_toonout.pth', local_dir='/app/weights', local_dir_use_symlinks=False)"

# Copy application code
COPY main.py .

# Expose port
EXPOSE 1337

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:1337/ping')" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1337"]
