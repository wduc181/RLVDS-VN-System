FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU Paddle is the default so the image runs on normal Docker installs.
# Override with:
#   PADDLE_PACKAGE=paddlepaddle-gpu==2.6.2 docker compose build
ARG PADDLE_PACKAGE=paddlepaddle==2.6.2
RUN python -m pip install --upgrade pip \
    && awk '!/^paddlepaddle-gpu==/' requirements.txt > /tmp/requirements-docker.txt \
    && python -m pip install "${PADDLE_PACKAGE}" \
    && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r /tmp/requirements-docker.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
