FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# TF 2.10 has keras 2.10 bundled with proper layer_utils support

# Copy source code
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY tests/ /app/tests/

# Set Python path
ENV PYTHONPATH=/app

# TF 2.15 uses Keras 2.15 by default (legacy Keras)
# Compatible with efficientnet-lite-keras, TFMOT, and TFLite INT8

# Environment for multi-GPU
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=1

# Default command
CMD ["python", "-m", "src.cli", "--help"]
