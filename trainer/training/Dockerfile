FROM tensorflow/tensorflow:2.16.1

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

# Copy source code
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY tests/ /app/tests/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.cli", "--help"]
