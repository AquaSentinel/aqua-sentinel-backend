FROM python:3.11-slim

# Disable .pyc & force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for ONNX, Pillow, OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libgomp1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy code + models
COPY . .

# Expose port
EXPOSE 8000

# Run directly (good for testing)
CMD ["python", "app.py"]
