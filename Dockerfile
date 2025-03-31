# Use PyTorch base image with CUDA 12.1
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements-new.txt* ./

# Install remaining Python dependencies
RUN pip3 install --no-cache-dir -r requirements-new.txt

RUN pip3 install sentence-transformers==2.2.2 
RUN pip install "numpy<2.0"
RUN pip install docling

RUN docling-tools models download
ENV OMP_NUM_THREADS=4

# Copy project files
COPY .chainlit/config.toml ./.chainlit/config.toml
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create .env file placeholder
RUN touch .env

# Create entrypoint script
RUN echo '#!/bin/bash\n\
mkdir -p /app/backend/data_images/xyz-abc-qwerty/\n\
uvicorn backend.appv4:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 500 --loop asyncio &\n\
echo "Waiting for backend to start..."\n\
until curl -s http://localhost:8001/docs > /dev/null; do\n\
  sleep 20\n\
done\n\
echo "Sending curl request..."\n\
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"what is the document about?\", \"data_path\": \"data\", \"user_id\": \"abc-123\", \"input_files\": [[\"/app/backend/2408.09869v5.pdf\", \"2408.09869v5.pdf\"]], \"collection\": \"xyz-abc-qwerty\"}"\n\
chainlit run frontend/chainlit_app.py --port 8000 --host 0.0.0.0\n' \
> /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8000 8001

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

