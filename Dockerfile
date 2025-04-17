FROM python:3.10-slim

WORKDIR /app

# Install system build deps (if you need to compile anything)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python libs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Ensure your scripts are executable
RUN chmod +x scripts/run_pipeline.sh scripts/fetch_models.sh

# Default entrypoint runs your pipeline
ENTRYPOINT ["bash", "scripts/run_pipeline.sh"]
