# Use the official Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Detect system and install relevant dependencies
RUN arch=$(uname -m) && \
    if [ "$arch" = "arm64" ]; then \
        echo "Detected Mac M-Series (Apple Silicon)"; \
        pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$arch" = "x86_64" ]; then \
        gpu_info=$(lspci | grep -i nvidia || true) && \
        if [ -n "$gpu_info" ]; then \
            echo "Detected NVIDIA GPU"; \
            pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
        else \
            amd_info=$(lspci | grep -i amd || true) && \
            if [ -n "$amd_info" ]; then \
                echo "Detected AMD GPU"; \
                pip install --no-cache-dir torch torchvision torchaudio rocm-pytorch; \
            else \
                echo "No GPU detected, installing CPU-only dependencies"; \
                pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu; \
            fi; \
        fi; \
    else \
        echo "Unsupported architecture, defaulting to CPU install"; \
        pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install other dependencies
RUN pip install --index-url https://pypi.org/simple --no-cache-dir langchain llama-index openai transformers pymilvus requests \
            peft bitsandbytes accelerate datasets paho-mqtt influxdb-client \
            statsmodels scikit-learn pandas numpy matplotlib openmeteo-py geopy

# Set default command to run the script
CMD ["python", "energy_optimizer.py"]

