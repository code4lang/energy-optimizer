# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --index-url https://pypi.org/simple --no-cache-dir langchain llama-index openai transformers torch pymilvus requests \
            peft bitsandbytes accelerate datasets paho-mqtt influxdb-client \
            statsmodels scikit-learn pandas numpy matplotlib openmeteo-py geopy

# Set the default command to run the script
CMD ["python", "energy_optimizer.py"]

