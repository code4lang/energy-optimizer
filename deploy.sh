#!/bin/bash

# Set variables
GIT_REPO="git@github.com:code4lang/energy-optimizer.git"
DOCKER_IMAGE="brunodefebrer/energy-optimizer:latest"

# Push changes to GitHub
echo "🚀 Pushing changes to GitHub..."
git add .
git commit -m "Auto-deploy: $(date)"
git push origin main

# Login to Docker Hub using environment variables
echo "🔐 Logging into Docker Hub..."
docker login

# Check if the image already exists locally
if docker image inspect $DOCKER_IMAGE >/dev/null 2>&1; then
    echo "✅ Existing image detected, skipping rebuild..."
else
    echo "🛠️ Building new Docker image..."
    docker build -t $DOCKER_IMAGE .
fi

# Push Docker image
echo "📦 Pushing Docker image to Docker Hub..."
docker push $DOCKER_IMAGE

# Clean up unused Docker data
echo "🗑️ Cleaning up unused images and containers..."
docker system prune -a -f

echo "✅ Deployment complete!"

