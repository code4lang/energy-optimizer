docker volume create energy-data
docker run -d --name energy-container -v energy-data:/app/data energy-optimizer
docker run -d --name energy-container -v $(pwd)/data:/app/data energy-optimizer


