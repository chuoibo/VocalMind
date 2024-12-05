#!/bin/bash

# File: scripts/dev-gpu.sh

# Description: Script to run Docker Compose for development with GPU support.

echo "Creating logs dir..."
mkdir -p ./logs

echo "Grant permission for log dir and common dir ..."
sudo chmod -R 777 ./logs
sudo chmod -R 777 ../common_dir

# Check if the first argument is "rebuild"
if [ "$1" = "rebuild" ]; then
    REBUILD="true"
else
    REBUILD="false"
fi

# Stop any running containers
echo "Stopping any existing containers..."
docker compose -f docker-compose-dev.yaml -f docker-compose-dev-gpu.yaml down

# Build and start the containers, with or without --build
if [ "$REBUILD" = "true" ]; then
    echo "Rebuilding and starting the containers with GPU support..."
    docker compose -f docker-compose-dev.yaml -f docker-compose-dev-gpu.yaml up --build -d
else
    echo "Starting the containers with GPU support without rebuilding..."
    docker compose -f docker-compose-dev.yaml -f docker-compose-dev-gpu.yaml up -d
fi

# Optional: Attach to logs
echo "Attaching to logs..."
docker compose -f docker-compose-dev.yaml -f docker-compose-dev-gpu.yaml logs -f