#!/bin/bash

# File: scripts/dev-gpu.sh

# Description: Script to run Docker Compose for development with GPU support.

echo "Creating logs dir..."
mkdir -p ./services/speech/logs

echo "Grant permission for log dir and common dir ..."
sudo chmod -R 777 ./services/speech/logs
sudo chmod -R 777 ./services/common_dir

# Stop any running containers
echo "Stopping any existing containers..."
docker compose -f docker-compose.yaml down

# Check if the first argument is "rebuild"
if [ "$1" = "rebuild" ]; then
    REBUILD="true"
else
    REBUILD="false"
fi

# Build and start the containers, with or without --build
if [ "$REBUILD" = "true" ]; then
    echo "Rebuilding and starting the containers..."
    docker compose -f docker-compose.yaml up --build -d
else
    echo "Starting the containers without rebuilding..."
    docker compose -f docker-compose.yaml up -d
fi

# Optional: Attach to logs
echo "Attaching to logs..."
docker compose -f docker-compose.yaml logs -f