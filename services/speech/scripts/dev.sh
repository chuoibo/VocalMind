#!/bin/bash

echo "Creating logs dir..."
mkdir -p ./logs

echo "Grant permission for log dir..."
sudo chmod -R 777 ./logs

# Check if the first argument is "rebuild"
if [ "$1" = "rebuild" ]; then
    REBUILD="true"
else
    REBUILD="false"
fi

# Stop any running containers
echo "Stopping any existing containers..."
docker compose -f docker-compose-dev.yaml down

# Build and start the containers, with or without --build
if [ "$REBUILD" = "true" ]; then
    echo "Rebuilding and starting the containers..."
    docker compose -f docker-compose-dev.yaml up --build -d
else
    echo "Starting the containers without rebuilding..."
    docker compose -f docker-compose-dev.yaml up -d
fi

# Optional: Attach to logs
echo "Attaching to logs..."
docker compose -f docker-compose-dev.yaml logs -f