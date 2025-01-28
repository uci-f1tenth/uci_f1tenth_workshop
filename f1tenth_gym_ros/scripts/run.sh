#!/bin/bash

# run the docker container
# open Docker desktop if the user has mac
# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed on this system."
  echo "Please install Docker from https://www.docker.com/products/docker-desktop."
  exit 1
fi

if [[ $(uname) == "Darwin" ]]; then
  echo "Starting Docker for macOS..."
  open /Applications/Docker.app
  
  echo "Waiting for Docker to start..."
  while ! docker info >/dev/null 2>&1; do
    sleep 1
  done
  echo "Docker is now running!"
fi

# run the contained
docker compose up -d

# execute the container
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
