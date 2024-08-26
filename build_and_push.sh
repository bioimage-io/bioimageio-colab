#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found!"
  exit 1
fi

# Ensure GHCR_PAT is set
if [ -z "$GHCR_PAT" ]; then
  echo "GHCR_PAT is not set in the .env file!"
  exit 1
fi

# Extract GitHub actor and repository info
GITHUB_ACTOR=$(git config --global user.email)
GITHUB_REPOSITORY=$(git remote get-url origin | sed -E 's|https://github.com/([^/]+/[^.]+).git|\1|')
IMAGE_NAME=ghcr.io/${GITHUB_REPOSITORY}:latest

# Log in to GHCR
echo "$GHCR_PAT" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin

# Generate a dynamic CACHEBUST value (timestamp)
export CACHEBUST=$(date +%Y%m%d%H%M%S)

# Build the Docker image using Docker Compose with the CACHEBUST argument
docker-compose build

# Push the Docker image to GHCR
docker push "$IMAGE_NAME"
