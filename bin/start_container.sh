#!/usr/bin/env bash

# Define variables
CONTAINER_NAME="sandbox"
IMAGE_NAME="sandbox:v1"
DOCKERFILE="Dockerfile"

# Retrieve Git user info from the host
GIT_USER_NAME=$(git config --get user.name)
GIT_USER_EMAIL=$(git config --get user.email)

# Stop and remove the container if it already exists
if docker ps --all --format '{{.Names}}' | grep --extended-regexp --quiet "^${CONTAINER_NAME}\$"; then
    echo "Stopping and removing existing container..."
    docker stop ${CONTAINER_NAME} || echo "Failed to stop container."
    docker rm ${CONTAINER_NAME} || echo "Failed to remove container."
fi

# Build the Docker image
echo "Building Docker image..."
docker build \
    --tag ${IMAGE_NAME} \
    --file ${DOCKERFILE} \
    --build-arg GIT_USER_NAME="${GIT_USER_NAME}" \
    --build-arg GIT_USER_EMAIL="${GIT_USER_EMAIL}" \
    . \
    || { echo "Failed to build Docker image."; exit 1; }

# Run the Docker container and mount the current host directory on the container's app directory
# This allows changes made in either directory to be mirrored in the other and, thus, to keep everything under version control
# Mounting overwrites the contents of the container's directory with the contents of the host's directory, but the two directories are identical at that point
echo "Starting Docker container..."
docker run \
    --tty \
    --detach \
    --name ${CONTAINER_NAME} \
    --network=host \
    --volume "${PWD}":/app \
    ${IMAGE_NAME} \
    || { echo "Failed to start Docker container."; exit 1; }
