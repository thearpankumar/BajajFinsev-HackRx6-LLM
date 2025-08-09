# Makefile for Docker Operations

# ==============================================================================
# Variables
# ==============================================================================

# The name of your Docker Hub repository and image
DOCKER_IMAGE_NAME := arpankumar1119/hackrx-bajaj

# ==============================================================================
# Targets
# ==============================================================================

.PHONY: dpush
dpush:
	@echo ">>> Building Docker image: $(DOCKER_IMAGE_NAME):latest..."
	@docker build --no-cache -t $(DOCKER_IMAGE_NAME):latest .
	@echo "\n>>> Pushing Docker image to Docker Hub..."
	@docker push $(DOCKER_IMAGE_NAME):latest
	@echo "\n>>> Image push complete."

.PHONY: build
build:
	@echo ">>> Building Docker image locally..."
	@docker build --no-cache -t $(DOCKER_IMAGE_NAME):latest .
	@echo "\n>>> Build complete. Image size:"
	@docker images | grep hackrx-bajaj

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make dpush - Build and push the Docker image to Docker Hub"
	@echo "  make build - Build the Docker image locally"
