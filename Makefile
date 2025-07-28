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
	@docker build -t $(DOCKER_IMAGE_NAME):latest .
	@echo "\n>>> Pushing Docker image to Docker Hub..."
	@docker push $(DOCKER_IMAGE_NAME):latest
	@echo "\n>>> Image push complete."

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make dpush    - Build the Docker image and push it to Docker Hub."


