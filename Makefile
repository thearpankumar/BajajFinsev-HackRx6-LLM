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

.PHONY: dpush-minimal
dpush-minimal:
	@echo ">>> Compiling minimal requirements..."
	@pip-compile requirements.minimal.in --output-file requirements.minimal.txt --no-build-isolation
	@echo ">>> Building minimal Docker image: $(DOCKER_IMAGE_NAME):minimal..."
	@docker build -f Dockerfile.minimal -t $(DOCKER_IMAGE_NAME):minimal .
	@echo "\n>>> Pushing minimal Docker image to Docker Hub..."
	@docker push $(DOCKER_IMAGE_NAME):minimal
	@echo "\n>>> Minimal image push complete."
	@echo "\n>>> Image size comparison:"
	@docker images | grep hackrx-bajaj

.PHONY: build-minimal
build-minimal:
	@echo ">>> Compiling minimal requirements..."
	@pip-compile requirements.minimal.in --output-file requirements.minimal.txt --no-build-isolation
	@echo ">>> Building minimal Docker image locally..."
	@docker build -f Dockerfile.minimal -t $(DOCKER_IMAGE_NAME):minimal .
	@echo "\n>>> Build complete. Image size:"
	@docker images | grep hackrx-bajaj

.PHONY: compile-minimal
compile-minimal:
	@echo ">>> Compiling minimal requirements..."
	@pip-compile requirements.minimal.in --output-file requirements.minimal.txt --no-build-isolation
	@echo "\n>>> Requirements compiled successfully!"
	@echo ">>> File size comparison:"
	@wc -l requirements.txt requirements.minimal.txt

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make dpush         - Build and push the full Docker image to Docker Hub"
	@echo "  make dpush-minimal - Build and push the minimal Docker image to Docker Hub"
	@echo "  make build-minimal - Build the minimal Docker image locally"
	@echo "  make compile-minimal - Compile minimal requirements from .in file"


