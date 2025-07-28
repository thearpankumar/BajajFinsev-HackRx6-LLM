#!/bin/bash

set -e

echo "Deploying RAG API to production..."

docker-compose down
docker-compose build --no-cache
docker-compose up -d

echo "Waiting for services to be healthy..."
sleep 30

curl -f http://localhost:8000/health || {
    echo "Health check failed!"
    docker-compose logs
    exit 1
}

echo "Deployment successful!"
echo "API is running at: https://llmnew.dev"
echo "Health check: https://llmnew.dev/hackrx/health"
echo "Documentation: https://llmnew.dev/docs"
