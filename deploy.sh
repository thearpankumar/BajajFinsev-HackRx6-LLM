#!/bin/bash

# Production Deployment Script for Embedding-based RAG System
# Usage: ./deploy.sh [build|start|stop|restart|logs|status]

set -e

PROJECT_NAME="hackrx-bajaj"
IMAGE_NAME="arpankumar1119/hackrx-bajaj:latest"

function show_help() {
    echo "Deployment Script for $PROJECT_NAME"
    echo ""
    echo "Usage: ./deploy.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     - Build and push Docker image"
    echo "  start     - Start the application services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  logs      - Show application logs"
    echo "  status    - Show service status"
    echo "  health    - Check application health"
    echo "  clean     - Clean up unused containers and images"
    echo ""
}

function check_env() {
    if [ ! -f .env ]; then
        echo "❌ Error: .env file not found!"
        echo "💡 Copy .env.production to .env and update with your keys:"
        echo "   cp .env.production .env"
        echo "   nano .env"
        exit 1
    fi
    
    # Check required environment variables
    source .env
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
        echo "❌ Error: OPENAI_API_KEY not set in .env file!"
        exit 1
    fi
    
    if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
        echo "❌ Error: GOOGLE_API_KEY not set in .env file!"
        exit 1
    fi
    
    echo "✅ Environment variables validated"
}

function build_image() {
    echo "🔨 Building Docker image..."
    
    # Build the image
    docker build -t $IMAGE_NAME .
    
    echo "✅ Docker image built successfully"
    
    # Push to registry (optional)
    read -p "🚀 Push to Docker registry? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📤 Pushing to registry..."
        docker push $IMAGE_NAME
        echo "✅ Image pushed to registry"
    fi
}

function start_services() {
    echo "🚀 Starting services..."
    check_env
    docker-compose up -d
    echo "✅ Services started successfully"
    echo ""
    sleep 5
    show_status
}

function stop_services() {
    echo "🛑 Stopping services..."
    docker-compose down
    echo "✅ Services stopped"
}

function restart_services() {
    echo "🔄 Restarting services..."
    stop_services
    sleep 2
    start_services
}

function show_logs() {
    echo "📋 Showing application logs..."
    docker-compose logs -f fastapi-app
}

function show_status() {
    echo "📊 Service Status:"
    docker-compose ps
    echo ""
    
    # Check application health
    echo "🏥 Health Check:"
    if curl -s -f http://localhost/health > /dev/null 2>&1; then
        echo "✅ Application is healthy and responding"
        curl -s http://localhost/health | jq '.' 2>/dev/null || curl -s http://localhost/health
    else
        echo "❌ Application health check failed"
        echo "🔍 Try: ./deploy.sh logs"
    fi
}

function health_check() {
    echo "🏥 Detailed Health Check..."
    
    # Check containers
    echo "Container Status:"
    docker-compose ps
    echo ""
    
    # Check application endpoints
    echo "API Endpoints:"
    if curl -s -f http://localhost/health > /dev/null 2>&1; then
        echo "✅ /health - OK"
    else
        echo "❌ /health - Failed"
    fi
    
    if curl -s -f http://localhost/api/v1/hackrx/health > /dev/null 2>&1; then
        echo "✅ /api/v1/hackrx/health - OK"
    else
        echo "❌ /api/v1/hackrx/health - Failed"
    fi
    
    # Check disk space for embedding cache
    echo ""
    echo "Disk Usage:"
    docker system df
}

function clean_up() {
    echo "🧹 Cleaning up unused containers and images..."
    docker-compose down --remove-orphans
    docker system prune -f
    docker volume prune -f
    echo "✅ Cleanup completed"
}

# Main command handling
case "${1:-help}" in
    "build")
        build_image
        ;;
    "start")
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "health")
        health_check
        ;;
    "clean")
        clean_up
        ;;
    "help"|*)
        show_help
        ;;
esac