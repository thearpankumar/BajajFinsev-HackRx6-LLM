# 🐳 Docker Compose Usage Guide

## 🚀 Running BajajFinsev RAG System with Docker Compose

This guide shows you how to run the complete BajajFinsev RAG System with HackRx Challenge Integration using Docker Compose.

## 📋 What Gets Started

When you run `docker-compose up`, you get:

- **🤖 Challenge Solver Server** (Port 8004) - Handles HackRx parallel world challenges
- **📚 Main RAG Server** (Port 8000) - Advanced document processing with intelligent routing
- **🗃️ Qdrant Vector Database** (Port 6333) - Vector storage for embeddings
- **🔴 Redis Cache** (Port 6379) - Performance caching
- **🌐 Nginx Proxy** (Ports 80/443) - SSL termination and load balancing
- **🔒 Certbot** (On-demand) - SSL certificate management

## 🛠️ Prerequisites

1. **Docker & Docker Compose installed**
2. **Environment file configured** (`.env`)
3. **Challenge files available** in `downloads/` directory

## ⚙️ Environment Setup

Create a `.env` file with your API keys:

```bash
# API Keys
API_KEY=your_secure_api_key_here
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key

# Optional: Override challenge solver URL (for debugging)
# CHALLENGE_SOLVER_URL=http://challenge-solver:8004
```

## 🚀 Starting the System

### Option 1: Full Production Stack
```bash
# Start all services including nginx and SSL
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Option 2: Development Mode (without SSL)
```bash
# Start just the core services
docker-compose up -d qdrant redis challenge-solver fastapi-app

# Test directly on port 8000
curl http://localhost:8000/api/v1/hackrx/health
```

### Option 3: Challenge Solver Only
```bash
# Start just the challenge solver for testing
docker-compose up -d challenge-solver

# Test the challenge solver
curl -X POST http://localhost:8004/call/solve_complete_challenge \
     -H "Content-Type: application/json" -d "{}"
```

## 🧪 Testing the Integration

### Test 1: Health Checks
```bash
# Check main RAG system
curl http://localhost:8000/api/v1/hackrx/health

# Check challenge solver (direct)
curl http://localhost:8004/

# Or through nginx proxy
curl https://your-domain.com/challenge/
```

### Test 2: HackRx Challenge Detection
```bash
curl -X POST http://localhost:8000/api/v1/hackrx/run \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "downloads/FinalRound4SubmissionPDF.pdf",
       "questions": [
         "What flight number will take Sachin back to the real world from the parallel world?"
       ]
     }'
```

**Expected Response:**
```json
{
  "answers": [
    "Sachin can return to the real world using flight number: abc123 (via Eiffel Tower)!"
  ]
}
```

### Test 3: Regular Document Processing
```bash
curl -X POST http://localhost:8000/api/v1/hackrx/run \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://example.com/document.pdf",
       "questions": ["What is the main topic of this document?"]
     }'
```

## 🔍 Service Architecture

### Internal Network Communication
- `fastapi-app` → `challenge-solver:8004` (automatic challenge detection)
- `fastapi-app` → `qdrant:6333` (vector storage)
- `fastapi-app` → `redis:6379` (caching)
- `nginx` → `fastapi-app:8000` (main API)
- `nginx` → `challenge-solver:8004` (direct challenge API via `/challenge/`)

### External Access
- **Production**: `https://your-domain.com/api/v1/hackrx/run`
- **Development**: `http://localhost:8000/api/v1/hackrx/run`
- **Challenge Solver**: `https://your-domain.com/challenge/call/solve_complete_challenge`

## 📁 Volume Mounts

The system mounts several volumes:

```yaml
volumes:
  - ./downloads:/app/downloads:ro        # Challenge files (read-only)
  - embedding_cache:/tmp/embedding_cache # AI model cache
  - vector_db:/app/vector_db            # FAISS database
  - document_cache:/app/document_cache   # Document processing cache
```

### Adding Challenge Files
```bash
# Place your challenge files in the downloads directory
mkdir -p downloads
cp FinalRound4SubmissionPDF.pdf downloads/

# Restart services to pick up new files
docker-compose restart challenge-solver fastapi-app
```

## 🐛 Troubleshooting

### Problem: Challenge not detected
```bash
# Check if both services are running
docker-compose ps

# Check challenge solver logs
docker-compose logs challenge-solver

# Check main app logs
docker-compose logs fastapi-app

# Test challenge solver directly
curl http://localhost:8004/call/get_challenge_status -X POST -d "{}"
```

### Problem: Services won't start
```bash
# Check for port conflicts
docker-compose down
docker-compose up -d

# Check logs for errors
docker-compose logs --tail=50
```

### Problem: Files not found in downloads
```bash
# Check volume mount
docker-compose exec fastapi-app ls -la /app/downloads/

# Make sure files exist on host
ls -la downloads/
```

### Problem: SSL/Nginx issues
```bash
# Start without nginx first
docker-compose up -d qdrant redis challenge-solver fastapi-app

# Test direct access
curl http://localhost:8000/api/v1/hackrx/health

# Then add nginx
docker-compose up -d nginx
```

## 📊 Monitoring & Logs

### View Real-time Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fastapi-app
docker-compose logs -f challenge-solver

# Last 100 lines
docker-compose logs --tail=100
```

### Resource Usage
```bash
# Check resource usage
docker stats

# Check specific container
docker stats fastapi-app-container challenge-solver-container
```

## 🔄 Updates & Maintenance

### Updating the System
```bash
# Pull latest images
docker-compose pull

# Restart with new images
docker-compose up -d --force-recreate

# Clean up old images
docker image prune -f
```

### Backup Important Data
```bash
# Backup vector database and caches
docker-compose down
cp -r vector_db/ backup_vector_db_$(date +%Y%m%d)
cp -r document_cache/ backup_document_cache_$(date +%Y%m%d)
docker-compose up -d
```

## ✅ Success Indicators

When everything is working correctly, you should see:

1. **All services healthy**: `docker-compose ps` shows all services as "Up"
2. **Challenge detection working**: Queries about flight numbers get automatic answers
3. **Regular RAG working**: Non-challenge queries get processed normally  
4. **Logs show**: "🤖 Running intelligent challenge detection..." and "✅ Challenge automatically solved"

## 🎉 Production Deployment

For production deployment:

1. **Update `.env`** with production API keys
2. **Configure SSL certificates** with Let's Encrypt
3. **Set up monitoring** and log aggregation
4. **Configure backups** for vector databases
5. **Set resource limits** in docker-compose.yml

The system is now fully integrated and ready for production use with automatic challenge detection and solving!