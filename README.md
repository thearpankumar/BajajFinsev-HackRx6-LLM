# RAG Document Analysis API - Task 1

A secure FastAPI application for document analysis using Retrieval-Augmented Generation (RAG).

## Quick Start

1. **Download & Extract**: Extract the zip file to your server
2. **Setup SSL**: `sudo ./scripts/setup-ssl.sh`
3. **Deploy**: `./scripts/deploy.sh`
4. **Test**: `./scripts/test-api.sh`

## Pre-Submission Checklist

- ✅ `/hackrx/run` endpoint is live at `https://llmnew.dev/hackrx/run`
- ✅ HTTPS enabled with Let's Encrypt SSL certificate
- ✅ Bearer token authentication: `Authorization: Bearer <token>`
- ✅ Handles POST requests with JSON response
- ✅ Response time < 30 seconds
- ✅ Tested with sample data

## Authentication

**Bearer Token**: `589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68`

## Testing

See `TESTING_GUIDE.md` for comprehensive testing instructions.

### Quick Test
\`\`\`bash
curl -X POST "https://llmnew.dev/hackrx/run" \
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?"]
  }'
\`\`\`

## API Documentation

- **Swagger UI**: https://llmnew.dev/docs
- **ReDoc**: https://llmnew.dev/redoc
- **Health Check**: https://llmnew.dev/hackrx/health

## Project Structure

\`\`\`
├── src/
│   ├── api/v1/
│   │   ├── endpoints/analysis.py
│   │   └── schemas.py
│   ├── core/security.py
│   └── main.py
├── nginx/
├── scripts/
├── tests/
├── TESTING_GUIDE.md
└── README.md
\`\`\`

## Security Features

- HTTPS enforced with SSL/TLS certificates
- Bearer token authentication
- Rate limiting and security headers
- Input validation with Pydantic
- CORS protection

## Deployment

The project includes automated deployment scripts for production use with Docker and Nginx reverse proxy.

Ready for submission!
