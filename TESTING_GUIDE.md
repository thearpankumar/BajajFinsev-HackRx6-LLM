# RAG API Testing Guide

This guide provides comprehensive instructions for testing the RAG Document Analysis API locally and in production.

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- curl or similar HTTP client
- jq (for JSON formatting, optional)
- Access to a server with domain `llmnew.dev`

## Testing Scenarios

### 1. Local Development Testing

#### Setup Local Environment
\`\`\`bash
git clone <repository>
cd rag-api-task1

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
\`\`\`

#### Run Local Server
\`\`\`bash
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
\`\`\`

#### Test Local Endpoints
\`\`\`bash
curl http://localhost:8000/hackrx/health

curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?", "What are the key points?"]
  }'

curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?"]
  }'
\`\`\`

### 2. Unit Testing

#### Run Test Suite
\`\`\`bash
pip install pytest pytest-asyncio httpx

pytest tests/ -v

pytest tests/test_schemas.py -v
pytest tests/test_security.py -v
pytest tests/test_endpoints.py -v

pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
\`\`\`

#### Expected Test Results
\`\`\`
tests/test_schemas.py::TestAnalysisRequest::test_valid_request PASSED
tests/test_schemas.py::TestAnalysisRequest::test_empty_documents_fails PASSED
tests/test_schemas.py::TestAnalysisRequest::test_empty_questions_fails PASSED
tests/test_schemas.py::TestAnalysisRequest::test_invalid_url_fails PASSED
tests/test_schemas.py::TestAnalysisResponse::test_valid_response PASSED
tests/test_schemas.py::TestAnalysisResponse::test_empty_answers PASSED
tests/test_security.py::TestSecurity::test_valid_bearer_token PASSED
tests/test_security.py::TestSecurity::test_invalid_bearer_token PASSED
tests/test_security.py::TestSecurity::test_validate_bearer_token_success PASSED
tests/test_security.py::TestSecurity::test_validate_bearer_token_failure PASSED
tests/test_endpoints.py::TestAnalysisEndpoint::test_run_analysis_success PASSED
tests/test_endpoints.py::TestAnalysisEndpoint::test_run_analysis_unauthorized PASSED
tests/test_endpoints.py::TestAnalysisEndpoint::test_run_analysis_invalid_token PASSED
tests/test_endpoints.py::TestAnalysisEndpoint::test_health_check PASSED
tests/test_endpoints.py::TestRootEndpoints::test_root_endpoint PASSED
tests/test_endpoints.py::TestRootEndpoints::test_health_endpoint PASSED
\`\`\`

### 3. Production Deployment Testing

#### Deploy to Production
\`\`\`bash
chmod +x scripts/setup-ssl.sh
chmod +x scripts/deploy.sh
chmod +x scripts/test-api.sh

sudo ./scripts/setup-ssl.sh

./scripts/deploy.sh
\`\`\`

#### Verify Production Deployment
\`\`\`bash
docker-compose ps

docker-compose logs rag-api
docker-compose logs nginx

./scripts/test-api.sh
\`\`\`

### 4. HTTPS & SSL Testing

#### Verify SSL Certificate
\`\`\`bash
openssl s_client -connect llmnew.dev:443 -servername llmnew.dev

curl -vI https://llmnew.dev 2>&1 | grep -i expire
\`\`\`

#### Test HTTPS Redirect
\`\`\`bash
curl -I http://llmnew.dev
\`\`\`

### 5. API Endpoint Testing

#### Test Main Endpoint
\`\`\`bash
curl -X POST "https://llmnew.dev/hackrx/run" \
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "https://example.com/document1.pdf",
      "https://example.com/document2.docx"
    ],
    "questions": [
      "What are the key terms and conditions?",
      "What is the cancellation policy?",
      "What are the main topics covered?"
    ]
  }'
\`\`\`

#### Test Authentication
\`\`\`bash
curl -X POST "https://llmnew.dev/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?"]
  }'

curl -X POST "https://llmnew.dev/hackrx/run" \
  -H "Authorization: Bearer invalid-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?"]
  }'
\`\`\`

#### Test Input Validation
\`\`\`bash
curl -X POST "https://llmnew.dev/hackrx/run" \
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [],
    "questions": ["What is the main topic?"]
  }'

curl -X POST "https://llmnew.dev/hackrx/run" \
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["not-a-valid-url"],
    "questions": ["What is the main topic?"]
  }'

curl -X POST "https://llmnew.dev/hackrx/run" \
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": []
  }'
\`\`\`

### 6. Performance Testing

#### Response Time Testing
\`\`\`bash
time curl -X POST "https://llmnew.dev/hackrx/run" \
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?"]
  }'
\`\`\`

#### Load Testing (Optional)
\`\`\`bash
sudo apt install apache2-utils

ab -n 100 -c 10 -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
   -H "Content-Type: application/json" \
   -p test_payload.json \
   https://llmnew.dev/hackrx/run
\`\`\`

### 7. Documentation Testing

#### Test API Documentation
\`\`\`bash
curl https://llmnew.dev/docs

curl https://llmnew.dev/redoc

curl https://llmnew.dev/openapi.json
\`\`\`

### 8. Health Check Testing

#### Test Health Endpoints
\`\`\`bash
curl https://llmnew.dev/hackrx/health

curl https://llmnew.dev/health
\`\`\`

## Troubleshooting

### Common Issues

#### 1. SSL Certificate Issues
\`\`\`bash
sudo certbot certificates

sudo certbot renew --dry-run

sudo nginx -t
\`\`\`

#### 2. Docker Issues
\`\`\`bash
docker-compose ps

docker-compose logs -f rag-api
docker-compose logs -f nginx

docker-compose restart
\`\`\`

#### 3. Network Issues
\`\`\`bash
netstat -tlnp | grep :80
netstat -tlnp | grep :443
netstat -tlnp | grep :8000

docker-compose exec nginx curl http://rag-api:8000/health
\`\`\`

### Expected Status Codes

| Scenario | Expected Status | Description |
|----------|----------------|-------------|
| Valid request with auth | 200 | Success |
| No authorization header | 403 | Forbidden |
| Invalid bearer token | 401 | Unauthorized |
| Invalid request body | 422 | Validation Error |
| Health check | 200 | Healthy |
| HTTP to HTTPS redirect | 301 | Moved Permanently |

## Pre-Submission Checklist

Before submitting, verify all these tests pass:

- [ ] Local development server starts successfully
- [ ] All unit tests pass (16/16)
- [ ] HTTPS certificate is valid and auto-renewing
- [ ] Main endpoint responds with 200 for valid requests
- [ ] Authentication properly rejects invalid tokens
- [ ] Input validation works correctly
- [ ] Response time is under 30 seconds
- [ ] Health checks return healthy status
- [ ] API documentation is accessible
- [ ] SSL Labs rating is A or higher (optional)

## Support

If you encounter issues:

1. Check the logs: `docker-compose logs`
2. Verify SSL certificates: `sudo certbot certificates`
3. Test individual components: `curl` commands above
4. Review Nginx configuration: `sudo nginx -t`
5. Check firewall settings: `sudo ufw status`

Your API should be fully functional at `https://llmnew.dev/hackrx/run` with Bearer token authentication.
