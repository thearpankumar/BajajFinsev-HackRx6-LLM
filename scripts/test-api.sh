#!/bin/bash

set -e

DOMAIN="llmnew.dev"
TOKEN="589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68"

echo "Testing RAG API endpoints..."

echo "1. Testing health check..."
curl -s https://$DOMAIN/hackrx/health | jq .

echo -e "\n2. Testing main endpoint with valid token..."
curl -s -X POST "https://$DOMAIN/hackrx/run" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?", "What are the key points?"]
  }' | jq .

echo -e "\n3. Testing authentication (should fail)..."
curl -s -X POST "https://$DOMAIN/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?"]
  }' | jq .

echo -e "\n4. Testing invalid token (should fail)..."
curl -s -X POST "https://$DOMAIN/hackrx/run" \
  -H "Authorization: Bearer invalid-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?"]
  }' | jq .

echo -e "\nAPI testing complete!"
