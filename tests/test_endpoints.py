import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.core.security import VALID_API_KEY

client = TestClient(app)

class TestAnalysisEndpoint:
    
    def test_run_analysis_success(self):
        response = client.post(
            "/hackrx/run",
            json={
                "documents": ["https://example.com/doc1.pdf"],
                "questions": ["What is the main topic?"]
            },
            headers={"Authorization": f"Bearer {VALID_API_KEY}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answers" in data
        assert len(data["answers"]) == 1
    
    def test_run_analysis_unauthorized(self):
        response = client.post(
            "/hackrx/run",
            json={
                "documents": ["https://example.com/doc1.pdf"],
                "questions": ["What is the main topic?"]
            }
        )
        assert response.status_code == 403
    
    def test_run_analysis_invalid_token(self):
        response = client.post(
            "/hackrx/run",
            json={
                "documents": ["https://example.com/doc1.pdf"],
                "questions": ["What is the main topic?"]
            },
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401
    
    def test_health_check(self):
        response = client.get("/hackrx/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

class TestRootEndpoints:
    
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
