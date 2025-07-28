from fastapi.testclient import TestClient
from src.main import app
import pytest
from unittest.mock import patch

client = TestClient(app)

# Test data
TEST_TOKEN = "12345678901"
AUTH_HEADER = {"Authorization": f"Bearer {TEST_TOKEN}"}

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Document Analysis and Processing API"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_file_without_auth():
    """Test file upload endpoint without authentication"""
    response = client.post("/api/v1/upload/file")
    assert response.status_code == 401

def test_upload_url_without_auth():
    """Test URL upload endpoint without authentication"""
    response = client.post("/api/v1/upload/url", json={"url": "http://example.com/test.pdf"})
    assert response.status_code == 401

@pytest.fixture
def mock_document_processing():
    """Mock document processing functions"""
    with patch("src.utils.document_parsers.parse_pdf") as mock_pdf, \
         patch("src.utils.document_parsers.parse_docx") as mock_docx, \
         patch("src.utils.document_parsers.parse_email") as mock_email:
        
        # Set return values for mocks
        mock_pdf.return_value = "Mocked PDF content"
        mock_docx.return_value = "Mocked DOCX content"
        mock_email.return_value = "Mocked email content"
        
        yield {
            "pdf": mock_pdf,
            "docx": mock_docx,
            "email": mock_email
        }

def test_upload_pdf_file(mock_document_processing):
    """Test PDF file upload with authentication"""
    # Create a test PDF file content
    test_content = b"Test PDF content"
    
    response = client.post(
        "/api/v1/upload/file",
        headers=AUTH_HEADER,
        files={"file": ("test.pdf", test_content, "application/pdf")}
    )
    
    assert response.status_code == 202
    assert response.json()["status"] == "completed"
    assert response.json()["filename"] == "test.pdf"
    assert "extracted_text" in response.json()

def test_upload_docx_file(mock_document_processing):
    """Test DOCX file upload with authentication"""
    test_content = b"Test DOCX content"
    
    response = client.post(
        "/api/v1/upload/file",
        headers=AUTH_HEADER,
        files={"file": ("test.docx", test_content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    
    assert response.status_code == 202
    assert response.json()["status"] == "completed"
    assert response.json()["filename"] == "test.docx"
    assert "extracted_text" in response.json()

def test_upload_unsupported_file():
    """Test upload of unsupported file type"""
    test_content = b"Test content"
    
    response = client.post(
        "/api/v1/upload/file",
        headers=AUTH_HEADER,
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

@patch("aiohttp.ClientSession.get")
def test_process_url(mock_get, mock_document_processing):
    """Test URL processing endpoint"""
    # Mock the URL response
    mock_get.return_value.__aenter__.return_value.status = 200
    mock_get.return_value.__aenter__.return_value.read.return_value = b"Test PDF content"
    mock_get.return_value.__aenter__.return_value.headers = {}
    
    response = client.post(
        "/api/v1/upload/url",
        headers=AUTH_HEADER,
        json={"url": "http://example.com/test.pdf"}
    )
    
    assert response.status_code == 202
    assert response.json()["status"] == "completed"
    assert "extracted_text" in response.json()

def test_process_invalid_url():
    """Test processing with invalid URL"""
    response = client.post(
        "/api/v1/upload/url",
        headers=AUTH_HEADER,
        json={"url": "not_a_valid_url"}
    )
    
    assert response.status_code == 422 
    assert "valid url" in response.json()["detail"][0]["msg"].lower()