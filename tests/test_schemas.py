import pytest
from pydantic import ValidationError
from src.api.v1.schemas import AnalysisRequest, AnalysisResponse

class TestAnalysisRequest:
    
    def test_valid_request(self):
        request = AnalysisRequest(
            documents=["https://example.com/doc1.pdf", "https://example.com/doc2.docx"],
            questions=["What is the main topic?", "What are the key points?"]
        )
        assert len(request.documents) == 2
        assert len(request.questions) == 2
    
    def test_empty_documents_fails(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(
                documents=[],
                questions=["What is the main topic?"]
            )
    
    def test_empty_questions_fails(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(
                documents=["https://example.com/doc1.pdf"],
                questions=[]
            )
    
    def test_invalid_url_fails(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(
                documents=["not-a-url"],
                questions=["What is the main topic?"]
            )

class TestAnalysisResponse:
    
    def test_valid_response(self):
        response = AnalysisResponse(
            answers=["Answer 1", "Answer 2"]
        )
        assert len(response.answers) == 2
    
    def test_empty_answers(self):
        response = AnalysisResponse(answers=[])
        assert len(response.answers) == 0
