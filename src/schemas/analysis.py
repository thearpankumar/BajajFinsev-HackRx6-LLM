from typing import List
from pydantic import BaseModel, HttpUrl, Field, ConfigDict

class AnalysisRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": "https://example.com/document1.pdf",
                "questions": [
                    "What are the key terms and conditions?",
                    "What is the cancellation policy?"
                ]
            }
        }
    )
    
    documents: HttpUrl = Field(
        ...,
        description="HTTP URL pointing to a document to analyze"
    )
    
    questions: List[str] = Field(
        ...,
        description="List of questions to ask about the documents",
        min_length=1,
        max_length=20
    )

class AnalysisResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answers": [
                    "The key terms include a 30-day notice period and automatic renewal clause.",
                    "Cancellation is allowed with 14 days written notice without penalty."
                ]
            }
        }
    )
    
    answers: List[str] = Field(
        ...,
        description="List of answers corresponding to the input questions"
    )
