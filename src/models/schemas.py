"""
Pydantic models for request/response schemas
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl, Field


class AnalysisRequest(BaseModel):
    """Request model for document analysis"""

    documents: HttpUrl = Field(..., description="URL to the document to analyze")
    questions: List[str] = Field(
        ..., description="List of questions to answer", min_items=1
    )


class AnalysisResponse(BaseModel):
    """Response model for document analysis"""

    answers: List[str] = Field(..., description="Answers to the questions")
    processing_time: float = Field(..., description="Processing time in seconds")
    document_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )


class StreamResponse(BaseModel):
    """Response model for streaming analysis"""

    initial_answers: List[str] = Field(..., description="Initial quick answers")
    status: str = Field(..., description="Processing status")
    estimated_completion_time: int = Field(
        ..., description="Estimated completion time in seconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Overall system status")
    components: Dict[str, str] = Field(
        ..., description="Status of individual components"
    )
    timestamp: float = Field(..., description="Timestamp of health check")
    error: Optional[str] = Field(None, description="Error message if any")


class PerformanceMetrics(BaseModel):
    """Response model for performance metrics"""

    total_requests: int = Field(..., description="Total number of requests processed")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    average_processing_time: float = Field(
        ..., description="Average processing time in seconds"
    )
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
