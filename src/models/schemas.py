"""
Pydantic models for request/response schemas
"""

from typing import Any, Union

from pydantic import BaseModel, Field, HttpUrl


class AnalysisRequest(BaseModel):
    """Request model for document analysis"""

    documents: HttpUrl = Field(..., description="URL to the document to analyze")
    questions: list[str] = Field(
        ..., description="List of questions to answer", min_items=1
    )



class StreamResponse(BaseModel):
    """Response model for streaming analysis"""

    initial_answers: list[str] = Field(..., description="Initial quick answers")
    status: str = Field(..., description="Processing status")
    estimated_completion_time: int = Field(
        ..., description="Estimated completion time in seconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Overall system status")
    components: dict[str, str] = Field(
        ..., description="Status of individual components"
    )
    timestamp: float = Field(..., description="Timestamp of health check")
    error: Union[str, None] = Field(None, description="Error message if any")


class PerformanceMetrics(BaseModel):
    """Response model for performance metrics"""

    total_requests: int = Field(..., description="Total number of requests processed")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    average_processing_time: float = Field(
        ..., description="Average processing time in seconds"
    )
    average_document_size: float = Field(..., description="Average document size")
    total_documents_processed: int = Field(..., description="Total documents processed")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    custom_metrics: dict[str, Any] | None = Field(None, description="Custom metrics")
