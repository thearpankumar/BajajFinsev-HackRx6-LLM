from .ingestion_service import IngestionService
from .rag_workflow import RAGWorkflowService
from .text_extraction_service import text_extraction_service as text_extraction_service
from .embedding_service import embedding_service as embedding_service
from .hierarchical_chunking_service import hierarchical_chunking_service as hierarchical_chunking_service
from .streaming_service import streaming_service as streaming_service

# Create singleton instances of the services that can be imported elsewhere
ingestion_service = IngestionService()
rag_workflow_service = RAGWorkflowService()
