from .ingestion_service import IngestionService
from .rag_workflow import RAGWorkflowService

# Create singleton instances of the services that can be imported elsewhere
ingestion_service = IngestionService()
rag_workflow_service = RAGWorkflowService()
