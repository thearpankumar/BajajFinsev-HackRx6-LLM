import logging
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
import uvicorn

from src.api.v1.router import api_router
from src.core.config import settings

# Suppress the specific FutureWarning from torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Analysis and Processing API",
    version="1.0.0",
    description="API for processing documents, running analysis, and answering questions using a RAG workflow.",
    openapi_tags=[
        {"name": "analysis", "description": "Endpoints for running analysis on documents."},
        {"name": "documents", "description": "Endpoints for uploading and processing documents."},
        {"name": "monitoring", "description": "Endpoints for health checks and monitoring."},
        {"name": "info", "description": "General API information."}
    ]
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": f"Enter your bearer token (e.g., {settings.API_KEY})"
        }
    }

    openapi_schema["security"] = [{"BearerAuth": []}]

    # Make health and info endpoints public in docs
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                tags = operation.get("tags", [])
                if "monitoring" in tags or "info" in tags:
                    operation["security"] = []

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["llmnew.dev", "www.llmnew.dev", "localhost", "127.0.0.1"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://llmnew.dev", "https://www.llmnew.dev"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["info"])
async def root():
    return {
        "name": "Document Analysis and Processing API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", tags=["monitoring"])
async def global_health():
    return {
        "status": "healthy",
        "message": "API is running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
