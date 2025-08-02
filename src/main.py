import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
import uvicorn

from src.api.v1.router import api_router
from src.core.config import settings
from src.worker import start_worker, stop_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Business Document Analysis API with AI-Powered RAG",
    version="5.0.0",
    description="An enterprise-grade API specialized for insurance, legal, HR, and compliance document analysis. Uses OpenAI GPT-4o-mini and advanced embedding-based retrieval for accurate business information extraction.",
    openapi_tags=[
        {"name": "analysis", "description": "Endpoints for business document analysis in insurance, legal, HR, and compliance domains."},
        {"name": "monitoring", "description": "Endpoints for health checks and performance monitoring."},
        {"name": "info", "description": "General API information and capabilities."}
    ]
)

@app.on_event("startup")
async def startup_event():
    """Actions to run on application startup."""
    start_worker()
    
    # Start load balancer
    try:
        from src.services import load_balancing_service
        await load_balancing_service.start_load_balancer()
        logger.info("Load balancer started")
    except Exception as e:
        logger.error(f"Failed to start load balancer: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Actions to run on application shutdown."""
    stop_worker()
    
    # Stop load balancer
    try:
        from src.services import load_balancing_service
        await load_balancing_service.stop_load_balancer()
        logger.info("Load balancer stopped")
    except Exception as e:
        logger.error(f"Failed to stop load balancer: {e}")

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
    allowed_hosts=["llmnow.dev", "www.llmnow.dev", "localhost", "127.0.0.1"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://llmnow.dev", "https://www.llmnow.dev"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["info"])
async def root():
    return {
        "name": "Business Document Analysis API with AI-Powered RAG",
        "version": "5.0.0",
        "status": "running",
        "specialization": "Insurance, Legal, HR, and Compliance domains",
        "ai_models": "OpenAI GPT-4o-mini + Google Gemini Flash",
        "docs": "/docs"
    }

@app.get("/health", tags=["monitoring"])
async def global_health():
    return {
        "status": "healthy",
        "message": "Business Document Analysis API is running",
        "version": "5.0.0",
        "domains": ["Insurance", "Legal", "HR", "Compliance"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )