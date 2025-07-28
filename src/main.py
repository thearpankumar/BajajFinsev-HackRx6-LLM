from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
import uvicorn
import logging

from src.api.v1.endpoints.analysis import router as analysis_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Analysis API",
    version="1.0.0",
    description="",
    openapi_tags=[
        {"name": "analysis", "description": ""},
        {"name": "monitoring", "description": ""},
        {"name": "info", "description": ""}
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
            "description": ""
        }
    }

    openapi_schema["security"] = [{"BearerAuth": []}]

    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if isinstance(operation, dict) and "operationId" in operation:
                if "/hackrx/run" in operation.get("operationId", "") or "run_analysis" in operation.get("operationId", ""):
                    operation["security"] = [{"BearerAuth": []}]
                elif "/health" in operation.get("operationId", "") or "/info" in operation.get("operationId", ""):
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

app.include_router(analysis_router)

@app.get("/", tags=["info"])
async def api_info():
    return {
        "name": "RAG Document Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/hackrx/health",
            "main_analysis": "/hackrx/run"
        },
        "authentication": "Bearer token required",
        "valid_token": "12345678901"
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
