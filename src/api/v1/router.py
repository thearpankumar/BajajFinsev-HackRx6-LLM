from fastapi import APIRouter
from src.api.v1.endpoints import analysis

api_router = APIRouter()
api_router.include_router(analysis.router, tags=["analysis"])
