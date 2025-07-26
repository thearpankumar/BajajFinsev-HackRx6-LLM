from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1.endpoints.documents import router as documents_router
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Document Processing API",
    description="API for processing various document types with OCR capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router, prefix="/api/v1", tags=["documents"])

@app.get("/")
async def root():
    return {"message": "Document Processing API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
