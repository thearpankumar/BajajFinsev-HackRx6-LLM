from typing import Any
from pydantic import BaseModel, AnyHttpUrl

class URLModel(BaseModel):
    url: AnyHttpUrl

class DocumentBase(BaseModel):
    url: str
    status: str | None = "pending"

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    summary: str | None = None
    status: str | None = None

class DocumentInDB(DocumentBase):
    id: int
    summary: str | None = None
    created_at: Any

    class Config:
        from_attributes = True

class ClauseBase(BaseModel):
    text: str
    embedding_id: str | None = None
    clause_metadata: dict | None = None

class ClauseCreate(ClauseBase):
    document_id: int

class ClauseInDB(ClauseBase):
    id: int
    document_id: int

    class Config:
        from_attributes = True