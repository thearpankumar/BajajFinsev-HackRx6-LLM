from sqlalchemy.orm import Session
from src.db.models import Document, Clause
from src.schemas.document import DocumentCreate, DocumentUpdate, ClauseCreate

def create_document(db: Session, *, document_in: DocumentCreate) -> Document:
    db_document = Document(**document_in.model_dump())
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def update_document(db: Session, *, db_obj: Document, obj_in: DocumentUpdate) -> Document:
    if isinstance(obj_in, dict):
        update_data = obj_in
    else:
        update_data = obj_in.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(db_obj, field, value)
        
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

def get_document(db: Session, document_id: int) -> Document | None:
    return db.query(Document).filter(Document.id == document_id).first()

def create_clauses(db: Session, *, clauses_in: list[ClauseCreate]) -> list[Clause]:
    db_clauses = [Clause(**clause.model_dump()) for clause in clauses_in]
    db.add_all(db_clauses)
    db.commit()
    for clause in db_clauses:
        db.refresh(clause)
    return db_clauses

def update_clause_embedding_id(db: Session, *, clause: Clause, embedding_id: str) -> Clause:
    clause.embedding_id = embedding_id
    db.add(clause)
    db.commit()
    db.refresh(clause)
    return clause
