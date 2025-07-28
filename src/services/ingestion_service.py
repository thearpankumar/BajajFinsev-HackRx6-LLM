from typing import List, Dict, Any, Optional
import aiohttp
import spacy
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from src.core.config import settings
from src.db.session import get_db
from src.crud import crud_document
from src.schemas.document import DocumentCreate, DocumentUpdate, ClauseCreate
from src.utils.document_parsers import get_parser
from src.services.llm_clients import groq_client, GROQ_MODEL_NAME


class IngestionService:
    def __init__(self):
        # Validate required environment variables
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
            
        self.embedding_model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        
        # Initialize Pinecone with error handling
        try:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

    async def download_document(self, url: str) -> bytes:
        """Download document content from URL asynchronously."""
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        raise Exception(f"Failed to download document: HTTP {response.status}")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error downloading document: {str(e)}")

    def extract_clauses_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract clauses from text using spaCy for semantic segmentation."""
        doc = self.nlp(text)
        clauses = []
        current_clause = ""
        clause_number = 1
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            # Check if this sentence starts a new clause (basic heuristic)
            if (sentence_text.startswith(tuple('0123456789')) or 
                sentence_text.lower().startswith(('article', 'section', 'clause', 'paragraph'))):
                
                # Save previous clause if it exists
                if current_clause.strip():
                    clauses.append({
                        "text": current_clause.strip(),
                        "clause_number": clause_number - 1,
                        "clause_metadata": {
                            "clause_number": clause_number - 1,
                            "word_count": len(current_clause.split())
                        }
                    })
                
                # Start new clause
                current_clause = sentence_text
                clause_number += 1
            else:
                # Add to current clause
                current_clause += " " + sentence_text
        
        # Add the last clause
        if current_clause.strip():
            clauses.append({
                "text": current_clause.strip(),
                "clause_number": clause_number - 1,
                "clause_metadata": {
                    "clause_number": clause_number - 1,
                    "word_count": len(current_clause.split())
                }
            })
        
        return clauses

    async def generate_summary(self, document_text: str) -> str:
        """Generate document summary using Kimi."""
        try:
            response = await groq_client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal document analyst. Provide a concise summary of the following legal document, highlighting key points, obligations, and important clauses."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize this legal document:\n\n{document_text[:120000]}"
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using sentence-transformers."""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()

    async def upsert_to_pinecone(self, clauses: List[Dict[str, Any]], document_id: int, document_url: str):
        """Upsert embeddings to Pinecone with metadata."""
        texts = [clause["text"] for clause in clauses]
        embeddings = self.generate_embeddings(texts)
        
        vectors = []
        for i, (clause, embedding) in enumerate(zip(clauses, embeddings)):
            vector_id = f"doc_{document_id}_clause_{i}"
            
            metadata = {
                "document_id": document_id,
                "clause_text": clause["text"][:1000],
                "source_document_url": document_url,
                "clause_index": i,
                "clause_number": clause.get("clause_number", i),
                "word_count": clause["clause_metadata"].get("word_count", 0)
            }
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.pinecone_index.upsert(vectors=batch)

    async def process_document(self, document_url: str, file_content: Optional[bytes] = None) -> int:
        """Main orchestration method for document processing."""
        db = next(get_db())
        document = None
        
        try:
            document = crud_document.create_document(db, document_in=DocumentCreate(url=document_url, status="processing"))
            
            if file_content is None:
                file_content = await self.download_document(document_url)
            
            parser = get_parser(document_url)
            if parser is None:
                raise Exception(f"No parser available for document: {document_url}")
            
            document_text = parser(file_content)
            
            summary = await self.generate_summary(document_text)
            
            crud_document.update_document(db, db_obj=document, obj_in=DocumentUpdate(summary=summary, status="summarized"))
            
            clauses_data = self.extract_clauses_with_spacy(document_text)
            
            clauses_to_create = [
                ClauseCreate(document_id=document.id, text=c["text"], clause_metadata=c["clause_metadata"]) for c in clauses_data
            ]
            clause_objects = crud_document.create_clauses(db, clauses_in=clauses_to_create)
            
            await self.upsert_to_pinecone(clauses_data, document.id, document_url)
            
            for i, clause in enumerate(clause_objects):
                crud_document.update_clause_embedding_id(db, clause=clause, embedding_id=f"doc_{document.id}_clause_{i}")
            
            crud_document.update_document(db, db_obj=document, obj_in=DocumentUpdate(status="completed"))
            
            return document.id
            
        except Exception as e:
            if document:
                crud_document.update_document(db, db_obj=document, obj_in=DocumentUpdate(status="failed"))
            raise e
        finally:
            db.close()

ingestion_service = IngestionService()