import os
from typing import List, Dict, Any, Optional
import aiohttp
import spacy
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import AsyncOpenAI

from ..db.base import get_db
from ..db.models import Document, Clause
from ..utils.document_parsers import get_parser


class IngestionService:
    def __init__(self):
        # Validate required environment variables
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
            
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.embedding_model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        
        # Initialize Pinecone with error handling
        try:
            pc = Pinecone(api_key=pinecone_key)
            self.pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "bajaj-legal-docs"))
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
                        "metadata": {
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
                "metadata": {
                    "clause_number": clause_number - 1,
                    "word_count": len(current_clause.split())
                }
            })
        
        return clauses

    async def generate_summary(self, document_text: str) -> str:
        """Generate document summary using OpenAI GPT-4."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal document analyst. Provide a concise summary of the following legal document, highlighting key points, obligations, and important clauses."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize this legal document:\n\n{document_text[:4000]}"  # Limit text to avoid token limits
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
                "clause_text": clause["text"][:1000],  # Limit text length for metadata
                "source_document_url": document_url,
                "clause_index": i,
                "clause_number": clause.get("clause_number", i),
                "word_count": clause["metadata"].get("word_count", 0)
            }
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.pinecone_index.upsert(vectors=batch)

    async def process_document(self, document_url: str, file_content: Optional[bytes] = None) -> int:
        """Main orchestration method for document processing."""
        db = next(get_db())
        document = None
        
        try:
            # Create document record
            document = Document(url=document_url, status="processing")
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Download or use provided content
            if file_content is None:
                file_content = await self.download_document(document_url)
            
            # Parse document
            parser = get_parser(document_url)
            if parser is None:
                raise Exception(f"No parser available for document: {document_url}")
            
            document_text = parser(file_content)
            
            # Generate summary
            summary = await self.generate_summary(document_text)
            
            # Update document with summary
            document.summary = summary
            document.status = "summarized"
            db.commit()
            
            # Extract clauses
            clauses_data = self.extract_clauses_with_spacy(document_text)
            
            # Save clauses to database and prepare for embedding
            clause_objects = []
            for i, clause_data in enumerate(clauses_data):
                clause = Clause(
                    document_id=document.id,
                    text=clause_data["text"],
                    metadata=clause_data["metadata"]
                )
                clause_objects.append(clause)
                db.add(clause)
            
            db.commit()
            
            # Generate embeddings and upsert to Pinecone
            await self.upsert_to_pinecone(clauses_data, document.id, document_url)
            
            # Update embedding IDs in database
            for i, clause in enumerate(clause_objects):
                clause.embedding_id = f"doc_{document.id}_clause_{i}"
            
            # Mark document as completed
            document.status = "completed"
            db.commit()
            
            return document.id
            
        except Exception as e:
            # Mark document as failed
            if 'document' in locals() and document:
                document.status = "failed"
                db.commit()
            raise e
        finally:
            db.close()


# Singleton instance
ingestion_service = IngestionService()