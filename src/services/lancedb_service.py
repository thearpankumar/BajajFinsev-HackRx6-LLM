import logging
import lancedb
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from lancedb.pydantic import pydantic_to_schema, Vector

from src.core.config import settings

logger = logging.getLogger(__name__)

class LanceDBSchema(BaseModel):
    vector: Vector(settings.EMBEDDING_DIMENSIONS)
    text: str
    document_url: str
    page_number: int
    chunk_index: int
    section_type: str = ""  # Type of document section
    section_summary: str = ""  # Summary of the section
    entities: str = "[]"  # JSON string representation of entities
    concepts: str = "[]"  # JSON string representation of concepts
    categories: str = "[]"  # JSON string representation of categories
    keywords: str = "[]"  # JSON string representation of keywords
    indexed_at: float = 0.0  # Timestamp when chunk was indexed
    document_hash: str = ""  # Hash of the document for versioning
    chunk_hash: str = ""  # Hash of the chunk for deduplication

class LanceDBService:
    """
    Service for interacting with the LanceDB local vector database.
    """
    def __init__(self, uri: str):
        self.db = lancedb.connect(uri)
        self.logger = logger

    def create_table_if_not_exists(self, table_name: str, force_recreate: bool = False):
        """
        Create a new LanceDB table with an explicit schema if it doesn't already exist.
        
        Args:
            table_name: Name of the table to create
            force_recreate: If True, delete existing table and recreate. If False, use existing table.
        """
        if table_name in self.db.table_names():
            if force_recreate:
                self.logger.info(f"LanceDB table '{table_name}' already exists. Force recreating as requested.")
                self.db.drop_table(table_name)
            else:
                self.logger.info(f"LanceDB table '{table_name}' already exists. Using existing table.")
                return

        self.logger.info(f"Creating new LanceDB table: {table_name}")
        try:
            # Convert the Pydantic model to a PyArrow schema for table creation.
            schema = pydantic_to_schema(LanceDBSchema)
            self.db.create_table(table_name, schema=schema)
            self.logger.info(f"Table '{table_name}' created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to create LanceDB table: {e}", exc_info=True)
            raise

    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics about the table including row count and schema info.
        """
        try:
            if table_name not in self.db.table_names():
                return {"exists": False, "row_count": 0}
            
            tbl = self.db.open_table(table_name)
            row_count = tbl.count_rows()
            
            return {
                "exists": True,
                "row_count": row_count,
                "schema": str(tbl.schema)
            }
        except Exception as e:
            self.logger.error(f"Error getting table stats: {e}")
            return {"exists": False, "row_count": 0, "error": str(e)}

    def upsert_chunks(self, table_name: str, vectors: List[List[float]], chunks: List[Dict[str, Any]]):
        """
        Upsert document chunks and their embeddings into the LanceDB table.
        """
        if not vectors or not chunks:
            self.logger.warning("Upsert called with empty vectors or chunks.")
            return

        tbl = self.db.open_table(table_name)
        
        data_to_upsert = []
        for i, (vector, chunk_data) in enumerate(zip(vectors, chunks)):
            # Convert metadata lists to JSON strings
            import json
            import hashlib
            entities_str = json.dumps(chunk_data['metadata'].get('entities', []))
            concepts_str = json.dumps(chunk_data['metadata'].get('concepts', []))
            categories_str = json.dumps(chunk_data['metadata'].get('categories', []))
            keywords_str = json.dumps(chunk_data['metadata'].get('keywords', []))
            
            # Generate chunk hash for deduplication
            chunk_text = chunk_data['metadata'].get('text', '')
            chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
            
            # Generate document hash for versioning
            document_url = chunk_data['metadata'].get('document_url', '')
            document_hash = hashlib.sha256(document_url.encode('utf-8')).hexdigest()
            
            data = {
                "vector": vector,
                "text": chunk_text,
                "document_url": document_url,
                "page_number": chunk_data['metadata'].get('page_number', -1),
                "chunk_index": i,
                "section_type": chunk_data['metadata'].get('section_type', ''),
                "section_summary": chunk_data['metadata'].get('section_summary', '')[:500],  # Limit to 500 chars
                "entities": entities_str,
                "concepts": concepts_str,
                "categories": categories_str,
                "keywords": keywords_str,
                "indexed_at": chunk_data['metadata'].get('indexed_at', 0.0),
                "document_hash": document_hash,
                "chunk_hash": chunk_hash
            }
            data_to_upsert.append(data)

        try:
            tbl.add(data_to_upsert)
            self.logger.info(f"Successfully upserted {len(data_to_upsert)} vectors into '{table_name}'.")
        except Exception as e:
            self.logger.error(f"Error upserting data to LanceDB: {e}", exc_info=True)
            raise

    def query(self, table_name: str, vector: List[float], top_k: int, filter_str: Optional[str] = None) -> List[Dict]:
        """
        Query the LanceDB table to find the most similar vectors.
        """
        tbl = self.db.open_table(table_name)
        try:
            # Explicitly specify the vector column name in the search query
            query_builder = tbl.search(vector, vector_column_name="vector").limit(top_k)
            if filter_str:
                query_builder = query_builder.where(filter_str, prefilter=True)
            
            results = query_builder.to_df().to_dict('records')
            
            # Add a 'score' field to mimic the Pinecone result structure for easier integration
            for res in results:
                res['score'] = res.pop('_distance')
                # Parse JSON strings back to lists
                import json
                try:
                    res['entities'] = json.loads(res.get('entities', '[]'))
                    res['concepts'] = json.loads(res.get('concepts', '[]'))
                    res['categories'] = json.loads(res.get('categories', '[]'))
                    res['keywords'] = json.loads(res.get('keywords', '[]'))
                except json.JSONDecodeError:
                    # Fallback to empty lists if parsing fails
                    res['entities'] = []
                    res['concepts'] = []
                    res['categories'] = []
                    res['keywords'] = []
                
                # Add section information
                res['section_type'] = res.get('section_type', '')
                res['section_summary'] = res.get('section_summary', '')
                res['document_hash'] = res.get('document_hash', '')
                res['chunk_hash'] = res.get('chunk_hash', '')

            self.logger.info(f"Query returned {len(results)} results from '{table_name}'.")
            return results
        except Exception as e:
            self.logger.error(f"Error querying LanceDB: {e}", exc_info=True)
            return []
    
    def query_hierarchical(self, table_name: str, vector: List[float], top_k: int,
                             document_hash: Optional[str] = None,
                             section_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Query the LanceDB table with hierarchical filtering for large-scale datasets.
        """
        tbl = self.db.open_table(table_name)
        try:
            # Build filter string for hierarchical querying
            filters = []
            if document_hash:
                filters.append(f"document_hash = '{document_hash}'")
            if section_types:
                # Create OR condition for section types
                section_filters = [f"section_type = '{stype}'" for stype in section_types]
                filters.append(f"({' OR '.join(section_filters)})")
            
            filter_str = " AND ".join(filters) if filters else None
            
            # Explicitly specify the vector column name in the search query
            query_builder = tbl.search(vector, vector_column_name="vector").limit(top_k)
            if filter_str:
                query_builder = query_builder.where(filter_str, prefilter=True)
            
            results = query_builder.to_df().to_dict('records')
            
            # Add a 'score' field and parse metadata
            for res in results:
                res['score'] = res.pop('_distance')
                # Parse JSON strings back to lists
                import json
                try:
                    res['entities'] = json.loads(res.get('entities', '[]'))
                    res['concepts'] = json.loads(res.get('concepts', '[]'))
                    res['categories'] = json.loads(res.get('categories', '[]'))
                    res['keywords'] = json.loads(res.get('keywords', '[]'))
                except json.JSONDecodeError:
                    # Fallback to empty lists if parsing fails
                    res['entities'] = []
                    res['concepts'] = []
                    res['categories'] = []
                    res['keywords'] = []
                
                # Add section information
                res['section_type'] = res.get('section_type', '')
                res['section_summary'] = res.get('section_summary', '')
                res['document_hash'] = res.get('document_hash', '')
                res['chunk_hash'] = res.get('chunk_hash', '')

            self.logger.info(f"Hierarchical query returned {len(results)} results from '{table_name}'.")
            return results
        except Exception as e:
            self.logger.error(f"Error in hierarchical querying LanceDB: {e}", exc_info=True)
            return []

# Global instance and factory function
lancedb_service: Optional[LanceDBService] = None

def get_lancedb_service():
    """Factory function to get a singleton instance of the LanceDB service."""
    global lancedb_service
    if lancedb_service is None:
        lancedb_service = LanceDBService(uri=settings.LANCEDB_PATH)
    return lancedb_service
