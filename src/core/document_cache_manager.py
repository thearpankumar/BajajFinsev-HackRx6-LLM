"""
Document Cache Manager - Persistent caching for processed documents
Avoids re-downloading and re-processing documents across server restarts
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import aiofiles

from src.core.config import settings
from src.core.enhanced_document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentCacheManager:
    """
    Manages persistent document caching to avoid re-downloading and re-processing
    """

    def __init__(self):
        self.cache_dir = Path(settings.DOCUMENT_CACHE_PATH)
        self.metadata_file = self.cache_dir / "document_metadata.json"
        self.chunks_dir = self.cache_dir / "chunks"
        
        # Ensure cache directories exist
        self._ensure_cache_directories()
        
        # In-memory cache for faster access
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _ensure_cache_directories(self):
        """Create cache directories if they don't exist"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Document cache directories ensured at {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directories: {str(e)}")
            raise

    async def initialize(self):
        """Initialize the cache manager and load existing metadata"""
        try:
            await self._load_metadata_cache()
            await self._cleanup_expired_documents()
            logger.info("Document cache manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document cache manager: {str(e)}")
            raise

    async def _load_metadata_cache(self):
        """Load document metadata from persistent storage"""
        try:
            if self.metadata_file.exists():
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    content = await f.read()
                    self._metadata_cache = json.loads(content)
                logger.info(f"Loaded metadata for {len(self._metadata_cache)} cached documents")
            else:
                self._metadata_cache = {}
                logger.info("No existing document cache found, starting fresh")
            
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load metadata cache: {str(e)}")
            self._metadata_cache = {}
            self._loaded = True

    async def _save_metadata_cache(self):
        """Save document metadata to persistent storage"""
        try:
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(self._metadata_cache, indent=2))
            logger.debug("Document metadata cache saved")
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {str(e)}")

    async def _cleanup_expired_documents(self):
        """Remove expired documents from cache"""
        if not settings.ENABLE_PERSISTENT_DOCUMENT_CACHE:
            return

        try:
            current_time = datetime.now()
            expired_urls = []
            
            for url, metadata in self._metadata_cache.items():
                cached_time = datetime.fromisoformat(metadata.get('cached_at', '1970-01-01'))
                expiry_time = cached_time + timedelta(hours=settings.DOCUMENT_CACHE_TTL_HOURS)
                
                if current_time > expiry_time:
                    expired_urls.append(url)
            
            if expired_urls:
                logger.info(f"Cleaning up {len(expired_urls)} expired documents")
                for url in expired_urls:
                    await self._remove_document_from_cache(url)
                
                await self._save_metadata_cache()
                logger.info("Expired document cleanup completed")
            else:
                logger.info("No expired documents found")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired documents: {str(e)}")

    def _generate_document_hash(self, document_url: str, content_hash: Optional[str] = None) -> str:
        """Generate a unique hash for the document"""
        hash_input = document_url
        if content_hash:
            hash_input += content_hash
        
        if settings.DOCUMENT_HASH_ALGORITHM == "sha256":
            return hashlib.sha256(hash_input.encode()).hexdigest()
        elif settings.DOCUMENT_HASH_ALGORITHM == "md5":
            return hashlib.md5(hash_input.encode()).hexdigest()
        else:
            return hashlib.sha256(hash_input.encode()).hexdigest()

    async def is_document_cached(self, document_url: str) -> bool:
        """Check if document is cached and not expired"""
        if not settings.ENABLE_PERSISTENT_DOCUMENT_CACHE:
            return False

        if not self._loaded:
            await self._load_metadata_cache()

        if document_url not in self._metadata_cache:
            return False

        # Check if document has expired
        try:
            metadata = self._metadata_cache[document_url]
            cached_time = datetime.fromisoformat(metadata.get('cached_at', '1970-01-01'))
            expiry_time = cached_time + timedelta(hours=settings.DOCUMENT_CACHE_TTL_HOURS)
            
            if datetime.now() > expiry_time:
                logger.info(f"Document {document_url} has expired, removing from cache")
                await self._remove_document_from_cache(document_url)
                return False
            
            # Check if chunks file exists
            chunks_file = self.chunks_dir / f"{metadata['document_hash']}.json"
            if not chunks_file.exists():
                logger.warning(f"Chunks file missing for {document_url}, removing from cache")
                await self._remove_document_from_cache(document_url)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking document cache for {document_url}: {str(e)}")
            return False

    async def get_cached_document(self, document_url: str) -> Optional[Tuple[List[DocumentChunk], Dict[str, Any]]]:
        """Retrieve cached document chunks and metadata"""
        if not await self.is_document_cached(document_url):
            return None

        try:
            metadata = self._metadata_cache[document_url]
            chunks_file = self.chunks_dir / f"{metadata['document_hash']}.json"
            
            # Load chunks from file
            async with aiofiles.open(chunks_file, 'r') as f:
                content = await f.read()
                chunks_data = json.loads(content)
            
            # Reconstruct DocumentChunk objects
            chunks = []
            for chunk_data in chunks_data['chunks']:
                chunk = DocumentChunk(
                    text=chunk_data['text'],
                    page_num=chunk_data['page_num'],
                    chunk_id=chunk_data['chunk_id'],
                    metadata=chunk_data.get('metadata', {})
                )
                chunks.append(chunk)
            
            # Return chunks and metadata
            doc_metadata = {
                **chunks_data['metadata'],
                'source': 'persistent_cache',
                'cached_at': metadata['cached_at'],
                'cache_hit': True
            }
            
            logger.info(f"Retrieved {len(chunks)} chunks from cache for {document_url}")
            return chunks, doc_metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached document {document_url}: {str(e)}")
            # Remove corrupted cache entry
            await self._remove_document_from_cache(document_url)
            return None

    async def cache_document(
        self, 
        document_url: str, 
        chunks: List[DocumentChunk], 
        metadata: Dict[str, Any]
    ) -> bool:
        """Cache processed document chunks and metadata"""
        if not settings.ENABLE_PERSISTENT_DOCUMENT_CACHE:
            return False

        try:
            # Generate document hash
            content_hash = metadata.get('content_hash', '')
            document_hash = self._generate_document_hash(document_url, content_hash)
            
            # Prepare chunks data for serialization
            chunks_data = {
                'chunks': [
                    {
                        'text': chunk.text,
                        'page_num': chunk.page_num,
                        'chunk_id': chunk.chunk_id,
                        'metadata': chunk.metadata
                    }
                    for chunk in chunks
                ],
                'metadata': metadata
            }
            
            # Save chunks to file
            chunks_file = self.chunks_dir / f"{document_hash}.json"
            async with aiofiles.open(chunks_file, 'w') as f:
                await f.write(json.dumps(chunks_data, indent=2))
            
            # Update metadata cache
            self._metadata_cache[document_url] = {
                'document_hash': document_hash,
                'cached_at': datetime.now().isoformat(),
                'num_chunks': len(chunks),
                'document_size': metadata.get('size', 0),
                'content_hash': content_hash,
                'metadata': {
                    'type': metadata.get('type', 'unknown'),
                    'pages': metadata.get('pages', 0),
                    'processing_time': metadata.get('processing_time', 0)
                }
            }
            
            # Save metadata
            await self._save_metadata_cache()
            
            logger.info(f"Cached document {document_url} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache document {document_url}: {str(e)}")
            return False

    async def _remove_document_from_cache(self, document_url: str):
        """Remove document from cache"""
        try:
            if document_url in self._metadata_cache:
                metadata = self._metadata_cache[document_url]
                
                # Remove chunks file
                chunks_file = self.chunks_dir / f"{metadata['document_hash']}.json"
                if chunks_file.exists():
                    chunks_file.unlink()
                
                # Remove from metadata cache
                del self._metadata_cache[document_url]
                
                logger.debug(f"Removed document {document_url} from cache")
                
        except Exception as e:
            logger.error(f"Failed to remove document {document_url} from cache: {str(e)}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._loaded:
            await self._load_metadata_cache()

        try:
            total_documents = len(self._metadata_cache)
            total_chunks = sum(meta.get('num_chunks', 0) for meta in self._metadata_cache.values())
            total_size = sum(meta.get('document_size', 0) for meta in self._metadata_cache.values())
            
            # Calculate cache directory size
            cache_size = 0
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    cache_size += file_path.stat().st_size
            
            return {
                'enabled': settings.ENABLE_PERSISTENT_DOCUMENT_CACHE,
                'cache_path': str(self.cache_dir),
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'total_document_size': total_size,
                'cache_directory_size': cache_size,
                'ttl_hours': settings.DOCUMENT_CACHE_TTL_HOURS,
                'oldest_document': min(
                    (meta.get('cached_at', '') for meta in self._metadata_cache.values()),
                    default=None
                ),
                'newest_document': max(
                    (meta.get('cached_at', '') for meta in self._metadata_cache.values()),
                    default=None
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {'error': str(e)}

    async def clear_cache(self) -> bool:
        """Clear all cached documents"""
        try:
            # Remove all chunk files
            for file_path in self.chunks_dir.glob('*.json'):
                file_path.unlink()
            
            # Clear metadata
            self._metadata_cache = {}
            await self._save_metadata_cache()
            
            logger.info("Document cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False

    async def remove_document(self, document_url: str) -> bool:
        """Remove specific document from cache"""
        try:
            await self._remove_document_from_cache(document_url)
            await self._save_metadata_cache()
            logger.info(f"Removed document {document_url} from cache")
            return True
        except Exception as e:
            logger.error(f"Failed to remove document {document_url}: {str(e)}")
            return False
