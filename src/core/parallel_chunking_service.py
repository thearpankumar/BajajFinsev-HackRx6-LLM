"""
Parallel Chunking Service
High-performance parallel document chunking for massive documents (650k+ tokens)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from src.core.config import config
from src.core.hierarchical_chunker import HierarchicalChunker

logger = logging.getLogger(__name__)


class ParallelChunkingService:
    """
    High-performance parallel chunking service for large documents
    Uses multiprocessing for CPU-intensive chunking operations
    """
    
    def __init__(self):
        # Use configured max workers for chunking
        self.max_workers = min(config.max_workers, multiprocessing.cpu_count())
        self.chunker = HierarchicalChunker()
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        
        logger.info(f"ParallelChunkingService initialized with {self.max_workers} workers")
    
    async def parallel_chunk_documents(
        self, 
        document_results: List[Dict[str, Any]], 
        chunking_strategy: str = "hierarchical",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process multiple documents in parallel with chunking
        
        Args:
            document_results: List of processed document results
            chunking_strategy: Chunking strategy to use
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with all chunks and metadata
        """
        logger.info(f"🚀 Starting parallel chunking for {len(document_results)} documents")
        start_time = time.time()
        
        all_chunks = []
        chunking_results = []
        errors = []
        
        try:
            # For large documents, use parallel processing
            if len(document_results) > 1 or self._estimate_large_document(document_results):
                # Use asyncio to manage parallel chunking
                chunking_tasks = []
                
                for i, doc_result in enumerate(document_results):
                    if not doc_result.get("has_content") or not doc_result.get("content_summary"):
                        continue
                        
                    # Create chunking task
                    task = asyncio.create_task(
                        self._chunk_single_document(doc_result, chunking_strategy, i)
                    )
                    chunking_tasks.append(task)
                
                # Process all documents in parallel
                if chunking_tasks:
                    logger.info(f"⚡ Processing {len(chunking_tasks)} documents in parallel")
                    
                    completed_tasks = 0
                    for completed_task in asyncio.as_completed(chunking_tasks):
                        chunk_result = await completed_task
                        
                        if chunk_result["status"] == "success":
                            chunks = chunk_result["chunks"]
                            all_chunks.extend(chunks)
                            chunking_results.append(chunk_result)
                        else:
                            errors.append(chunk_result["error"])
                        
                        completed_tasks += 1
                        if progress_callback:
                            progress = 40 + (completed_tasks / len(chunking_tasks)) * 30
                            await progress_callback(
                                f"Chunked {completed_tasks}/{len(chunking_tasks)} documents", 
                                progress
                            )
                            
            else:
                # Single document - process normally but optimized
                doc_result = document_results[0]
                chunk_result = await self._chunk_single_document(doc_result, chunking_strategy, 0)
                
                if chunk_result["status"] == "success":
                    all_chunks.extend(chunk_result["chunks"])
                    chunking_results.append(chunk_result)
                else:
                    errors.append(chunk_result["error"])
            
            processing_time = time.time() - start_time
            
            result = {
                "status": "success" if all_chunks else "error",
                "total_chunks": len(all_chunks),
                "chunks": all_chunks,
                "chunking_results": chunking_results,
                "processing_time": processing_time,
                "parallel_workers_used": self.max_workers,
                "documents_processed": len([r for r in chunking_results if r["status"] == "success"]),
                "errors": errors
            }
            
            logger.info(f"✅ Parallel chunking completed: {len(all_chunks)} chunks in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Parallel chunking failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "total_chunks": len(all_chunks),
                "chunks": all_chunks,
                "processing_time": time.time() - start_time
            }
    
    async def _chunk_single_document(
        self, 
        doc_result: Dict[str, Any], 
        chunking_strategy: str,
        doc_index: int
    ) -> Dict[str, Any]:
        """Chunk a single document asynchronously"""
        try:
            # Get document text from content field
            document_text = ""
            if doc_result.get("content") and "full_text" in doc_result["content"]:
                document_text = doc_result["content"]["full_text"]
            
            if not document_text:
                return {
                    "status": "error",
                    "error": f"No text content for document: {doc_result['document_url']}",
                    "chunks": [],
                    "document_index": doc_index
                }
            
            # Create source info for chunking
            source_info = {
                "document_url": doc_result["document_url"],
                "file_path": doc_result["file_path"],
                "processing_time": doc_result["processing_time"],
                "worker_id": doc_result.get("worker_id"),
                "content_summary": doc_result.get("content_summary")
            }
            
            # For very large documents, use optimized chunking
            if len(document_text) > 100000:  # > 100k characters
                chunk_result = await self._optimize_large_document_chunking(
                    document_text, source_info, chunking_strategy
                )
            else:
                # Regular chunking for smaller documents
                chunk_result = await self.chunker.chunk_document(
                    document_text,
                    source_info,
                    chunking_strategy
                )
            
            if chunk_result.chunks:
                return {
                    "status": "success",
                    "chunks": chunk_result.chunks,
                    "chunk_count": len(chunk_result.chunks),
                    "document_url": doc_result["document_url"],
                    "document_index": doc_index,
                    "chunking_metadata": {
                        "strategy": chunking_strategy,
                        "chunk_size": self.chunk_size,
                        "overlap": self.chunk_overlap,
                        "source_text_length": len(document_text)
                    }
                }
            else:
                return {
                    "status": "error",
                    "error": f"No chunks created for document: {doc_result['document_url']}",
                    "chunks": [],
                    "document_index": doc_index
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Chunking failed for document {doc_index}: {str(e)}",
                "chunks": [],
                "document_index": doc_index
            }
    
    async def _optimize_large_document_chunking(
        self, 
        document_text: str, 
        source_info: Dict[str, Any],
        chunking_strategy: str
    ) -> Any:
        """Optimized chunking for very large documents using divide-and-conquer"""
        logger.info(f"🔥 Using optimized chunking for large document ({len(document_text)} chars)")
        
        # For extremely large documents, split into manageable segments first
        if len(document_text) > 500000:  # > 500k characters
            return await self._segment_and_chunk(document_text, source_info, chunking_strategy)
        else:
            # Use regular hierarchical chunking but with optimized parameters
            return await self.chunker.chunk_document(
                document_text,
                source_info,
                chunking_strategy
            )
    
    async def _segment_and_chunk(
        self, 
        document_text: str, 
        source_info: Dict[str, Any],
        chunking_strategy: str
    ) -> Any:
        """Segment extremely large documents and chunk in parallel"""
        logger.info(f"⚡ Segmenting massive document ({len(document_text)} chars) for parallel processing")
        
        # Split document into large segments (around 100k chars each)
        segment_size = 100000
        segments = []
        
        for i in range(0, len(document_text), segment_size):
            # Make sure we don't split in the middle of a sentence
            end_pos = min(i + segment_size, len(document_text))
            
            # Look for sentence boundary within last 1000 chars
            if end_pos < len(document_text):
                boundary_search = document_text[end_pos-1000:end_pos+1000]
                sentence_ends = ['. ', '. \n', '.\n', '! ', '? ']
                
                best_boundary = -1
                for sentence_end in sentence_ends:
                    pos = boundary_search.rfind(sentence_end)
                    if pos != -1:
                        best_boundary = max(best_boundary, pos)
                
                if best_boundary != -1:
                    end_pos = end_pos - 1000 + best_boundary + 2
            
            segment_text = document_text[i:end_pos]
            if segment_text.strip():
                segments.append({
                    "text": segment_text,
                    "start_pos": i,
                    "end_pos": end_pos,
                    "segment_index": len(segments)
                })
        
        logger.info(f"📦 Split document into {len(segments)} segments")
        
        # Process segments in parallel
        all_chunks = []
        segment_tasks = []
        
        for segment in segments:
            # Create modified source info for segment
            segment_source = source_info.copy()
            segment_source["segment_index"] = segment["segment_index"]
            segment_source["segment_start"] = segment["start_pos"]
            segment_source["segment_end"] = segment["end_pos"]
            
            # Create chunking task for segment
            task = asyncio.create_task(
                self.chunker.chunk_document(
                    segment["text"],
                    segment_source,
                    chunking_strategy
                )
            )
            segment_tasks.append(task)
        
        # Wait for all segments to complete
        for completed_task in asyncio.as_completed(segment_tasks):
            segment_result = await completed_task
            if segment_result.chunks:
                all_chunks.extend(segment_result.chunks)
        
        logger.info(f"✅ Parallel segmented chunking created {len(all_chunks)} chunks")
        
        # Create a combined result object
        class CombinedChunkResult:
            def __init__(self, chunks):
                self.chunks = chunks
                self.metadata = {
                    "segmented_chunking": True,
                    "segment_count": len(segments),
                    "total_chunks": len(chunks)
                }
        
        return CombinedChunkResult(all_chunks)
    
    def _estimate_large_document(self, document_results: List[Dict[str, Any]]) -> bool:
        """Estimate if documents are large enough to benefit from parallel processing"""
        total_size = 0
        for doc_result in document_results:
            if doc_result.get("content") and "full_text" in doc_result["content"]:
                total_size += len(doc_result["content"]["full_text"])
        
        # Consider "large" if total text is over 50k characters
        return total_size > 50000


# Global instance
parallel_chunking_service = ParallelChunkingService()