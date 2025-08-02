import logging
from typing import Dict, Any
import asyncio
from datetime import datetime
import time
from src.core.config import settings

logger = logging.getLogger(__name__)

class LoadBalancingService:
    """
    Service for implementing sophisticated load balancing mechanisms.
    """
    
    def __init__(self):
        self.logger = logger
        self.request_queue = asyncio.Queue()
        self.worker_tasks = []
        self.max_workers = settings.PARALLEL_BATCHES
        self.current_load = {}
        self.response_times = {}
        
    async def start_load_balancer(self):
        """
        Start the load balancer with worker tasks.
        """
        self.logger.info("Starting load balancer")
        
        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        self.logger.info(f"Load balancer started with {self.max_workers} workers")
    
    async def stop_load_balancer(self):
        """
        Stop the load balancer and all worker tasks.
        """
        self.logger.info("Stopping load balancer")
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.logger.info("Load balancer stopped")
    
    async def _worker(self, worker_id: str):
        """
        Worker task that processes requests from the queue.
        
        Args:
            worker_id: Identifier for the worker
        """
        self.logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get request from queue
                request_data = await self.request_queue.get()
                
                # Process request
                start_time = time.time()
                await self._process_request(request_data, worker_id)
                end_time = time.time()
                
                # Record response time
                response_time = end_time - start_time
                if worker_id not in self.response_times:
                    self.response_times[worker_id] = []
                self.response_times[worker_id].append(response_time)
                
                # Mark task as done
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                self.logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in worker {worker_id}: {e}")
    
    async def _process_request(self, request_data: Dict[str, Any], worker_id: str):
        """
        Process a single request.
        
        Args:
            request_data: Request data to process
            worker_id: Identifier for the worker processing the request
        """
        try:
            # Extract request information
            request_id = request_data.get('request_id', 'unknown')
            document_url = request_data.get('document_url', '')
            questions = request_data.get('questions', [])
            
            self.logger.info(f"Worker {worker_id} processing request {request_id}")
            
            # Import required services
            from src.services import ingestion_service, rag_workflow_service
            
            # Process document
            document_chunks = await ingestion_service.process_and_extract(document_url)
            
            # Index document
            await rag_workflow_service.embedding_service.embed_and_upsert_chunks(document_url, document_chunks)
            
            # Process questions
            results = await rag_workflow_service.run_parallel_workflow(document_url, questions, document_chunks)
            
            # Store results
            request_data['results'] = results
            request_data['status'] = 'completed'
            
            self.logger.info(f"Worker {worker_id} completed request {request_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing request in worker {worker_id}: {e}")
            request_data['error'] = str(e)
            request_data['status'] = 'failed'
    
    async def submit_request(self, request_data: Dict[str, Any]) -> str:
        """
        Submit a request to the load balancer.
        
        Args:
            request_data: Request data to process
            
        Returns:
            Request ID
        """
        # Add request ID and timestamp
        request_id = f"req-{int(time.time() * 1000)}"
        request_data['request_id'] = request_id
        request_data['timestamp'] = datetime.now().isoformat()
        request_data['status'] = 'queued'
        
        # Add to queue
        await self.request_queue.put(request_data)
        
        self.logger.info(f"Request {request_id} queued")
        return request_id
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """
        Get the current status of the load balancer.
        
        Returns:
            Load balancer status information
        """
        # Calculate average response times
        avg_response_times = {}
        for worker_id, times in self.response_times.items():
            if times:
                avg_response_times[worker_id] = sum(times) / len(times)
        
        # Get queue size
        queue_size = self.request_queue.qsize()
        
        return {
            'workers': self.max_workers,
            'queue_size': queue_size,
            'avg_response_times': avg_response_times,
            'current_load': self.current_load
        }
    
    def get_worker_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for all workers.
        
        Returns:
            Worker performance metrics
        """
        performance_data = {}
        
        for worker_id, times in self.response_times.items():
            if times:
                performance_data[worker_id] = {
                    'total_requests': len(times),
                    'avg_response_time': sum(times) / len(times),
                    'min_response_time': min(times),
                    'max_response_time': max(times)
                }
        
        return performance_data

# Global instance
load_balancing_service = LoadBalancingService()