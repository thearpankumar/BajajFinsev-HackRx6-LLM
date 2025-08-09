"""
Smart Service Manager - Optimized Initialization and Resource Sharing
Keeps all existing services but eliminates performance bottlenecks through:
- Lazy loading of heavy components
- Shared resource pools
- Parallel initialization where possible  
- Smart dependency management
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import asynccontextmanager

from src.core.config import config

logger = logging.getLogger(__name__)

@dataclass
class ServiceInfo:
    """Service metadata and initialization info"""
    name: str
    instance: Any = None
    is_initialized: bool = False
    initialization_time: float = 0.0
    dependencies: List[str] = None
    is_heavy: bool = False  # GPU models, large files, etc.
    is_essential: bool = True  # Required for basic functionality
    lazy_load: bool = False  # Load only when first used
    init_function: Optional[Callable] = None
    error: Optional[str] = None

class SharedResourcePool:
    """Shared resources to avoid duplication across services"""
    
    def __init__(self):
        self._resources = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, key: str, factory: Callable, *args, **kwargs):
        """Get existing resource or create new one"""
        with self._lock:
            if key not in self._resources:
                self._resources[key] = factory(*args, **kwargs)
            return self._resources[key]
    
    def set(self, key: str, resource: Any):
        """Set a resource"""
        with self._lock:
            self._resources[key] = resource
    
    def get(self, key: str, default=None):
        """Get a resource"""
        with self._lock:
            return self._resources.get(key, default)

class SmartServiceManager:
    """
    Intelligent service initialization and management
    - Parallel initialization of independent services
    - Resource sharing to eliminate duplication
    - Lazy loading of heavy components
    - Smart error handling and fallbacks
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.shared_resources = SharedResourcePool()
        self.initialization_order: List[List[str]] = []  # Parallel groups
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.is_initialized = False
        self.total_initialization_time = 0.0
        
        # Performance tracking
        self.stats = {
            'services_initialized': 0,
            'services_failed': 0,
            'lazy_loaded': 0,
            'resource_reuses': 0,
            'parallel_groups': 0
        }
        
        logger.info("Smart Service Manager initialized")
    
    def register_service(
        self, 
        name: str, 
        service_class: type,
        dependencies: List[str] = None,
        is_heavy: bool = False,
        is_essential: bool = True,
        lazy_load: bool = False,
        init_args: tuple = (),
        init_kwargs: dict = None
    ):
        """Register a service for smart initialization"""
        
        def init_function():
            if init_kwargs:
                return service_class(*init_args, **init_kwargs)
            else:
                return service_class(*init_args)
        
        self.services[name] = ServiceInfo(
            name=name,
            dependencies=dependencies or [],
            is_heavy=is_heavy,
            is_essential=is_essential,
            lazy_load=lazy_load,
            init_function=init_function
        )
        
        logger.debug(f"Registered service: {name} (heavy: {is_heavy}, lazy: {lazy_load})")
    
    def _build_initialization_order(self):
        """Build parallel initialization groups based on dependencies"""
        # Simple topological sort with parallelization
        remaining = set(self.services.keys())
        self.initialization_order = []
        
        while remaining:
            # Find services with no unmet dependencies
            ready = set()
            for service_name in remaining:
                service = self.services[service_name]
                if service.lazy_load:
                    continue  # Skip lazy-loaded services
                    
                dependencies_met = all(
                    dep_name in [s.name for s in self.services.values() if s.is_initialized]
                    or dep_name not in self.services
                    for dep_name in service.dependencies
                )
                
                if dependencies_met:
                    ready.add(service_name)
            
            if not ready and remaining:
                # Handle circular dependencies by forcing initialization
                ready = {next(iter(remaining))}
                logger.warning(f"Potential circular dependency, forcing initialization of: {ready}")
            
            if ready:
                self.initialization_order.append(list(ready))
                remaining -= ready
            else:
                break
    
    async def initialize_all(self) -> Dict[str, Any]:
        """Initialize all services with smart optimization"""
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting Smart Service Initialization...")
            
            # Setup shared resources first
            await self._setup_shared_resources()
            
            # Build initialization order
            self._build_initialization_order()
            
            # Initialize services in parallel groups
            for group_idx, service_group in enumerate(self.initialization_order):
                logger.info(f"🔄 Initializing group {group_idx + 1}: {service_group}")
                await self._initialize_service_group(service_group)
                self.stats['parallel_groups'] += 1
            
            # Verify essential services
            failed_essential = [
                name for name, service in self.services.items()
                if service.is_essential and not service.is_initialized and not service.lazy_load
            ]
            
            if failed_essential:
                return {
                    "status": "partial",
                    "message": f"Some essential services failed: {failed_essential}",
                    "failed_services": failed_essential
                }
            
            self.is_initialized = True
            self.total_initialization_time = time.time() - start_time
            
            logger.info(f"✅ Smart initialization completed in {self.total_initialization_time:.2f}s")
            
            return {
                "status": "success",
                "message": f"All services initialized successfully",
                "total_time": self.total_initialization_time,
                "stats": self.stats,
                "services": {
                    name: {
                        "initialized": service.is_initialized,
                        "time": service.initialization_time,
                        "lazy": service.lazy_load
                    }
                    for name, service in self.services.items()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Smart initialization failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    async def _setup_shared_resources(self):
        """Setup shared resources that multiple services will use"""
        logger.info("🔧 Setting up shared resources...")
        
        # Shared GPU service - only one instance needed
        from src.core.gpu_service import GPUService
        gpu_service = GPUService()
        gpu_result = gpu_service.initialize()
        self.shared_resources.set('gpu_service', gpu_service)
        
        # Shared Redis connection - reused across services
        from src.services.redis_cache import redis_manager
        if not redis_manager.is_connected:
            try:
                await redis_manager.initialize()
                self.shared_resources.set('redis_manager', redis_manager)
                logger.info("✅ Shared Redis connection established")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
                self.shared_resources.set('redis_manager', None)
        
        # Shared thread pool for CPU-bound operations
        self.shared_resources.set('cpu_thread_pool', self.thread_pool)
        
        logger.info("✅ Shared resources ready")
    
    async def _initialize_service_group(self, service_names: List[str]):
        """Initialize a group of services in parallel"""
        tasks = []
        
        for service_name in service_names:
            task = asyncio.create_task(self._initialize_single_service(service_name))
            tasks.append(task)
        
        # Wait for all services in group to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for service_name, result in zip(service_names, results):
            service = self.services[service_name]
            if isinstance(result, Exception):
                service.error = str(result)
                logger.error(f"❌ {service_name} initialization failed: {result}")
                if service.is_essential:
                    self.stats['services_failed'] += 1
            else:
                service.is_initialized = True
                self.stats['services_initialized'] += 1
                logger.info(f"✅ {service_name} initialized in {service.initialization_time:.2f}s")
    
    async def _initialize_single_service(self, service_name: str):
        """Initialize a single service with resource sharing"""
        service = self.services[service_name]
        start_time = time.time()
        
        try:
            # Create service instance with shared resources
            if service_name == 'embedding_service':
                # Use shared GPU service
                gpu_service = self.shared_resources.get('gpu_service')
                from src.services.embedding_service import EmbeddingService
                service.instance = EmbeddingService(gpu_service)
                self.stats['resource_reuses'] += 1
                
            elif service_name == 'parallel_vector_store':
                # Use shared GPU and embedding services
                gpu_service = self.shared_resources.get('gpu_service')
                embedding_service = self.get_service('embedding_service')
                from src.core.parallel_vector_store import ParallelVectorStore
                service.instance = ParallelVectorStore(embedding_service, gpu_service)
                self.stats['resource_reuses'] += 2
                
            else:
                # Use standard initialization
                service.instance = service.init_function()
            
            # Initialize the service if it has an initialize method
            if hasattr(service.instance, 'initialize'):
                if asyncio.iscoroutinefunction(service.instance.initialize):
                    await service.instance.initialize()
                else:
                    # Run sync initialization in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self.thread_pool, service.instance.initialize)
            
            service.initialization_time = time.time() - start_time
            
        except Exception as e:
            service.error = str(e)
            service.initialization_time = time.time() - start_time
            raise e
    
    def get_service(self, name: str, auto_initialize: bool = True):
        """Get a service, with lazy loading if needed"""
        if name not in self.services:
            raise KeyError(f"Service '{name}' not registered")
        
        service = self.services[name]
        
        # Handle lazy loading
        if service.lazy_load and not service.is_initialized and auto_initialize:
            logger.info(f"🔄 Lazy loading service: {name}")
            
            # Synchronous lazy loading (for now)
            try:
                start_time = time.time()
                service.instance = service.init_function()
                
                if hasattr(service.instance, 'initialize'):
                    service.instance.initialize()
                
                service.is_initialized = True
                service.initialization_time = time.time() - start_time
                self.stats['lazy_loaded'] += 1
                
                logger.info(f"✅ Lazy loaded {name} in {service.initialization_time:.2f}s")
                
            except Exception as e:
                service.error = str(e)
                logger.error(f"❌ Lazy loading failed for {name}: {e}")
                raise e
        
        if not service.is_initialized:
            raise RuntimeError(f"Service '{name}' is not initialized")
        
        return service.instance
    
    def get_initialization_stats(self) -> Dict[str, Any]:
        """Get detailed initialization statistics"""
        initialized_services = {
            name: service for name, service in self.services.items() 
            if service.is_initialized
        }
        
        failed_services = {
            name: service.error for name, service in self.services.items()
            if service.error
        }
        
        return {
            'total_services': len(self.services),
            'initialized': len(initialized_services),
            'failed': len(failed_services),
            'lazy_loaded': self.stats['lazy_loaded'],
            'resource_reuses': self.stats['resource_reuses'],
            'parallel_groups': self.stats['parallel_groups'],
            'total_time': self.total_initialization_time,
            'average_time_per_service': self.total_initialization_time / max(1, len(initialized_services)),
            'failed_services': failed_services,
            'service_times': {
                name: service.initialization_time 
                for name, service in initialized_services.items()
            }
        }


# Global instance
smart_service_manager = SmartServiceManager()