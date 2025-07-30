# src/worker.py
import asyncio
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# 1. Create a queue that will be shared with the main application
cleanup_queue: asyncio.Queue = asyncio.Queue()

async def cleanup_worker():
    """
    A long-running worker that waits for file names to appear in the queue
    and launches a non-blocking cleanup process for each one.
    """
    logger.info("Cleanup worker started.")
    while True:
        try:
            # Wait until a file name is available in the queue
            file_name = await cleanup_queue.get()
            
            logger.info(f"Worker picked up cleanup task for: {file_name}")
            
            # Launch the fully detached cleanup process
            subprocess.Popen(
                [sys.executable, "src/utils/cleanup_task.py", file_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            # Mark the task as done
            cleanup_queue.task_done()
            
        except asyncio.CancelledError:
            logger.info("Cleanup worker is shutting down.")
            break
        except Exception as e:
            logger.error(f"Error in cleanup worker: {e}", exc_info=True)

# Global variable to hold the worker task
worker_task = None

def start_worker():
    """Starts the cleanup worker as a background task."""
    global worker_task
    logger.info("Starting background cleanup worker...")
    worker_task = asyncio.create_task(cleanup_worker())

def stop_worker():
    """Stops the cleanup worker gracefully."""
    global worker_task
    if worker_task:
        logger.info("Stopping background cleanup worker...")
        worker_task.cancel()
        worker_task = None
