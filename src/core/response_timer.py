"""
Response Timer Utility
Ensures consistent response times for better user experience
"""

import asyncio
import logging
import random
import time
from typing import Any

from src.core.config import settings

logger = logging.getLogger(__name__)


class ResponseTimer:
    """Handles response timing to ensure consistent user experience"""

    def __init__(self):
        self.start_time = None
        self.min_time = settings.min_response_time_seconds
        self.max_time = settings.max_response_time_seconds
        self.enabled = settings.enable_response_delay

    def start(self):
        """Start timing the response"""
        self.start_time = time.time()
        logger.info("⏱️ Response timer started")

    async def ensure_minimum_time(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure response takes at least minimum time
        If processing was faster, add artificial delay
        If processing was slower, return immediately
        """
        if not self.enabled or self.start_time is None:
            return result

        elapsed_time = time.time() - self.start_time

        if elapsed_time < self.min_time:
            # Process was too fast, add delay
            target_time = random.uniform(self.min_time, self.max_time)
            delay_needed = target_time - elapsed_time

            logger.info(f"⏳ Process completed in {elapsed_time:.2f}s, adding {delay_needed:.2f}s delay")
            logger.info(f"🎯 Target response time: {target_time:.2f}s")

            await asyncio.sleep(delay_needed)

            # Update the processing time in result
            final_time = time.time() - self.start_time
            if 'processing_time' in result:
                result['processing_time'] = final_time

            logger.info(f"✅ Response delivered in {final_time:.2f}s")
        else:
            # Process took longer than minimum, no delay needed
            logger.info(f"✅ Process completed in {elapsed_time:.2f}s (no delay needed)")

        return result

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


# Global instance for easy access
response_timer = ResponseTimer()
