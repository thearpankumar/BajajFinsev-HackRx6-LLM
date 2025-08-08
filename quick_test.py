#!/usr/bin/env python3
"""
Quick test for flight number extraction
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.algorithm_executor import AlgorithmExecutor

async def test_flight_extraction():
    """Test flight number extraction"""
    
    executor = AlgorithmExecutor()
    
    # Test response from the API
    test_response = '{"success":true,"message":"Bangalore flight number generated successfully","status":200,"data":{"flightNumber":"b5eb30"}}'
    
    print(f"Test response: {test_response}")
    
    flight_number = executor._extract_flight_number(test_response)
    print(f"Extracted flight number: {flight_number}")
    
    # Test full algorithm
    print("\nTesting full algorithm:")
    result = await executor.execute_flight_discovery_algorithm()
    print(f"Algorithm result: {result}")

if __name__ == "__main__":
    asyncio.run(test_flight_extraction())