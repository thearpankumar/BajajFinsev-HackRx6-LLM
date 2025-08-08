#!/usr/bin/env python3
"""
Direct test for payload 22 algorithm execution
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.algorithm_executor import AlgorithmExecutor

async def test_payload22_algorithm():
    """Test the algorithm execution for payload 22"""
    
    print("✈️ Testing Payload 22 Flight Discovery Algorithm")
    
    # Create executor
    executor = AlgorithmExecutor()
    
    try:
        # Execute the algorithm directly
        print("\n🚀 Executing flight discovery algorithm...")
        result = await executor.execute_flight_discovery_algorithm()
        
        print(f"\n✅ Flight number result: {result}")
        
        # Test detection
        test_text = """
        The document describes a flight discovery algorithm with steps:
        1. Call API at register.hackrx.in
        2. Get city name
        3. Map to landmark
        4. Call flight endpoint
        """
        
        algorithm_type = executor.detect_algorithm_in_text(test_text)
        print(f"\n🤖 Algorithm detection test: {algorithm_type}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_payload22_algorithm())