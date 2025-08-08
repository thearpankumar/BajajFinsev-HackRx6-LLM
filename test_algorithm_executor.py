#!/usr/bin/env python3
"""
Test script for Algorithm Executor
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.algorithm_executor import AlgorithmExecutor

async def test_algorithm_executor():
    """Test the Algorithm Executor"""
    
    print("🤖 Testing Algorithm Executor")
    
    # Create executor
    try:
        executor = AlgorithmExecutor()
        print("✅ Algorithm Executor created")
    except Exception as e:
        print(f"❌ Failed to create executor: {e}")
        return
    
    # Test algorithm detection
    test_text = """
    This document describes Sachin's Parallel World Discovery algorithm:
    Step 1: Call the API at https://register.hackrx.in/submissions/myFavouriteCity
    Step 2: Map the city to a landmark
    Step 3: Call the appropriate flight endpoint based on the landmark
    Step 4: Return the flight number
    """
    
    print(f"\nTesting algorithm detection...")
    algorithm_type = executor.detect_algorithm_in_text(test_text)
    print(f"Detected algorithm: {algorithm_type}")
    
    if algorithm_type:
        print(f"\n🚀 Executing {algorithm_type} algorithm...")
        try:
            result = await executor.execute_algorithm(algorithm_type)
            print(f"✅ Algorithm result: {result}")
        except Exception as e:
            print(f"❌ Algorithm execution failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No algorithm detected")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_algorithm_executor())