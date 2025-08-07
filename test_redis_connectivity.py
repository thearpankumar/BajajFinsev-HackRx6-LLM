"""
Redis Connectivity Test Script
Tests the Redis connection manager and caching functionality
"""

import asyncio
import sys
import json
from datetime import datetime
from src.services.redis_cache import initialize_redis, shutdown_redis, redis_operation

async def test_basic_connectivity():
    """Test basic Redis connection"""
    print("🔍 Testing Redis basic connectivity...")
    
    try:
        # Initialize Redis
        manager, cache = await initialize_redis(host="localhost", port=6379)
        
        # Test ping
        health = await manager.health_check(force=True)
        if health:
            print("✅ Redis connection successful")
        else:
            print("❌ Redis connection failed")
            return False
            
        # Get connection stats
        stats = await manager.get_connection_stats()
        print(f"📊 Connection stats: {json.dumps(stats, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connectivity test failed: {str(e)}")
        return False


async def test_cache_operations():
    """Test Redis cache operations"""
    print("\n🧪 Testing Redis cache operations...")
    
    try:
        async with redis_operation() as cache:
            # Test basic set/get
            test_key = "test_basic"
            test_value = {"message": "Hello Redis!", "timestamp": str(datetime.now())}
            
            # Set value
            set_result = await cache.set(test_key, test_value, ttl=300)
            print(f"SET result: {'✅' if set_result else '❌'}")
            
            # Get value
            retrieved = await cache.get(test_key)
            print(f"GET result: {'✅' if retrieved == test_value else '❌'}")
            print(f"Retrieved value: {retrieved}")
            
            # Test existence check
            exists = await cache.exists(test_key)
            print(f"EXISTS result: {'✅' if exists else '❌'}")
            
            # Test increment
            counter_key = "test_counter"
            count1 = await cache.increment(counter_key)
            count2 = await cache.increment(counter_key, amount=5)
            print(f"INCREMENT results: {count1} -> {count2} {'✅' if count2 == 6 else '❌'}")
            
            return True
            
    except Exception as e:
        print(f"❌ Cache operations test failed: {str(e)}")
        return False


async def test_hash_operations():
    """Test Redis hash operations"""
    print("\n🗂️ Testing Redis hash operations...")
    
    try:
        async with redis_operation() as cache:
            # Test hash operations
            hash_key = "test_hash"
            hash_data = {
                "field1": "value1",
                "field2": {"nested": "data"},
                "field3": [1, 2, 3, 4],
                "field4": 42
            }
            
            # Set hash
            set_result = await cache.set_hash(hash_key, hash_data)
            print(f"HSET result: {'✅' if set_result else '❌'}")
            
            # Get hash
            retrieved_hash = await cache.get_hash(hash_key)
            print(f"HGET result: {'✅' if retrieved_hash else '❌'}")
            print(f"Retrieved hash: {json.dumps(retrieved_hash, indent=2)}")
            
            return True
            
    except Exception as e:
        print(f"❌ Hash operations test failed: {str(e)}")
        return False


async def test_cache_with_prefixes():
    """Test cache operations with prefixes"""
    print("\n🏷️ Testing cache operations with prefixes...")
    
    try:
        async with redis_operation() as cache:
            # Test with different prefixes
            test_cases = [
                ("embedding", "model_123", {"vector": [0.1, 0.2, 0.3]}),
                ("query_result", "query_456", {"answer": "Test answer", "confidence": 0.95}),
                ("document_chunk", "doc_789", {"text": "Document content", "page": 1}),
                ("session", "sess_abc", {"user_id": "user123", "timestamp": str(datetime.now())})
            ]
            
            # Set values with prefixes
            for prefix, key, value in test_cases:
                set_result = await cache.set(key, value, prefix=prefix, ttl=600)
                print(f"SET {prefix}:{key} -> {'✅' if set_result else '❌'}")
            
            # Get values with prefixes
            for prefix, key, expected_value in test_cases:
                retrieved = await cache.get(key, prefix=prefix)
                matches = retrieved == expected_value
                print(f"GET {prefix}:{key} -> {'✅' if matches else '❌'}")
                if not matches:
                    print(f"  Expected: {expected_value}")
                    print(f"  Retrieved: {retrieved}")
            
            return True
            
    except Exception as e:
        print(f"❌ Prefix operations test failed: {str(e)}")
        return False


async def test_cache_statistics():
    """Test cache statistics functionality"""
    print("\n📈 Testing cache statistics...")
    
    try:
        async with redis_operation() as cache:
            # Get cache stats
            stats = await cache.get_cache_stats()
            
            print("📊 Cache Statistics:")
            print(f"  Redis Version: {stats.get('redis_version', 'N/A')}")
            print(f"  Used Memory: {stats.get('used_memory_human', 'N/A')}")
            print(f"  Connected Clients: {stats.get('connected_clients', 'N/A')}")
            print(f"  Total Commands: {stats.get('total_commands_processed', 'N/A')}")
            print(f"  Cache Hit Rate: {stats.get('cache_hit_rate', 0):.2f}%")
            
            # Connection stats
            conn_stats = stats.get('connection_stats', {})
            print(f"  Max Connections: {conn_stats.get('max_connections', 'N/A')}")
            print(f"  Is Healthy: {conn_stats.get('is_healthy', 'N/A')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Cache statistics test failed: {str(e)}")
        return False


async def test_error_handling():
    """Test Redis error handling"""
    print("\n🚨 Testing error handling...")
    
    try:
        async with redis_operation() as cache:
            # Test operations that might fail gracefully
            
            # Try to get non-existent key
            result = await cache.get("non_existent_key")
            print(f"Non-existent key: {'✅' if result is None else '❌'}")
            
            # Try to delete non-existent key
            delete_result = await cache.delete("non_existent_key")
            print(f"Delete non-existent: {'✅' if not delete_result else '❌'}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False


async def cleanup_test_data():
    """Clean up test data from Redis"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        async with redis_operation() as cache:
            # Clean up test keys
            test_patterns = [
                "test_*",
                "emb:model_*",
                "qr:query_*", 
                "doc:doc_*",
                "sess:sess_*"
            ]
            
            total_deleted = 0
            for pattern in test_patterns:
                deleted = await cache.clear_pattern(pattern)
                total_deleted += deleted
                
            print(f"🗑️ Cleaned up {total_deleted} test keys")
            return True
            
    except Exception as e:
        print(f"❌ Cleanup failed: {str(e)}")
        return False


async def main():
    """Run all Redis connectivity tests"""
    print("🚀 Starting Redis Connectivity Tests")
    print("=" * 50)
    
    test_results = []
    
    try:
        # Run all tests
        tests = [
            ("Basic Connectivity", test_basic_connectivity),
            ("Cache Operations", test_cache_operations), 
            ("Hash Operations", test_hash_operations),
            ("Prefix Operations", test_cache_with_prefixes),
            ("Cache Statistics", test_cache_statistics),
            ("Error Handling", test_error_handling),
            ("Cleanup", cleanup_test_data)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = await test_func()
                test_results.append((test_name, result))
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"\n{status}: {test_name}")
            except Exception as e:
                test_results.append((test_name, False))
                print(f"\n❌ FAILED: {test_name} - {str(e)}")
        
        # Final results
        print(f"\n{'='*50}")
        print("🏁 Test Results Summary:")
        print(f"{'='*50}")
        
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL" 
            print(f"  {status} {test_name}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 All tests passed! Redis is ready for the hybrid RAG system.")
            return True
        else:
            print(f"\n⚠️ {total-passed} test(s) failed. Please check Redis configuration.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {str(e)}")
        return False
        
    finally:
        # Shutdown Redis connections
        try:
            await shutdown_redis()
            print("\n🔌 Redis connections closed")
        except Exception as e:
            print(f"\n⚠️ Error closing Redis connections: {str(e)}")


if __name__ == "__main__":
    # Run the test suite
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)