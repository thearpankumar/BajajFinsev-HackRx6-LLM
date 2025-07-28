#!/usr/bin/env python3
"""
Comprehensive test script to verify authentication behavior.
This will help identify if the issue is with the API or Swagger UI.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_authentication_comprehensive():
    """Comprehensive authentication testing."""
    
    # Test data
    test_data = {
        "documents": ["https://example.com/document1.pdf"],
        "questions": ["What is this document about?"]
    }
    
    print("🔐 Comprehensive Authentication Testing")
    print("=" * 60)
    
    # Test 1: No token
    print("\n1️⃣ Testing with NO token:")
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", json=test_data)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        expected = "401" if response.status_code == 401 else "❌ UNEXPECTED"
        print(f"   Expected: 401, Got: {response.status_code} {expected}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Invalid token
    print("\n2️⃣ Testing with INVALID token:")
    try:
        headers = {"Authorization": "Bearer invalid_token"}
        response = requests.post(f"{BASE_URL}/hackrx/run", json=test_data, headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        expected = "401" if response.status_code == 401 else "❌ UNEXPECTED"
        print(f"   Expected: 401, Got: {response.status_code} {expected}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Empty token
    print("\n3️⃣ Testing with EMPTY token:")
    try:
        headers = {"Authorization": "Bearer "}
        response = requests.post(f"{BASE_URL}/hackrx/run", json=test_data, headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        expected = "401" if response.status_code == 401 else "❌ UNEXPECTED"
        print(f"   Expected: 401, Got: {response.status_code} {expected}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Valid token
    print("\n4️⃣ Testing with VALID token:")
    try:
        headers = {"Authorization": "Bearer 12345678901"}
        response = requests.post(f"{BASE_URL}/hackrx/run", json=test_data, headers=headers)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
            print(f"   Expected: 200, Got: {response.status_code} ✅")
        else:
            print(f"   Response: {response.text[:200]}...")
            print(f"   Expected: 200, Got: {response.status_code} ❌")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Health check (should work without auth)
    print("\n5️⃣ Testing health check (no auth required):")
    try:
        response = requests.get(f"{BASE_URL}/hackrx/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
            print(f"   Expected: 200, Got: {response.status_code} ✅")
        else:
            print(f"   Response: {response.text[:200]}...")
            print(f"   Expected: 200, Got: {response.status_code} ❌")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 6: API info (should work without auth)
    print("\n6️⃣ Testing API info (no auth required):")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
            print(f"   Expected: 200, Got: {response.status_code} ✅")
        else:
            print(f"   Response: {response.text[:200]}...")
            print(f"   Expected: 200, Got: {response.status_code} ❌")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Comprehensive authentication test completed!")
    print("\n📋 Summary:")
    print("   - Tests 1-3 should return 401 (Unauthorized)")
    print("   - Test 4 should return 200 (Success)")
    print("   - Tests 5-6 should return 200 (No auth required)")

def test_swagger_ui_endpoints():
    """Test Swagger UI specific endpoints."""
    print("\n🌐 Testing Swagger UI Endpoints")
    print("=" * 40)
    
    # Test OpenAPI schema
    print("\n1️⃣ Testing OpenAPI schema:")
    try:
        response = requests.get(f"{BASE_URL}/openapi.json")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            schema = response.json()
            if "securitySchemes" in schema.get("components", {}):
                print("   ✅ Security schemes found in OpenAPI schema")
            else:
                print("   ❌ Security schemes missing from OpenAPI schema")
        else:
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test docs endpoint
    print("\n2️⃣ Testing docs endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Swagger UI docs accessible")
        else:
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")

if _name_ == "_main_":
    print("🚀 Starting Authentication Tests...")
    print("Make sure your FastAPI server is running on localhost:8000")
    print("=" * 60)
    
    test_authentication_comprehensive()
    test_swagger_ui_endpoints()
    
    print("\n🎯 Next Steps:")
    print("1. Run this test to verify API authentication works correctly")
    print("2. If API tests pass but Swagger UI fails, the issue is with Swagger UI")
    print("3. Try clearing browser cache and cookies for Swagger UI")
    print("4. Check browser developer tools for any JavaScript errors")