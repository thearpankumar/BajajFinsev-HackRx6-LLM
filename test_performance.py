#!/usr/bin/env python3
"""
Performance test script for the BajajFinsev RAG System
Tests the 30-second requirement with payload1.json
"""

import asyncio
import aiohttp
import json
import time
import sys
import argparse

async def test_performance(base_url="http://localhost:8000", auth_token="589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68"):
    """Test the performance of the RAG system with payload1.json"""
    
    # Load the test payload
    try:
        with open("payloads/payload1.json", "r") as f:
            payload = json.load(f)
    except FileNotFoundError:
        print("❌ Error: payloads/payload1.json not found")
        return False
    
    print("🚀 Starting Performance Test")
    print(f"Document URL: {payload['documents']}")
    print(f"Number of questions: {len(payload['questions'])}")
    print("=" * 80)
    
    # Expected answers for accuracy comparison
    expected_answers = [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    ]
    
    # Headers for API request
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"⏱️  Starting request at {time.strftime('%H:%M:%S')}")
            
            async with session.post(
                f"{base_url}/api/v1/hackrx/run",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
            ) as response:
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"⏱️  Response received at {time.strftime('%H:%M:%S')}")
                print(f"🕐 Total processing time: {processing_time:.2f} seconds")
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Check if we got answers
                    if "answers" in result:
                        actual_answers = result["answers"]
                        print(f"✅ Received {len(actual_answers)} answers")
                        
                        # Performance check
                        if processing_time <= 30:
                            print(f"🎯 ✅ PERFORMANCE TARGET MET: {processing_time:.2f}s <= 30s")
                        else:
                            print(f"❌ PERFORMANCE TARGET MISSED: {processing_time:.2f}s > 30s")
                        
                        # Accuracy check
                        print("\n" + "=" * 80)
                        print("📊 ACCURACY ANALYSIS")
                        print("=" * 80)
                        
                        accuracy_scores = []
                        for i, (actual, expected) in enumerate(zip(actual_answers, expected_answers)):
                            print(f"\nQuestion {i+1}: {payload['questions'][i]}")
                            print(f"Expected: {expected}")
                            print(f"Actual:   {actual}")
                            
                            # Simple accuracy check based on key terms
                            accuracy = calculate_answer_accuracy(actual, expected)
                            accuracy_scores.append(accuracy)
                            print(f"Accuracy: {accuracy:.1%}")
                            print("-" * 40)
                        
                        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                        print(f"\n📈 OVERALL ACCURACY: {avg_accuracy:.1%}")
                        
                        if avg_accuracy >= 0.95:
                            print("🎯 ✅ ACCURACY TARGET MET: >= 95%")
                        else:
                            print("❌ ACCURACY TARGET MISSED: < 95%")
                        
                        # Final verdict
                        print("\n" + "=" * 80)
                        print("🏆 FINAL RESULTS")
                        print("=" * 80)
                        print(f"⏱️  Processing Time: {processing_time:.2f}s (Target: ≤30s)")
                        print(f"📊 Accuracy: {avg_accuracy:.1%} (Target: ≥95%)")
                        
                        if processing_time <= 30 and avg_accuracy >= 0.95:
                            print("🎉 ✅ ALL TARGETS ACHIEVED!")
                            return True
                        else:
                            print("❌ TARGETS NOT FULLY ACHIEVED")
                            return False
                        
                    else:
                        print("❌ Error: No answers in response")
                        print(f"Response: {result}")
                        return False
                        
                else:
                    error_text = await response.text()
                    print(f"❌ Error: HTTP {response.status}")
                    print(f"Response: {error_text}")
                    return False
                    
    except asyncio.TimeoutError:
        print("❌ Error: Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def calculate_answer_accuracy(actual, expected):
    """Calculate accuracy score between actual and expected answers"""
    
    # Convert to lowercase for comparison
    actual_lower = actual.lower()
    expected_lower = expected.lower()
    
    # Extract key terms from expected answer
    import re
    
    # Extract numbers, percentages, and key terms
    expected_numbers = re.findall(r'\d+(?:\.\d+)?', expected_lower)
    expected_percentages = re.findall(r'\d+(?:\.\d+)?%', expected_lower)
    
    # Key domain terms
    key_terms = [
        'grace period', 'waiting period', 'thirty days', 'thirty-six', '36', 'months',
        'maternity', 'pre-existing', 'coverage', 'benefit', 'cataract', 'organ donor',
        'no claim discount', 'health check', 'ayush', 'hospital', 'room rent', 'icu'
    ]
    
    # Calculate score based on presence of key elements
    score = 0.0
    total_checks = 0
    
    # Check numbers
    for num in expected_numbers:
        total_checks += 1
        if num in actual_lower:
            score += 1
    
    # Check percentages
    for pct in expected_percentages:
        total_checks += 1
        if pct in actual_lower:
            score += 1
    
    # Check key terms
    for term in key_terms:
        if term in expected_lower:
            total_checks += 1
            if term in actual_lower:
                score += 1
    
    # Overall semantic similarity (simplified)
    expected_words = set(expected_lower.split())
    actual_words = set(actual_lower.split())
    
    if expected_words:
        word_overlap = len(expected_words.intersection(actual_words)) / len(expected_words)
        score += word_overlap * 3  # Weight word overlap
        total_checks += 3
    
    return score / max(total_checks, 1)

async def check_server_health(base_url="http://localhost:8000"):
    """Check if the server is running and healthy"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/v1/hackrx/health", timeout=5) as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Server is healthy: {health_data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ Server health check failed: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {str(e)}")
        return False

async def main():
    parser = argparse.ArgumentParser(description='Test BajajFinsev RAG System Performance')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL of the API')
    parser.add_argument('--token', default='589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68', help='Auth token')
    
    args = parser.parse_args()
    
    print("🔍 Checking server health...")
    if not await check_server_health(args.url):
        print("❌ Server is not accessible. Please start the server first.")
        return False
    
    print("🧪 Running performance test...")
    result = await test_performance(args.url, args.token)
    
    if result:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())