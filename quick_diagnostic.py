#!/usr/bin/env python3
"""
🔍 QUICK BAYESIAN OPTIMIZATION DIAGNOSTIC TEST
═══════════════════════════════════════════════════════════════════════════════════

MISSION: Quickly diagnose if the Bayesian optimization endpoints are still timing out
with very short timeouts to avoid hanging tests.

Author: Testing Agent
Version: 1.0.0
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

# Backend URL from environment
BACKEND_URL = "https://efb05ca6-d049-4715-907b-1090362ca79b.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

async def quick_diagnostic():
    """Quick diagnostic test with short timeouts."""
    print("🔍 QUICK BAYESIAN OPTIMIZATION DIAGNOSTIC")
    print("=" * 50)
    print("Testing with 10-second timeout to avoid hanging...")
    
    # Very short timeout to avoid hanging
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        # Test 1: Health endpoint (should work)
        print("\n1️⃣ Testing Health Endpoint (baseline)...")
        try:
            start_time = time.time()
            async with session.get(f"{API_BASE}/health") as response:
                response_time = time.time() - start_time
                result = await response.json()
                print(f"   ✅ Health: {response_time:.3f}s | Status: {result.get('status', 'unknown')}")
        except Exception as e:
            response_time = time.time() - start_time
            print(f"   ❌ Health: {response_time:.3f}s | Error: {str(e)}")
        
        # Test 2: Optimization Start (the problematic endpoint)
        print("\n2️⃣ Testing Optimization Start (10s timeout)...")
        test_data = {
            "test_texts": [
                "Quick test for lightweight Bayesian cluster optimization.",
                "This should complete much faster than the previous system."
            ],
            "n_initial_samples": 3,
            "n_optimization_iterations": 3,
            "max_optimization_time": 15.0
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{API_BASE}/optimization/start", json=test_data) as response:
                response_time = time.time() - start_time
                if response.content_type == 'application/json':
                    result = await response.json()
                    print(f"   ✅ Start: {response_time:.3f}s | Status: {result.get('status', 'unknown')}")
                    optimization_id = result.get('optimization_id', '')
                    if optimization_id:
                        print(f"   📝 Optimization ID: {optimization_id[:12]}...")
                        
                        # Test 3: Status endpoint with the real ID
                        print("\n3️⃣ Testing Status Endpoint...")
                        try:
                            start_time = time.time()
                            async with session.get(f"{API_BASE}/optimization/status/{optimization_id}") as status_response:
                                status_time = time.time() - start_time
                                if status_response.content_type == 'application/json':
                                    status_result = await status_response.json()
                                    print(f"   ✅ Status: {status_time:.3f}s | Status: {status_result.get('status', 'unknown')}")
                                else:
                                    print(f"   ❌ Status: {status_time:.3f}s | Non-JSON response")
                        except Exception as e:
                            status_time = time.time() - start_time
                            print(f"   ❌ Status: {status_time:.3f}s | Error: {str(e)}")
                else:
                    print(f"   ❌ Start: {response_time:.3f}s | Non-JSON response")
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            print(f"   ❌ Start: {response_time:.3f}s | TIMEOUT (10s limit exceeded)")
        except Exception as e:
            response_time = time.time() - start_time
            print(f"   ❌ Start: {response_time:.3f}s | Error: {str(e)}")
        
        # Test 4: Status endpoint with fake ID (should be fast)
        print("\n4️⃣ Testing Status with Fake ID (should be fast)...")
        try:
            start_time = time.time()
            async with session.get(f"{API_BASE}/optimization/status/fake_id_12345") as response:
                response_time = time.time() - start_time
                print(f"   ✅ Fake Status: {response_time:.3f}s | HTTP {response.status} (expected 404)")
        except Exception as e:
            response_time = time.time() - start_time
            print(f"   ❌ Fake Status: {response_time:.3f}s | Error: {str(e)}")
        
        # Test 5: List endpoint (should be fast)
        print("\n5️⃣ Testing List Endpoint (should be fast)...")
        try:
            start_time = time.time()
            async with session.get(f"{API_BASE}/optimization/list") as response:
                response_time = time.time() - start_time
                if response.content_type == 'application/json':
                    result = await response.json()
                    optimizations = result.get('optimizations', [])
                    print(f"   ✅ List: {response_time:.3f}s | Found {len(optimizations)} optimizations")
                else:
                    print(f"   ❌ List: {response_time:.3f}s | Non-JSON response")
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            print(f"   ❌ List: {response_time:.3f}s | TIMEOUT (10s limit exceeded)")
        except Exception as e:
            response_time = time.time() - start_time
            print(f"   ❌ List: {response_time:.3f}s | Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC SUMMARY:")
    print("If optimization endpoints timeout but health works,")
    print("the issue is in the Bayesian optimization module.")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(quick_diagnostic())