#!/usr/bin/env python3
"""
🎯 SIMPLE BAYESIAN OPTIMIZATION STATUS CHECK
═══════════════════════════════════════════════════════════════════════════════════

Quick test to check if the performance optimizations have been applied.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

# Backend URL from environment
BACKEND_URL = "https://efb05ca6-d049-4715-907b-1090362ca79b.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

async def quick_bayesian_test():
    """Quick test of Bayesian optimization endpoints."""
    
    print("🎯 QUICK BAYESIAN OPTIMIZATION STATUS CHECK")
    print("=" * 60)
    
    timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        # Test 1: Quick optimization start with minimal parameters
        print("🚀 Test 1: Quick optimization start (10s timeout)...")
        
        test_data = {
            "test_texts": [
                "Quick test for optimized Bayesian cluster resolution.",
                "Performance improvements should make this much faster."
            ],
            "n_initial_samples": 3,
            "n_optimization_iterations": 5,
            "max_optimization_time": 20.0,
            "parallel_evaluations": False,
            "max_workers": 1
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{API_BASE}/optimization/start", json=test_data) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    print(f"    ✅ SUCCESS: {response_time:.3f}s")
                    print(f"    Optimization ID: {result.get('optimization_id', 'N/A')[:12]}...")
                    print(f"    Status: {result.get('status', 'unknown')}")
                    optimization_id = result.get('optimization_id', '')
                else:
                    result = await response.json()
                    print(f"    ❌ FAILED: {response_time:.3f}s | Status: {response.status}")
                    print(f"    Error: {result.get('error', 'Unknown error')}")
                    optimization_id = None
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            print(f"    ❌ TIMEOUT: {response_time:.3f}s - Still experiencing performance issues")
            optimization_id = None
        except Exception as e:
            response_time = time.time() - start_time
            print(f"    ❌ ERROR: {response_time:.3f}s - {str(e)}")
            optimization_id = None
        
        # Test 2: Status monitoring
        print("\n📊 Test 2: Status monitoring (should be fast)...")
        
        fake_id = "opt_fake_test_12345"
        start_time = time.time()
        try:
            async with session.get(f"{API_BASE}/optimization/status/{fake_id}") as response:
                response_time = time.time() - start_time
                
                if response.status == 404:
                    print(f"    ✅ SUCCESS: {response_time:.3f}s - Proper 404 handling")
                elif response.status == 200:
                    result = await response.json()
                    print(f"    ⚠️ UNEXPECTED: {response_time:.3f}s - Got 200 for fake ID")
                else:
                    print(f"    ❌ FAILED: {response_time:.3f}s | Status: {response.status}")
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            print(f"    ❌ TIMEOUT: {response_time:.3f}s - Status endpoint timing out")
        except Exception as e:
            response_time = time.time() - start_time
            print(f"    ❌ ERROR: {response_time:.3f}s - {str(e)}")
        
        # Test 3: List optimizations
        print("\n📋 Test 3: List optimizations...")
        
        start_time = time.time()
        try:
            async with session.get(f"{API_BASE}/optimization/list") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    print(f"    ✅ SUCCESS: {response_time:.3f}s")
                    print(f"    Found optimizations: {len(result.get('optimizations', []))}")
                else:
                    result = await response.json()
                    print(f"    ❌ FAILED: {response_time:.3f}s | Status: {response.status}")
                    print(f"    Error: {result.get('error', 'Unknown error')}")
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            print(f"    ❌ TIMEOUT: {response_time:.3f}s - List endpoint timing out")
        except Exception as e:
            response_time = time.time() - start_time
            print(f"    ❌ ERROR: {response_time:.3f}s - {str(e)}")
    
    print("\n" + "=" * 60)
    print("🏁 QUICK TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(quick_bayesian_test())