#!/usr/bin/env python3
"""
Smart Buffer Streaming Test - Phase 3 Validation

Tests the smart buffer streaming system for real-time ML training data analysis.
"""

import requests
import json
import time
import asyncio

BASE_URL = "http://localhost:8001/api"

def test_smart_buffer_streaming():
    """Test the smart buffer streaming system."""
    print("🧪 Testing Smart Buffer Streaming System - Phase 3")
    print("=" * 60)
    
    # Test 1: Configure Stream Buffer
    print("\n1. Testing Stream Buffer Configuration...")
    try:
        config_response = requests.post(f"{BASE_URL}/ml/stream/configure", json={
            "max_tokens": 100,
            "max_time_seconds": 2.0,
            "semantic_threshold": 0.7,
            "performance_threshold_ms": 50.0,
            "pattern_detection": True
        }, timeout=10)
        
        if config_response.status_code == 200:
            data = config_response.json()
            print(f"✅ Stream Buffer Configuration Success")
            print(f"   Status: {data.get('status')}")
            print(f"   Buffer State: {data.get('buffer_state')}")
        else:
            print(f"❌ Configuration Failed: {config_response.status_code}")
            print(f"   Error: {config_response.text}")
            return
            
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return
    
    # Test 2: Stream Tokens (Simple)
    print("\n2. Testing Token Streaming - Simple Case...")
    try:
        # Send a small batch of tokens
        token_response = requests.post(f"{BASE_URL}/ml/stream/tokens", json={
            "tokens": ["Hello", "world", "this", "is"],
            "session_id": "test_session_1",
            "training_step": 1,
            "batch_id": "batch_001"
        }, timeout=10)
        
        if token_response.status_code == 200:
            data = token_response.json()
            print(f"✅ Token Streaming Success")
            print(f"   Analysis Triggered: {data.get('analysis_triggered')}")
            if data.get('buffer_metrics'):
                metrics = data['buffer_metrics']
                print(f"   Buffer Utilization: {metrics.get('buffer_utilization', 0):.2f}")
        else:
            print(f"❌ Token Streaming Failed: {token_response.status_code}")
            
    except Exception as e:
        print(f"❌ Token Streaming Error: {e}")
    
    # Test 3: Stream More Tokens to Trigger Analysis
    print("\n3. Testing Token Streaming - Trigger Analysis...")
    try:
        # Send enough tokens to trigger analysis
        large_token_batch = ["a", "helpful", "and", "respectful", "message", "that", "should", "be", "analyzed", "for", "ethical", "content"] * 10
        
        token_response = requests.post(f"{BASE_URL}/ml/stream/tokens", json={
            "tokens": large_token_batch,
            "session_id": "test_session_1",
            "training_step": 2
        }, timeout=30)
        
        if token_response.status_code == 200:
            data = token_response.json()
            print(f"✅ Analysis Triggered: {data.get('analysis_triggered')}")
            if data.get('analysis_triggered'):
                print(f"   Ethical Score: {data.get('ethical_score', 'N/A')}")
                print(f"   Risk Level: {data.get('risk_level', 'N/A')}")
                print(f"   Intervention Required: {data.get('intervention_required', 'N/A')}")
                if data.get('recommendations'):
                    print(f"   Recommendations: {len(data['recommendations'])} items")
        else:
            print(f"❌ Analysis Trigger Failed: {token_response.status_code}")
            
    except Exception as e:
        print(f"❌ Analysis Trigger Error: {e}")
    
    # Test 4: Get Stream Metrics
    print("\n4. Testing Stream Metrics...")
    try:
        metrics_response = requests.get(f"{BASE_URL}/ml/stream/metrics", timeout=10)
        
        if metrics_response.status_code == 200:
            data = metrics_response.json()
            print(f"✅ Stream Metrics Success")
            print(f"   Status: {data.get('status')}")
            print(f"   Buffer State: {data.get('buffer_state')}")
            if data.get('metrics'):
                metrics = data['metrics']
                print(f"   Tokens Processed: {metrics.get('tokens_processed', 0)}")
                print(f"   Evaluations Completed: {metrics.get('evaluations_completed', 0)}")
                print(f"   Interventions Triggered: {metrics.get('interventions_triggered', 0)}")
        else:
            print(f"❌ Metrics Failed: {metrics_response.status_code}")
            
    except Exception as e:
        print(f"❌ Metrics Error: {e}")
    
    # Test 5: Force Flush Buffer
    print("\n5. Testing Buffer Flush...")
    try:
        # Add some tokens first
        requests.post(f"{BASE_URL}/ml/stream/tokens", json={
            "tokens": ["final", "test", "tokens"],
            "session_id": "test_session_1"
        }, timeout=10)
        
        # Then flush
        flush_response = requests.post(f"{BASE_URL}/ml/stream/flush", timeout=30)
        
        if flush_response.status_code == 200:
            data = flush_response.json()
            print(f"✅ Buffer Flush Success")
            print(f"   Status: {data.get('status')}")
            if data.get('analysis'):
                analysis = data['analysis']
                print(f"   Ethical Score: {analysis.get('ethical_score', 'N/A')}")
                print(f"   Risk Level: {analysis.get('risk_level', 'N/A')}")
        else:
            print(f"❌ Buffer Flush Failed: {flush_response.status_code}")
            
    except Exception as e:
        print(f"❌ Buffer Flush Error: {e}")
    
    # Test 6: Reset Buffer
    print("\n6. Testing Buffer Reset...")
    try:
        reset_response = requests.post(f"{BASE_URL}/ml/stream/reset", timeout=10)
        
        if reset_response.status_code == 200:
            data = reset_response.json()
            print(f"✅ Buffer Reset Success")
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}")
        else:
            print(f"❌ Buffer Reset Failed: {reset_response.status_code}")
            
    except Exception as e:
        print(f"❌ Buffer Reset Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Smart Buffer Streaming Test Complete!")

if __name__ == "__main__":
    # First check if the API is healthy
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Backend is healthy - starting tests")
            test_smart_buffer_streaming()
        else:
            print("❌ Backend health check failed")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")