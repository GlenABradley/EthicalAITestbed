#!/usr/bin/env python3
"""
Multi-Modal Evaluation System Test - Phase 4 Validation

Tests the comprehensive multi-modal evaluation system with pre/post/stream evaluation modes,
orchestration capabilities, and enterprise-grade features.

Channels the testing philosophy of legendary developers:
- Donald Knuth's algorithmic precision in test design
- Barbara Liskov's data abstraction verification
- Alan Kay's messaging system validation
"""

import requests
import json
import time
import asyncio

BASE_URL = "http://localhost:8001/api"

def test_multimodal_evaluation_system():
    """Test the complete multi-modal evaluation system."""
    print("🧪 Testing Multi-Modal Evaluation System - Phase 4")
    print("=" * 70)
    print("Channeling the expertise of Knuth, Liskov, and Kay...")
    print("=" * 70)
    
    # Test 1: System Capabilities
    print("\n1. Testing System Capabilities (Alan Kay's Message-Oriented Architecture)...")
    try:
        response = requests.get(f"{BASE_URL}/multimodal/capabilities", timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Capabilities Retrieved Successfully")
            print(f"   Available Modes: {len(data.get('available_modes', []))}")
            print(f"   Supported Features: {len(data.get('features', []))}")
            print(f"   API Version: {data.get('api_version', 'N/A')}")
            
            # List available modes
            for mode in data.get('available_modes', []):
                print(f"   📋 {mode.get('mode', 'Unknown')}: {mode.get('description', 'No description')}")
        else:
            print(f"❌ Capabilities Test Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Capabilities Test Error: {e}")
        return False
    
    # Test 2: Pre-Evaluation Mode (Knuth's Algorithmic Precision)
    print("\n2. Testing Pre-Evaluation Mode (Input Analysis & Safety Screening)...")
    try:
        test_content = "I would like to learn about machine learning ethics and how to implement fair AI systems."
        
        response = requests.post(f"{BASE_URL}/multimodal/evaluate", json={
            "content": test_content,
            "mode": "pre_evaluation",
            "priority": "medium",
            "user_id": "test_user_001",
            "session_id": "session_001"
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pre-Evaluation Success")
            print(f"   Overall Score: {data.get('overall_ethical_score', 'N/A'):.3f}")
            print(f"   Decision: {data.get('decision', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A'):.3f}")
            
            # Check pre-evaluation specific results
            pre_eval = data.get('pre_evaluation', {})
            if pre_eval:
                print(f"   Should Proceed: {pre_eval.get('should_proceed', 'N/A')}")
                print(f"   Intent Analysis: {len(pre_eval.get('intent_analysis', {}))} factors")
                print(f"   Risk Factors: {len(pre_eval.get('risk_factors', []))} identified")
                print(f"   Processing Recommendations: {len(pre_eval.get('processing_recommendations', []))}")
            
        else:
            print(f"❌ Pre-Evaluation Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Pre-Evaluation Error: {e}")
    
    # Test 3: Post-Evaluation Mode (Liskov's Data Abstraction)
    print("\n3. Testing Post-Evaluation Mode (Output Validation & Alignment)...")
    try:
        test_output = "Machine learning ethics involves ensuring AI systems are fair, transparent, and do not harm people. Key principles include avoiding bias, protecting privacy, and maintaining human oversight."
        original_input = "Explain machine learning ethics"
        
        response = requests.post(f"{BASE_URL}/multimodal/evaluate", json={
            "content": test_output,
            "mode": "post_evaluation", 
            "priority": "high",
            "original_input": original_input,
            "intended_purpose": "educational explanation",
            "session_id": "session_001"
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Post-Evaluation Success")
            print(f"   Overall Score: {data.get('overall_ethical_score', 'N/A'):.3f}")
            print(f"   Decision: {data.get('decision', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A'):.3f}")
            
            # Check post-evaluation specific results
            post_eval = data.get('post_evaluation', {})
            if post_eval:
                print(f"   Output Approved: {post_eval.get('output_approved', 'N/A')}")
                print(f"   Alignment Score: {post_eval.get('alignment_score', 'N/A'):.3f}")
                print(f"   Human Review Required: {post_eval.get('human_review_required', 'N/A')}")
                print(f"   Improvement Suggestions: {len(post_eval.get('improvement_suggestions', []))}")
            
        else:
            print(f"❌ Post-Evaluation Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Post-Evaluation Error: {e}")
    
    # Test 4: Batch Evaluation (System Scalability)
    print("\n4. Testing Batch Evaluation (Concurrent Processing)...")
    try:
        test_batch = [
            "This is helpful educational content",
            "Machine learning can benefit society",
            "Artificial intelligence should be used responsibly", 
            "Ethical AI development is important"
        ]
        
        response = requests.post(f"{BASE_URL}/multimodal/batch-evaluate", json={
            "content_items": test_batch,
            "mode": "pre_evaluation",
            "priority": "batch",
            "batch_metadata": {"test_batch": True}
        }, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch Evaluation Success")
            
            batch_stats = data.get('batch_stats', {})
            print(f"   Total Items: {batch_stats.get('total_items', 'N/A')}")
            print(f"   Successful: {batch_stats.get('successful_evaluations', 'N/A')}")
            print(f"   Success Rate: {batch_stats.get('success_rate', 0):.1%}")
            print(f"   Avg Processing Time: {batch_stats.get('average_processing_time', 'N/A'):.3f}s")
            print(f"   Avg Ethical Score: {batch_stats.get('average_ethical_score', 'N/A'):.3f}")
            
        else:
            print(f"❌ Batch Evaluation Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Batch Evaluation Error: {e}")
    
    # Test 5: System Metrics and Monitoring
    print("\n5. Testing System Metrics (Performance Monitoring)...")
    try:
        response = requests.get(f"{BASE_URL}/multimodal/metrics", timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Metrics Retrieved Successfully")
            print(f"   Active Requests: {data.get('active_requests', 'N/A')}")
            print(f"   Total Processed: {data.get('total_requests_processed', 'N/A')}")
            
            # Mode performance
            mode_perf = data.get('mode_performance', {})
            for mode_name, perf in mode_perf.items():
                print(f"   📊 {mode_name}:")
                print(f"      Success Rate: {perf.get('success_rate', 0):.1%}")
                print(f"      Avg Processing: {perf.get('average_processing_time', 0):.3f}s")
                
        else:
            print(f"❌ Metrics Test Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Metrics Test Error: {e}")
    
    # Test 6: Health Check (System Resilience)
    print("\n6. Testing System Health Check (Circuit Breaker & Resilience)...")
    try:
        response = requests.post(f"{BASE_URL}/multimodal/health-check", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check Completed")
            print(f"   Overall Status: {data.get('overall_status', 'N/A')}")
            print(f"   Health Check Duration: {data.get('health_check_duration', 'N/A'):.3f}s")
            
            # Mode health
            mode_health = data.get('mode_health', {})
            for mode_name, health in mode_health.items():
                status = health.get('status', 'unknown')
                emoji = "✅" if status == "healthy" else "❌"
                print(f"   {emoji} {mode_name}: {status}")
                if 'response_time' in health:
                    print(f"      Response Time: {health['response_time']:.3f}s")
            
            # Circuit breaker status
            cb_status = data.get('circuit_breaker_status', {})
            healthy_cbs = sum(1 for cb in cb_status.values() if cb.get('healthy', False))
            print(f"   🔧 Circuit Breakers: {healthy_cbs}/{len(cb_status)} healthy")
            
            # Recommendations
            recommendations = data.get('recommendations', [])
            if recommendations:
                print(f"   💡 Recommendations: {len(recommendations)}")
                for rec in recommendations[:3]:  # Show first 3
                    print(f"      • {rec}")
            
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
    
    # Test 7: Mode Configuration (Dynamic Adjustment)
    print("\n7. Testing Mode Configuration (Dynamic Tuning)...")
    try:
        config = {
            "risk_thresholds": {
                "manipulation_risk": 0.25,
                "bias_risk": 0.35
            },
            "safety_boundaries": {
                "content_length": {"max": 60000}
            }
        }
        
        response = requests.post(f"{BASE_URL}/multimodal/configure-mode", json={
            "mode": "pre_evaluation",
            "configuration": config
        }, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Mode Configuration Success")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Mode: {data.get('mode', 'N/A')}")
            print(f"   Configuration Applied: ✅")
            
        else:
            print(f"❌ Configuration Failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Multi-Modal Evaluation System Testing Complete!")
    print("   Following the architectural excellence of:")
    print("   • Donald Knuth - Algorithmic precision and correctness")
    print("   • Barbara Liskov - Data abstraction and interface design")
    print("   • Alan Kay - Message-oriented system architecture")
    print("=" * 70)

if __name__ == "__main__":
    # First check if the API is healthy
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Backend is healthy - starting comprehensive Phase 4 tests")
            test_multimodal_evaluation_system()
        else:
            print("❌ Backend health check failed")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")