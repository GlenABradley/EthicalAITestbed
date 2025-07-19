#!/usr/bin/env python3
"""
Focused test for new dynamic scaling and learning features
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_threshold_scaling():
    """Test threshold scaling endpoints"""
    print("🧪 Testing Threshold Scaling...")
    
    # Test exponential scaling
    response = requests.post(
        f"{API_BASE}/threshold-scaling",
        json={"slider_value": 0.5, "use_exponential": True},
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Exponential scaling: {data['slider_value']} -> {data['scaled_threshold']:.4f}")
    else:
        print(f"❌ Exponential scaling failed: HTTP {response.status_code}")
    
    # Test linear scaling
    response = requests.post(
        f"{API_BASE}/threshold-scaling",
        json={"slider_value": 0.5, "use_exponential": False},
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Linear scaling: {data['slider_value']} -> {data['scaled_threshold']:.4f}")
    else:
        print(f"❌ Linear scaling failed: HTTP {response.status_code}")

def test_learning_stats():
    """Test learning stats endpoint"""
    print("\n🧪 Testing Learning Stats...")
    
    response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Learning stats: {data['total_learning_entries']} entries, avg feedback: {data['average_feedback_score']:.3f}")
        return True
    else:
        print(f"❌ Learning stats failed: HTTP {response.status_code}")
        if response.status_code == 500:
            print(f"   Error: {response.text}")
        return False

def test_dynamic_scaling_evaluation():
    """Test evaluation with dynamic scaling enabled"""
    print("\n🧪 Testing Dynamic Scaling Evaluation...")
    
    # Enable dynamic scaling
    param_response = requests.post(
        f"{API_BASE}/update-parameters",
        json={"parameters": {
            "enable_dynamic_scaling": True,
            "enable_cascade_filtering": True,
            "enable_learning_mode": True
        }},
        timeout=10
    )
    
    if param_response.status_code != 200:
        print(f"❌ Failed to enable dynamic scaling: HTTP {param_response.status_code}")
        return None
    
    # Test evaluation
    response = requests.post(
        f"{API_BASE}/evaluate",
        json={"text": "This is a test for dynamic scaling"},
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        evaluation = data.get('evaluation', {})
        dynamic_info = evaluation.get('dynamic_scaling', {})
        
        print(f"✅ Dynamic scaling evaluation completed")
        print(f"   Used dynamic scaling: {dynamic_info.get('used_dynamic_scaling', False)}")
        print(f"   Used cascade filtering: {dynamic_info.get('used_cascade_filtering', False)}")
        print(f"   Ambiguity score: {dynamic_info.get('ambiguity_score', 0.0):.3f}")
        print(f"   Processing stages: {dynamic_info.get('processing_stages', [])}")
        
        return evaluation.get('evaluation_id')
    else:
        print(f"❌ Dynamic scaling evaluation failed: HTTP {response.status_code}")
        return None

def test_feedback_submission(evaluation_id):
    """Test feedback submission"""
    print("\n🧪 Testing Feedback Submission...")
    
    if not evaluation_id:
        print("❌ No evaluation_id provided")
        return False
    
    response = requests.post(
        f"{API_BASE}/feedback",
        json={
            "evaluation_id": evaluation_id,
            "feedback_score": 0.8,
            "user_comment": "Good result"
        },
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Feedback submitted: {data.get('message', 'Success')}")
        return True
    else:
        print(f"❌ Feedback submission failed: HTTP {response.status_code}")
        if response.status_code == 500:
            print(f"   Error: {response.text}")
        return False

def test_empty_text_handling():
    """Test empty text handling"""
    print("\n🧪 Testing Empty Text Handling...")
    
    test_cases = ["", "   ", "\n\t"]
    
    for i, text in enumerate(test_cases):
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": text},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Empty text case {i+1} handled correctly")
        else:
            print(f"❌ Empty text case {i+1} failed: HTTP {response.status_code}")

def main():
    """Run focused tests for new features"""
    print(f"🚀 Testing new dynamic scaling and learning features")
    print(f"Backend URL: {API_BASE}")
    print("=" * 60)
    
    # Test individual components
    test_threshold_scaling()
    learning_working = test_learning_stats()
    evaluation_id = test_dynamic_scaling_evaluation()
    
    if learning_working and evaluation_id:
        test_feedback_submission(evaluation_id)
    
    test_empty_text_handling()
    
    print("\n" + "=" * 60)
    print("🏁 Focused testing complete")

if __name__ == "__main__":
    main()