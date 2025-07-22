#!/usr/bin/env python3
"""
ML Ethics API Test Script - Phase 2 Validation

Tests the enhanced ML Ethics API endpoints with the advanced vector engine.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001/api"

def test_ml_ethics_api():
    """Test all ML Ethics API endpoints."""
    print("🧪 Testing Enhanced ML Ethics API - Phase 2")
    print("=" * 50)
    
    # Test data
    test_training_data = [
        "This is helpful and respectful content",
        "Machine learning can benefit humanity when used ethically"
    ]
    
    # Test 1: Training Data Evaluation
    print("\n1. Testing Training Data Evaluation...")
    try:
        response = requests.post(f"{BASE_URL}/ml/training-data-eval", json={
            "training_data": test_training_data,
            "dataset_type": "curated",
            "training_phase": "initial"
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training Data Evaluation Success")
            print(f"   Dataset Analysis: {data.get('dataset_analysis', {})}")
        else:
            print(f"❌ Training Data Evaluation Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Training Data Evaluation Error: {e}")
    
    # Test 2: Ethical Vector Generation
    print("\n2. Testing Enhanced Ethical Vector Generation...")
    try:
        response = requests.post(f"{BASE_URL}/ml/ethical-vectors", json={
            "training_data": test_training_data,
            "dataset_type": "curated",
            "training_phase": "initial"
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ethical Vector Generation Success")
            vectors = data.get('ethical_vectors', {})
            print(f"   Autonomy Vectors: {len(vectors.get('autonomy_vectors', []))} dimensions")
            print(f"   Risk Assessment: {data.get('risk_assessment', {})}")
        else:
            print(f"❌ Ethical Vector Generation Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Ethical Vector Generation Error: {e}")
    
    # Test 3: Training Guidance
    print("\n3. Testing Enhanced Training Guidance...")
    try:
        response = requests.post(f"{BASE_URL}/ml/training-guidance", json={
            "batch_data": test_training_data,
            "training_step": 100,
            "loss_value": 0.5
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training Guidance Success")
            print(f"   Continue Training: {data.get('continue_training', 'N/A')}")
            print(f"   Ethical Score: {data.get('ethical_score', 'N/A'):.3f}")
            print(f"   Warnings: {len(data.get('warnings', []))} warnings")
        else:
            print(f"❌ Training Guidance Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Training Guidance Error: {e}")
    
    # Test 4: Advanced Analysis
    print("\n4. Testing Advanced ML Ethical Analysis...")
    try:
        response = requests.post(f"{BASE_URL}/ml/advanced-analysis", json={
            "training_data": test_training_data,
            "dataset_type": "curated",
            "training_phase": "initial",
            "analysis_depth": "standard"
        }, timeout=45)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Advanced Analysis Success")
            print(f"   Dataset Overview: {data.get('dataset_overview', {})}")
            print(f"   Risk Assessment: {data.get('risk_assessment', {})}")
        else:
            print(f"❌ Advanced Analysis Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Advanced Analysis Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 ML Ethics API Phase 2 Testing Complete!")

if __name__ == "__main__":
    test_ml_ethics_api()