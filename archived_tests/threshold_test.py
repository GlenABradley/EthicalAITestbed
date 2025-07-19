#!/usr/bin/env python3
"""
Quick test to check ethical evaluation with different thresholds
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_with_lower_thresholds():
    """Test ethical evaluation with lower thresholds"""
    
    # First, set lower thresholds
    lower_params = {
        "virtue_threshold": 0.15,  # Lower from 0.4
        "deontological_threshold": 0.15,  # Lower from 0.35
        "consequentialist_threshold": 0.15  # Lower from 0.45
    }
    
    print("Setting lower thresholds...")
    response = requests.post(
        f"{API_BASE}/update-parameters",
        json={"parameters": lower_params},
        timeout=10
    )
    
    if response.status_code == 200:
        print("✅ Thresholds updated successfully")
        
        # Test problematic text again
        test_text = "You are stupid and worthless"
        
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": test_text},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            evaluation = data.get('evaluation', {})
            
            print(f"Text: '{test_text}'")
            print(f"Overall ethical: {evaluation.get('overall_ethical', 'unknown')}")
            print(f"Minimal violations: {len(evaluation.get('minimal_spans', []))}")
            
            # Show some span scores
            spans = evaluation.get('spans', [])
            for span in spans[:5]:  # Show first 5 spans
                if span.get('any_violation', False):
                    print(f"  VIOLATION: '{span['text']}' - virtue:{span['virtue_score']:.3f}, deont:{span['deontological_score']:.3f}, conseq:{span['consequentialist_score']:.3f}")
                    
        else:
            print(f"❌ Evaluation failed: HTTP {response.status_code}")
    else:
        print(f"❌ Parameter update failed: HTTP {response.status_code}")

if __name__ == "__main__":
    test_with_lower_thresholds()