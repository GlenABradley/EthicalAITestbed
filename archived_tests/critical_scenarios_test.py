#!/usr/bin/env python3
"""
FOCUSED CRITICAL USER SCENARIOS TEST
Tests the specific user examples at various thresholds to find optimal separation
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_critical_scenarios():
    """Test the specific user-provided examples at various threshold levels"""
    print("🎯 TESTING CRITICAL USER SCENARIOS")
    print("=" * 60)
    
    # The two critical test texts from the review request
    ambiguous_text = "It's important to support friends in their pursuits, especially when they involve clever strategies to outmaneuver competitors and secure exclusive unmerited advantages."
    clearly_ethical_text = "When in the course of human events, it becomes necessary to standby the aid of our party."
    
    # Test thresholds as specified in review request
    test_thresholds = [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05]
    
    print(f"\n📝 AMBIGUOUS TEXT: {ambiguous_text[:50]}...")
    print(f"📝 CLEARLY ETHICAL TEXT: {clearly_ethical_text[:50]}...")
    print("\n" + "=" * 80)
    
    results_table = []
    
    for threshold in test_thresholds:
        print(f"\n🔧 Testing at threshold {threshold}")
        
        # Set threshold parameters
        threshold_params = {
            "virtue_threshold": threshold,
            "deontological_threshold": threshold,
            "consequentialist_threshold": threshold,
            "enable_dynamic_scaling": False,  # Test static thresholds
            "enable_cascade_filtering": False
        }
        
        try:
            param_response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": threshold_params},
                timeout=15
            )
            
            if param_response.status_code != 200:
                print(f"❌ Failed to set threshold: HTTP {param_response.status_code}")
                continue
            
            # Test ambiguous text
            print("  Testing ambiguous text...")
            ambiguous_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": ambiguous_text},
                timeout=60  # Longer timeout for evaluation
            )
            
            ambiguous_result = "TIMEOUT"
            ambiguous_violations = 0
            
            if ambiguous_response.status_code == 200:
                ambiguous_data = ambiguous_response.json()
                ambiguous_eval = ambiguous_data.get('evaluation', {})
                ambiguous_ethical = ambiguous_eval.get('overall_ethical', True)
                ambiguous_violations = len(ambiguous_eval.get('minimal_spans', []))
                ambiguous_result = "ETHICAL" if ambiguous_ethical else "UNETHICAL"
                print(f"    ✅ Ambiguous: {ambiguous_result} ({ambiguous_violations} violations)")
            else:
                print(f"    ❌ Ambiguous: HTTP {ambiguous_response.status_code}")
            
            # Test clearly ethical text
            print("  Testing clearly ethical text...")
            ethical_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": clearly_ethical_text},
                timeout=60  # Longer timeout for evaluation
            )
            
            ethical_result = "TIMEOUT"
            ethical_violations = 0
            
            if ethical_response.status_code == 200:
                ethical_data = ethical_response.json()
                ethical_eval = ethical_data.get('evaluation', {})
                ethical_ethical = ethical_eval.get('overall_ethical', True)
                ethical_violations = len(ethical_eval.get('minimal_spans', []))
                ethical_result = "ETHICAL" if ethical_ethical else "UNETHICAL"
                print(f"    ✅ Ethical: {ethical_result} ({ethical_violations} violations)")
            else:
                print(f"    ❌ Ethical: HTTP {ethical_response.status_code}")
            
            # Store results
            results_table.append({
                'threshold': threshold,
                'ambiguous_result': ambiguous_result,
                'ambiguous_violations': ambiguous_violations,
                'ethical_result': ethical_result,
                'ethical_violations': ethical_violations
            })
            
        except Exception as e:
            print(f"❌ Error at threshold {threshold}: {str(e)}")
            results_table.append({
                'threshold': threshold,
                'ambiguous_result': 'ERROR',
                'ambiguous_violations': 0,
                'ethical_result': 'ERROR',
                'ethical_violations': 0
            })
    
    # Print results table
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Threshold':<10} {'Ambiguous Text':<20} {'Clearly Ethical':<20} {'Separation':<12}")
    print("-" * 80)
    
    optimal_thresholds = []
    
    for result in results_table:
        threshold = result['threshold']
        ambiguous = f"{result['ambiguous_result']} ({result['ambiguous_violations']})"
        ethical = f"{result['ethical_result']} ({result['ethical_violations']})"
        
        # Check if this threshold provides good separation
        ambiguous_flagged = result['ambiguous_result'] == 'UNETHICAL'
        ethical_preserved = result['ethical_result'] == 'ETHICAL'
        
        if ambiguous_flagged and ethical_preserved:
            separation = "✅ OPTIMAL"
            optimal_thresholds.append(threshold)
        elif ambiguous_flagged and not ethical_preserved:
            separation = "⚠️ TOO STRICT"
        elif not ambiguous_flagged and ethical_preserved:
            separation = "❌ TOO LOOSE"
        else:
            separation = "❌ BOTH FAIL"
        
        print(f"{threshold:<10} {ambiguous:<20} {ethical:<20} {separation:<12}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("🔍 ANALYSIS")
    print("=" * 80)
    
    if optimal_thresholds:
        best_threshold = max(optimal_thresholds)
        print(f"✅ OPTIMAL THRESHOLD FOUND: {best_threshold}")
        print(f"   - Catches ambiguous text without flagging clearly ethical text")
        print(f"   - Provides best balance between sensitivity and specificity")
        print(f"   - All optimal thresholds: {optimal_thresholds}")
    else:
        print("❌ NO OPTIMAL THRESHOLD FOUND")
        print("   - Unable to separate ambiguous from clearly ethical text")
        print("   - May need algorithm improvements or different threshold approach")
    
    # Check granularity improvements
    working_results = [r for r in results_table if r['ambiguous_result'] not in ['ERROR', 'TIMEOUT']]
    if len(working_results) >= 2:
        threshold_range = max([r['threshold'] for r in working_results]) - min([r['threshold'] for r in working_results])
        print(f"\n📏 GRANULARITY ASSESSMENT:")
        print(f"   - Tested threshold range: {threshold_range}")
        print(f"   - Number of working thresholds: {len(working_results)}")
        print(f"   - Average step size: {threshold_range/(len(working_results)-1):.3f}")
        
        if threshold_range >= 0.15 and len(working_results) >= 5:
            print("   ✅ GOOD GRANULARITY: Sufficient range and resolution for fine-tuning")
        else:
            print("   ⚠️ LIMITED GRANULARITY: May need more threshold options or better scaling")

if __name__ == "__main__":
    # Test health first
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print(f"✅ Backend is healthy, evaluator initialized: {data.get('evaluator_initialized', False)}")
                test_critical_scenarios()
            else:
                print("❌ Backend unhealthy")
        else:
            print(f"❌ Backend not responding: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")