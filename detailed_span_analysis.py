#!/usr/bin/env python3
"""
DETAILED SPAN SCORING ANALYSIS
Investigate the actual span-level scoring to understand calibration issues
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

def analyze_span_scoring():
    """Analyze detailed span scoring to understand calibration issues"""
    print("🔬 DETAILED SPAN SCORING ANALYSIS")
    print("=" * 60)
    
    # Test texts from review request
    test_texts = [
        ("It's important to support friends in their pursuits, especially when they involve clever strategies to outmaneuver competitors and secure exclusive unmerited advantages.", "AMBIGUOUS"),
        ("When in the course of human events, it becomes necessary to standby the aid of our party.", "CLEARLY_ETHICAL"),
        ("You are stupid and worthless", "PROBLEMATIC"),
        ("I love helping people", "POSITIVE")
    ]
    
    # Test at different thresholds
    test_thresholds = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    
    for text, description in test_texts:
        print(f"\n📝 ANALYZING: {description}")
        print(f"Text: '{text[:80]}{'...' if len(text) > 80 else ''}'")
        print("-" * 60)
        
        for threshold in test_thresholds:
            # Set threshold
            threshold_params = {
                "virtue_threshold": threshold,
                "deontological_threshold": threshold,
                "consequentialist_threshold": threshold,
                "enable_dynamic_scaling": False,
                "enable_cascade_filtering": False
            }
            
            param_response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": threshold_params},
                timeout=10
            )
            
            if param_response.status_code != 200:
                print(f"❌ Failed to set threshold {threshold}")
                continue
            
            # Evaluate text
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": text},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                evaluation = data.get('evaluation', {})
                
                overall_ethical = evaluation.get('overall_ethical', True)
                minimal_spans = evaluation.get('minimal_spans', [])
                all_spans = evaluation.get('all_spans_with_scores', [])
                
                print(f"Threshold {threshold:.2f}: {'ETHICAL' if overall_ethical else 'UNETHICAL'} ({len(minimal_spans)} violations)")
                
                # Show detailed span scores
                if all_spans:
                    violation_spans = [span for span in all_spans if span.get('any_violation', False)]
                    if violation_spans:
                        print(f"  Violated spans ({len(violation_spans)}):")
                        for i, span in enumerate(violation_spans[:3]):  # Show first 3
                            span_text = span.get('span_text', 'unknown')
                            virtue_score = span.get('virtue_score', 1.0)
                            deont_score = span.get('deontological_score', 1.0)
                            conseq_score = span.get('consequentialist_score', 1.0)
                            
                            violations = []
                            if virtue_score < threshold: violations.append(f"V:{virtue_score:.3f}")
                            if deont_score < threshold: violations.append(f"D:{deont_score:.3f}")
                            if conseq_score < threshold: violations.append(f"C:{conseq_score:.3f}")
                            
                            print(f"    '{span_text[:40]}...' -> {', '.join(violations)}")
                    else:
                        # Show some non-violating spans to understand score distribution
                        print(f"  Sample spans (no violations):")
                        for i, span in enumerate(all_spans[:2]):  # Show first 2
                            span_text = span.get('span_text', 'unknown')
                            virtue_score = span.get('virtue_score', 1.0)
                            deont_score = span.get('deontological_score', 1.0)
                            conseq_score = span.get('consequentialist_score', 1.0)
                            
                            print(f"    '{span_text[:40]}...' -> V:{virtue_score:.3f}, D:{deont_score:.3f}, C:{conseq_score:.3f}")
                else:
                    print(f"  No span data available")
            else:
                print(f"❌ Failed to evaluate at threshold {threshold}: HTTP {response.status_code}")
            
            time.sleep(0.2)  # Brief pause
        
        print()

def test_mathematical_framework():
    """Test the mathematical framework to understand score compression"""
    print("\n🧮 MATHEMATICAL FRAMEWORK ANALYSIS")
    print("=" * 60)
    
    # Set a very low threshold to capture all scoring details
    threshold = 0.01
    threshold_params = {
        "virtue_threshold": threshold,
        "deontological_threshold": threshold,
        "consequentialist_threshold": threshold,
        "enable_dynamic_scaling": False,
        "enable_cascade_filtering": False
    }
    
    requests.post(f"{API_BASE}/update-parameters", json={"parameters": threshold_params}, timeout=10)
    
    # Test a variety of texts to understand score distribution
    test_texts = [
        "I love you",
        "I hate you", 
        "You are wonderful",
        "You are terrible",
        "This is good",
        "This is bad",
        "Help others",
        "Hurt others",
        "Be kind",
        "Be cruel"
    ]
    
    all_scores = []
    
    for text in test_texts:
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": text},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            evaluation = data.get('evaluation', {})
            all_spans = evaluation.get('all_spans_with_scores', [])
            
            print(f"'{text}' -> {len(all_spans)} spans")
            
            for span in all_spans:
                virtue_score = span.get('virtue_score', 1.0)
                deont_score = span.get('deontological_score', 1.0)
                conseq_score = span.get('consequentialist_score', 1.0)
                
                all_scores.extend([virtue_score, deont_score, conseq_score])
                
                span_text = span.get('span_text', 'unknown')
                print(f"  '{span_text}' -> V:{virtue_score:.3f}, D:{deont_score:.3f}, C:{conseq_score:.3f}")
    
    # Analyze score distribution
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score
        
        # Count scores in different ranges
        below_0_1 = sum(1 for s in all_scores if s < 0.1)
        below_0_2 = sum(1 for s in all_scores if s < 0.2)
        below_0_5 = sum(1 for s in all_scores if s < 0.5)
        total_scores = len(all_scores)
        
        print(f"\n📊 SCORE DISTRIBUTION:")
        print(f"Total scores: {total_scores}")
        print(f"Range: {min_score:.3f} to {max_score:.3f} (span: {score_range:.3f})")
        print(f"Below 0.1: {below_0_1}/{total_scores} ({below_0_1/total_scores:.1%})")
        print(f"Below 0.2: {below_0_2}/{total_scores} ({below_0_2/total_scores:.1%})")
        print(f"Below 0.5: {below_0_5}/{total_scores} ({below_0_5/total_scores:.1%})")
        
        # Check for compression to 0.0-0.2 range
        if max_score <= 0.2:
            print("❌ SEVERE COMPRESSION: All scores compressed to 0.0-0.2 range")
        elif below_0_2/total_scores > 0.8:
            print("⚠️ SIGNIFICANT COMPRESSION: >80% of scores below 0.2")
        else:
            print("✅ ADEQUATE DISTRIBUTION: Scores span reasonable range")

if __name__ == "__main__":
    analyze_span_scoring()
    test_mathematical_framework()