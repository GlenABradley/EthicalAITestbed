#!/usr/bin/env python3
"""
Focused Heat-Map Testing for Phase 4A
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

def test_heat_map_comprehensive():
    """Comprehensive heat-map testing as requested in review"""
    
    print("🔥 PHASE 4A HEAT-MAP VISUALIZATION COMPREHENSIVE TESTING")
    print("=" * 70)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"API Base: {API_BASE}")
    print()
    
    results = []
    
    # Test cases as specified in review request
    test_cases = [
        ("Hello world", "Short text"),
        ("This is a test of ethical evaluation with some content", "Medium text"),
        ("AI systems should respect human autonomy and avoid manipulation or deception in their interactions with users to maintain trust and ethical standards", "Long text"),
        ("", "Empty text"),
        ("Testing with émojis 🚀 and special chars @#$%", "Special characters")
    ]
    
    for text, description in test_cases:
        print(f"Testing {description}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Test heat-map-mock endpoint
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": text},
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ['evaluations', 'overallGrades', 'textLength', 'originalEvaluation']
                structure_valid = all(field in data for field in required_fields)
                
                # Validate evaluations
                evaluations = data.get('evaluations', {})
                eval_types = ['short', 'medium', 'long', 'stochastic']
                eval_structure_valid = all(eval_type in evaluations for eval_type in eval_types)
                
                # Validate data quality
                quality_issues = []
                
                for eval_type, eval_data in evaluations.items():
                    spans = eval_data.get('spans', [])
                    for i, span in enumerate(spans):
                        # Check span positions
                        span_pos = span.get('span', [])
                        if len(span_pos) == 2:
                            start, end = span_pos
                            if start < 0 or end > len(text) or start >= end:
                                quality_issues.append(f"{eval_type} span {i}: invalid position")
                        
                        # Check scores
                        scores = span.get('scores', {})
                        for score_type, score_val in scores.items():
                            if not (0.0 <= score_val <= 1.0):
                                quality_issues.append(f"{eval_type} span {i}: {score_type} score out of range")
                
                # Check text length
                text_length_correct = data.get('textLength') == len(text)
                
                # Check grades format
                overall_grades = data.get('overallGrades', {})
                grades_valid = all('(' in grade and ')' in grade for grade in overall_grades.values())
                
                # Performance check
                performance_ok = response_time < 100  # Target: <100ms
                
                result = {
                    'test': description,
                    'success': structure_valid and eval_structure_valid and text_length_correct and grades_valid and len(quality_issues) == 0,
                    'response_time_ms': response_time,
                    'performance_ok': performance_ok,
                    'structure_valid': structure_valid,
                    'eval_structure_valid': eval_structure_valid,
                    'text_length_correct': text_length_correct,
                    'grades_valid': grades_valid,
                    'quality_issues': quality_issues,
                    'span_counts': {eval_type: len(eval_data.get('spans', [])) for eval_type, eval_data in evaluations.items()},
                    'overall_grades': overall_grades
                }
                
                results.append(result)
                
                status = "✅" if result['success'] else "❌"
                print(f"  {status} {description}: {response_time:.1f}ms, spans: {result['span_counts']}")
                if quality_issues:
                    print(f"    Quality issues: {quality_issues}")
                
            else:
                print(f"  ❌ {description}: HTTP {response.status_code}")
                results.append({
                    'test': description,
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'response_text': response.text[:200]
                })
                
        except Exception as e:
            print(f"  ❌ {description}: Error - {str(e)}")
            results.append({
                'test': description,
                'success': False,
                'error': str(e)
            })
    
    print()
    
    # Test error handling
    print("Testing Error Handling:")
    
    # Test missing text field
    try:
        response = requests.post(f"{API_BASE}/heat-map-mock", json={}, timeout=10)
        error_handling_ok = response.status_code in [400, 422]
        print(f"  {'✅' if error_handling_ok else '❌'} Missing text field: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ❌ Missing text field: Error - {str(e)}")
    
    # Test integration with existing endpoints
    print("\nTesting Integration:")
    
    try:
        health_response = requests.get(f"{API_BASE}/health", timeout=10)
        health_ok = health_response.status_code == 200
        print(f"  {'✅' if health_ok else '❌'} Health endpoint: HTTP {health_response.status_code}")
        
        params_response = requests.get(f"{API_BASE}/parameters", timeout=10)
        params_ok = params_response.status_code == 200
        print(f"  {'✅' if params_ok else '❌'} Parameters endpoint: HTTP {params_response.status_code}")
        
        integration_ok = health_ok and params_ok
        print(f"  {'✅' if integration_ok else '❌'} Integration with existing v1.1 features")
        
    except Exception as e:
        print(f"  ❌ Integration test: Error - {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("HEAT-MAP TESTING SUMMARY")
    print("=" * 70)
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successful tests: {len(successful_tests)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    print(f"📊 Total tests: {len(results)}")
    
    if successful_tests:
        avg_response_time = sum(r.get('response_time_ms', 0) for r in successful_tests) / len(successful_tests)
        fast_responses = sum(1 for r in successful_tests if r.get('performance_ok', False))
        print(f"⚡ Average response time: {avg_response_time:.1f}ms")
        print(f"🚀 Fast responses (<100ms): {fast_responses}/{len(successful_tests)}")
    
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for test in failed_tests:
            print(f"  - {test['test']}: {test.get('error', 'Unknown error')}")
    
    # Detailed analysis for successful tests
    if successful_tests:
        print(f"\n📊 DETAILED ANALYSIS:")
        for test in successful_tests:
            print(f"  {test['test']}:")
            print(f"    Response time: {test.get('response_time_ms', 0):.1f}ms")
            print(f"    Span counts: {test.get('span_counts', {})}")
            print(f"    Grades: {test.get('overall_grades', {})}")
    
    return len(failed_tests) == 0

if __name__ == "__main__":
    success = test_heat_map_comprehensive()
    exit(0 if success else 1)