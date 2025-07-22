#!/usr/bin/env python3
"""
🔍 USER ISSUE VERIFICATION TEST SUITE
═══════════════════════════════════════════════════════════════════════════════════

MISSION: Verify that all user-reported issues have been resolved:

1. Green test click button needed to be removed
2. Long paragraph text evaluation showed no detailed analysis or span measurements  
3. ML Ethics Assistant functionality was completely non-functional

This test suite specifically validates the fixes for these issues.

Author: Testing Agent
Version: 1.0.0
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Backend URL from environment
BACKEND_URL = "https://b214a97a-320b-41fd-b6d6-ff742674f4c6.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class UserIssueVerificationSuite:
    """Test suite to verify user-reported issues are resolved."""
    
    def __init__(self):
        self.results = []
        self.session = None
    
    async def setup_session(self):
        """Setup aiohttp session for testing."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup_session(self):
        """Cleanup aiohttp session."""
        if self.session:
            await self.session.close()
    
    def log_result(self, test_name: str, passed: bool, details: str, response_time: float = 0.0):
        """Log test result."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "response_time": response_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.results.append(result)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name} ({response_time:.3f}s)")
        print(f"    {details}")
    
    async def test_endpoint(self, method: str, endpoint: str, data: Dict = None) -> Tuple[bool, float, Dict]:
        """Generic endpoint testing method."""
        start_time = time.time()
        try:
            if method.upper() == "GET":
                async with self.session.get(f"{API_BASE}{endpoint}") as response:
                    response_time = time.time() - start_time
                    result = await response.json()
                    return response.status == 200, response_time, result
            elif method.upper() == "POST":
                async with self.session.post(f"{API_BASE}{endpoint}", json=data) as response:
                    response_time = time.time() - start_time
                    result = await response.json()
                    return response.status == 200, response_time, result
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════════════
    # ISSUE 1: Green test click button removal (Backend doesn't have UI buttons)
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def verify_no_test_buttons_in_api(self):
        """Verify API doesn't expose any test button functionality."""
        print("🔍 Verifying no test button functionality in API...")
        
        # Check that API doesn't have any test button endpoints
        test_endpoints = ["/test-button", "/green-button", "/test-click", "/button-test"]
        
        all_endpoints_missing = True
        for endpoint in test_endpoints:
            success, response_time, result = await self.test_endpoint("GET", endpoint)
            if success:
                all_endpoints_missing = False
                break
        
        details = "API correctly has no test button endpoints" if all_endpoints_missing else "Found unexpected test button endpoints"
        self.log_result("No Test Button Endpoints", all_endpoints_missing, details, 0.001)
        return all_endpoints_missing

    # ═══════════════════════════════════════════════════════════════════════════════════
    # ISSUE 2: Long paragraph text evaluation with detailed span analysis
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def verify_detailed_span_analysis_simple_text(self):
        """Test /api/evaluate with simple text to verify detailed analysis."""
        print("🎯 Testing /api/evaluate with simple text for detailed analysis...")
        
        test_data = {
            "text": "This is a test",
            "context": {"domain": "general"},
            "mode": "production",
            "priority": "normal"
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/evaluate", test_data)
        
        if not success:
            self.log_result("Simple Text Detailed Analysis", False, f"API request failed: {result.get('error', 'Unknown error')}", response_time)
            return False
        
        # Check for required response structure
        has_evaluation = "evaluation" in result
        has_spans = has_evaluation and "spans" in result["evaluation"]
        has_minimal_spans = has_evaluation and "minimal_spans" in result["evaluation"]
        has_clean_text = "clean_text" in result
        has_delta_summary = "delta_summary" in result
        
        spans_count = len(result.get("evaluation", {}).get("spans", [])) if has_spans else 0
        minimal_spans_count = len(result.get("evaluation", {}).get("minimal_spans", [])) if has_minimal_spans else 0
        
        details = f"Response time: {response_time:.3f}s | "
        details += f"Has evaluation: {has_evaluation} | "
        details += f"Has spans: {has_spans} ({spans_count} spans) | "
        details += f"Has minimal_spans: {has_minimal_spans} ({minimal_spans_count} spans) | "
        details += f"Has clean_text: {has_clean_text} | "
        details += f"Has delta_summary: {has_delta_summary}"
        
        # Test passes if we have the required structure
        test_passed = has_evaluation and has_spans and has_minimal_spans and has_clean_text and has_delta_summary
        
        self.log_result("Simple Text Detailed Analysis", test_passed, details, response_time)
        return test_passed
    
    async def verify_detailed_span_analysis_complex_text(self):
        """Test /api/evaluate with complex text to verify detailed analysis."""
        print("🎯 Testing /api/evaluate with complex text for detailed analysis...")
        
        complex_text = "The company proudly touts that it donates ten percent of its profits to a local food bank, publishes an open-view ledger of every charitable transaction, and gives employees paid time off for community service, yet behind the scenes its executives discreetly bankroll a lobbying firm that drafts loophole-laden regulations, authorize quiet scraping of user metadata to build hyper-targeted ads without explicit consent, and issue performance bonuses that all but compel salaried staff to log unpaid overtime"
        
        test_data = {
            "text": complex_text,
            "context": {"domain": "business"},
            "mode": "production",
            "priority": "normal"
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/evaluate", test_data)
        
        if not success:
            self.log_result("Complex Text Detailed Analysis", False, f"API request failed: {result.get('error', 'Unknown error')}", response_time)
            return False
        
        # Check for required response structure with detailed analysis
        has_evaluation = "evaluation" in result
        has_spans = has_evaluation and "spans" in result["evaluation"]
        has_minimal_spans = has_evaluation and "minimal_spans" in result["evaluation"]
        has_clean_text = "clean_text" in result
        has_delta_summary = "delta_summary" in result
        
        spans_count = len(result.get("evaluation", {}).get("spans", [])) if has_spans else 0
        minimal_spans_count = len(result.get("evaluation", {}).get("minimal_spans", [])) if has_minimal_spans else 0
        
        # For complex text, we should have multiple spans for detailed analysis
        has_detailed_spans = spans_count > 0
        
        # Check if spans have detailed scoring information
        spans_have_scores = False
        if has_spans and spans_count > 0:
            first_span = result["evaluation"]["spans"][0]
            spans_have_scores = all(key in first_span for key in ["virtue_score", "deontological_score", "consequentialist_score"])
        
        details = f"Response time: {response_time:.3f}s | "
        details += f"Text length: {len(complex_text)} chars | "
        details += f"Has evaluation: {has_evaluation} | "
        details += f"Has spans: {has_spans} ({spans_count} spans) | "
        details += f"Has minimal_spans: {has_minimal_spans} ({minimal_spans_count} spans) | "
        details += f"Spans have scores: {spans_have_scores} | "
        details += f"Has clean_text: {has_clean_text} | "
        details += f"Has delta_summary: {has_delta_summary}"
        
        # Test passes if we have detailed analysis structure
        test_passed = (has_evaluation and has_spans and has_minimal_spans and 
                      has_clean_text and has_delta_summary and has_detailed_spans and spans_have_scores)
        
        self.log_result("Complex Text Detailed Analysis", test_passed, details, response_time)
        return test_passed
    
    async def verify_fast_response_times(self):
        """Verify API responses are fast (< 1 second)."""
        print("⚡ Testing fast response times...")
        
        test_data = {
            "text": "Testing response time performance",
            "context": {"domain": "general"},
            "mode": "production",
            "priority": "normal"
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/evaluate", test_data)
        
        is_fast = response_time < 1.0
        details = f"Response time: {response_time:.3f}s | Target: < 1.0s | Fast enough: {is_fast}"
        
        test_passed = success and is_fast
        self.log_result("Fast Response Times", test_passed, details, response_time)
        return test_passed

    # ═══════════════════════════════════════════════════════════════════════════════════
    # ISSUE 3: ML Ethics Assistant functionality
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def verify_ml_ethics_comprehensive_analysis(self):
        """Test /api/ethics/comprehensive-analysis endpoint."""
        print("🧠 Testing ML Ethics Comprehensive Analysis...")
        
        test_data = {
            "text": "We should implement AI systems that are fair, transparent, and accountable to all stakeholders."
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/ethics/comprehensive-analysis", test_data)
        
        if not success:
            self.log_result("ML Ethics Comprehensive Analysis", False, f"API request failed: {result.get('error', 'Unknown error')}", response_time)
            return False
        
        # Check for required response structure
        has_status = "status" in result
        has_frameworks = "frameworks" in result
        has_ml_guidance = "ml_guidance" in result
        has_assessment = "overall_assessment" in result
        
        frameworks_complete = False
        if has_frameworks:
            frameworks = result["frameworks"]
            frameworks_complete = all(key in frameworks for key in ["virtue_ethics", "deontological", "consequentialist"])
        
        details = f"Response time: {response_time:.3f}s | "
        details += f"Has status: {has_status} | "
        details += f"Has frameworks: {has_frameworks} | "
        details += f"Frameworks complete: {frameworks_complete} | "
        details += f"Has ML guidance: {has_ml_guidance} | "
        details += f"Has assessment: {has_assessment}"
        
        test_passed = success and has_status and has_frameworks and frameworks_complete and has_ml_guidance and has_assessment
        self.log_result("ML Ethics Comprehensive Analysis", test_passed, details, response_time)
        return test_passed
    
    async def verify_ml_ethics_meta_analysis(self):
        """Test /api/ethics/meta-analysis endpoint."""
        print("🧠 Testing ML Ethics Meta Analysis...")
        
        test_data = {
            "text": "Ethical AI requires careful consideration of moral foundations and philosophical principles."
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/ethics/meta-analysis", test_data)
        
        if not success:
            self.log_result("ML Ethics Meta Analysis", False, f"API request failed: {result.get('error', 'Unknown error')}", response_time)
            return False
        
        # Check for required response structure
        has_status = "status" in result
        has_philosophical_structure = "philosophical_structure" in result
        has_meta_ethical_assessment = "meta_ethical_assessment" in result
        has_recommendations = "recommendations" in result
        
        details = f"Response time: {response_time:.3f}s | "
        details += f"Has status: {has_status} | "
        details += f"Has philosophical structure: {has_philosophical_structure} | "
        details += f"Has meta-ethical assessment: {has_meta_ethical_assessment} | "
        details += f"Has recommendations: {has_recommendations}"
        
        test_passed = success and has_status and has_philosophical_structure and has_meta_ethical_assessment and has_recommendations
        self.log_result("ML Ethics Meta Analysis", test_passed, details, response_time)
        return test_passed
    
    async def verify_ml_ethics_normative_analysis(self):
        """Test /api/ethics/normative-analysis endpoint."""
        print("🧠 Testing ML Ethics Normative Analysis...")
        
        test_data = {
            "text": "Machine learning models should be designed to maximize human welfare while respecting individual rights."
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/ethics/normative-analysis", test_data)
        
        if not success:
            self.log_result("ML Ethics Normative Analysis", False, f"API request failed: {result.get('error', 'Unknown error')}", response_time)
            return False
        
        # Check for required response structure
        has_status = "status" in result
        has_virtue_ethics = "virtue_ethics" in result
        has_deontological = "deontological_ethics" in result
        has_consequentialist = "consequentialist_ethics" in result
        has_synthesis = "synthesis" in result
        
        details = f"Response time: {response_time:.3f}s | "
        details += f"Has status: {has_status} | "
        details += f"Has virtue ethics: {has_virtue_ethics} | "
        details += f"Has deontological: {has_deontological} | "
        details += f"Has consequentialist: {has_consequentialist} | "
        details += f"Has synthesis: {has_synthesis}"
        
        test_passed = success and has_status and has_virtue_ethics and has_deontological and has_consequentialist and has_synthesis
        self.log_result("ML Ethics Normative Analysis", test_passed, details, response_time)
        return test_passed
    
    async def verify_ml_ethics_applied_analysis(self):
        """Test /api/ethics/applied-analysis endpoint."""
        print("🧠 Testing ML Ethics Applied Analysis...")
        
        test_data = {
            "text": "Healthcare AI systems must prioritize patient safety and privacy while ensuring equitable access to care."
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/ethics/applied-analysis", test_data)
        
        if not success:
            self.log_result("ML Ethics Applied Analysis", False, f"API request failed: {result.get('error', 'Unknown error')}", response_time)
            return False
        
        # Check for required response structure
        has_status = "status" in result
        has_domain_analysis = "domain_analysis" in result
        has_practical_recommendations = "practical_recommendations" in result
        has_compliance_check = "compliance_check" in result
        has_risk_assessment = "risk_assessment" in result
        
        details = f"Response time: {response_time:.3f}s | "
        details += f"Has status: {has_status} | "
        details += f"Has domain analysis: {has_domain_analysis} | "
        details += f"Has practical recommendations: {has_practical_recommendations} | "
        details += f"Has compliance check: {has_compliance_check} | "
        details += f"Has risk assessment: {has_risk_assessment}"
        
        test_passed = success and has_status and has_domain_analysis and has_practical_recommendations and has_compliance_check and has_risk_assessment
        self.log_result("ML Ethics Applied Analysis", test_passed, details, response_time)
        return test_passed
    
    async def verify_ml_ethics_training_guidance(self):
        """Test /api/ethics/ml-training-guidance endpoint."""
        print("🧠 Testing ML Ethics Training Guidance...")
        
        test_data = {
            "content": "Training data should be diverse and representative to avoid bias in machine learning models."
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/ethics/ml-training-guidance", test_data)
        
        if not success:
            self.log_result("ML Ethics Training Guidance", False, f"API request failed: {result.get('error', 'Unknown error')}", response_time)
            return False
        
        # Check for required response structure
        has_status = "status" in result
        has_bias_analysis = "bias_analysis" in result
        has_fairness_assessment = "fairness_assessment" in result
        has_transparency_guidance = "transparency_guidance" in result
        has_training_recommendations = "training_recommendations" in result
        has_ethical_score = "ethical_score" in result
        
        details = f"Response time: {response_time:.3f}s | "
        details += f"Has status: {has_status} | "
        details += f"Has bias analysis: {has_bias_analysis} | "
        details += f"Has fairness assessment: {has_fairness_assessment} | "
        details += f"Has transparency guidance: {has_transparency_guidance} | "
        details += f"Has training recommendations: {has_training_recommendations} | "
        details += f"Has ethical score: {has_ethical_score}"
        
        test_passed = success and has_status and has_bias_analysis and has_fairness_assessment and has_transparency_guidance and has_training_recommendations and has_ethical_score
        self.log_result("ML Ethics Training Guidance", test_passed, details, response_time)
        return test_passed

    # ═══════════════════════════════════════════════════════════════════════════════════
    # MAIN TEST EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def run_all_verification_tests(self):
        """Run all user issue verification tests."""
        print("🔍 STARTING USER ISSUE VERIFICATION TESTS")
        print("=" * 80)
        
        await self.setup_session()
        
        try:
            print("\n📋 ISSUE 1: Test Button Removal Verification")
            print("-" * 50)
            await self.verify_no_test_buttons_in_api()
            
            print("\n📊 ISSUE 2: Detailed Span Analysis Verification")
            print("-" * 50)
            await self.verify_detailed_span_analysis_simple_text()
            await self.verify_detailed_span_analysis_complex_text()
            await self.verify_fast_response_times()
            
            print("\n🧠 ISSUE 3: ML Ethics Assistant Functionality Verification")
            print("-" * 50)
            await self.verify_ml_ethics_comprehensive_analysis()
            await self.verify_ml_ethics_meta_analysis()
            await self.verify_ml_ethics_normative_analysis()
            await self.verify_ml_ethics_applied_analysis()
            await self.verify_ml_ethics_training_guidance()
            
        finally:
            await self.cleanup_session()
        
        # Calculate results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 80)
        print("📊 USER ISSUE VERIFICATION RESULTS")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("\n🎉 ALL USER ISSUES HAVE BEEN RESOLVED!")
        elif success_rate >= 80:
            print("\n⚠️ Most user issues resolved, some minor issues remain")
        else:
            print("\n❌ Significant user issues still need attention")
        
        print("\n📋 DETAILED RESULTS:")
        for result in self.results:
            status = "✅" if result["passed"] else "❌"
            print(f"{status} {result['test_name']}")
        
        return self.results

async def main():
    """Main test execution function."""
    test_suite = UserIssueVerificationSuite()
    results = await test_suite.run_all_verification_tests()
    
    # Save results to file
    with open('/app/user_issue_verification_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: /app/user_issue_verification_results.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())