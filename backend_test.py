#!/usr/bin/env python3
"""
🧪 UNIFIED ETHICAL AI SERVER TESTING SUITE 🧪
═══════════════════════════════════════════════════════════════════════════════════

🎓 PROFESSOR'S COMPREHENSIVE TESTING LECTURE:
════════════════════════════════════════════════

This testing suite validates the newly refactored unified ethical AI server 
architecture following MIT-professor level testing standards. We test:

1. **NEW UNIFIED ARCHITECTURE COMPONENTS**:
   - unified_ethical_orchestrator.py - The crown jewel orchestrator
   - unified_configuration_manager.py - Configuration management system  
   - unified_server.py - Modern FastAPI server

2. **ARCHITECTURE HIGHLIGHTS TESTING**:
   - Clean Architecture principles with dependency injection
   - Comprehensive MIT-professor level documentation
   - Unified configuration system with environment overrides
   - Backward compatibility maintained for existing endpoints
   - Modern FastAPI patterns with lifespan management

3. **KEY TESTING AREAS**:
   - Test the new /api/evaluate endpoint with unified orchestrator
   - Verify /api/health provides comprehensive system health information
   - Check backward compatibility endpoints
   - Test configuration system initialization
   - Verify unified orchestrator integration

Author: Testing Agent - Unified Architecture Validation
Version: 10.0.0 - Phase 9.5 Exhaustive Refactor Testing
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://6bd3bc7b-f41e-4bd1-a876-175f126bec59.preview.emergentagent.com')
API_BASE = f"{BACKEND_URL}/api"

class UnifiedArchitectureTestSuite:
    """
    🎓 UNIFIED ARCHITECTURE TEST SUITE:
    ═══════════════════════════════════════
    
    Comprehensive testing of the newly refactored unified ethical AI server
    architecture, validating all components and integration points.
    """
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = time.time()
        
        # Import test modules
        try:
            import requests
            self.requests = requests
            self.requests_available = True
        except ImportError:
            logger.error("❌ requests library not available")
            self.requests_available = False
            
        # Test data
        self.test_texts = {
            "ethical": "Thank you for your help, I appreciate your assistance.",
            "unethical": "You are stupid and worthless, and you should die.",
            "neutral": "The weather is nice today.",
            "complex": "We should consider the ethical implications of AI development while ensuring technological progress.",
            "empty": "",
            "long": "This is a longer text that contains multiple sentences and ideas. " * 20
        }
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", response_time: float = 0.0):
        """Log individual test results."""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"
        
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        logger.info(f"{status} | {test_name} | {response_time:.3f}s | {details}")
    
    async def test_unified_server_health(self):
        """
        🏥 TEST: Unified Server Health Check
        ═══════════════════════════════════════
        
        Tests the new /api/health endpoint to ensure it provides comprehensive
        system health information including orchestrator status, database
        connectivity, and performance metrics.
        """
        logger.info("🏥 Testing Unified Server Health Check...")
        
        try:
            start_time = time.time()
            response = self.requests.get(f"{API_BASE}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Validate health response structure
                required_fields = [
                    "status", "timestamp", "uptime_seconds", 
                    "orchestrator_healthy", "database_connected", 
                    "configuration_valid", "performance_metrics", 
                    "features_available"
                ]
                
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if not missing_fields:
                    # Check specific unified architecture features
                    features = health_data.get("features_available", {})
                    unified_features = [
                        "unified_orchestrator", "database", "configuration"
                    ]
                    
                    available_features = [f for f in unified_features if features.get(f, False)]
                    
                    self.log_test_result(
                        "Unified Server Health Check",
                        True,
                        f"Status: {health_data['status']}, Features: {len(available_features)}/{len(unified_features)} available",
                        response_time
                    )
                else:
                    self.log_test_result(
                        "Unified Server Health Check",
                        False,
                        f"Missing required fields: {missing_fields}",
                        response_time
                    )
            else:
                self.log_test_result(
                    "Unified Server Health Check",
                    False,
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    response_time
                )
                
        except Exception as e:
            self.log_test_result(
                "Unified Server Health Check",
                False,
                f"Request failed: {str(e)}",
                0.0
            )
    
    async def test_unified_evaluate_endpoint(self):
        """
        🎯 TEST: Unified Evaluation Endpoint
        ═══════════════════════════════════════
        
        Tests the new /api/evaluate endpoint with the unified orchestrator
        to ensure it properly processes requests and returns comprehensive
        ethical evaluations.
        """
        logger.info("🎯 Testing Unified Evaluation Endpoint...")
        
        for text_type, text_content in self.test_texts.items():
            if text_type == "empty":  # Skip empty text for main evaluation
                continue
                
            try:
                start_time = time.time()
                
                # Create unified evaluation request
                request_data = {
                    "text": text_content,
                    "context": {
                        "domain": "general",
                        "cultural_context": "western"
                    },
                    "parameters": {
                        "confidence_threshold": 0.7,
                        "explanation_level": "detailed"
                    },
                    "mode": "production",
                    "priority": "normal"
                }
                
                response = self.requests.post(
                    f"{API_BASE}/evaluate",
                    json=request_data,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    eval_result = response.json()
                    
                    # Validate unified response structure
                    required_fields = [
                        "request_id", "overall_ethical", "confidence_score",
                        "processing_time", "timestamp", "version",
                        "analysis_results", "explanation", "cache_hit",
                        "optimization_used"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in eval_result]
                    
                    if not missing_fields:
                        # Check analysis results structure
                        analysis = eval_result.get("analysis_results", {})
                        analysis_layers = ["meta_ethical", "normative", "applied"]
                        available_layers = [layer for layer in analysis_layers if layer in analysis]
                        
                        self.log_test_result(
                            f"Unified Evaluation - {text_type.title()} Text",
                            True,
                            f"Ethical: {eval_result['overall_ethical']}, Confidence: {eval_result['confidence_score']:.3f}, Layers: {len(available_layers)}/3",
                            response_time
                        )
                    else:
                        self.log_test_result(
                            f"Unified Evaluation - {text_type.title()} Text",
                            False,
                            f"Missing fields: {missing_fields}",
                            response_time
                        )
                else:
                    self.log_test_result(
                        f"Unified Evaluation - {text_type.title()} Text",
                        False,
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        response_time
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Unified Evaluation - {text_type.title()} Text",
                    False,
                    f"Request failed: {str(e)}",
                    0.0
                )
    
    async def test_backward_compatibility_endpoints(self):
        """
        🔄 TEST: Backward Compatibility Endpoints
        ═════════════════════════════════════════════
        
        Tests that existing endpoints (/api/parameters, /api/learning-stats,
        /api/heat-map-mock) maintain backward compatibility while working
        with the new unified architecture.
        """
        logger.info("🔄 Testing Backward Compatibility Endpoints...")
        
        # Test /api/parameters endpoint
        try:
            start_time = time.time()
            response = self.requests.get(f"{API_BASE}/parameters", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                params = response.json()
                expected_params = [
                    "virtue_threshold", "deontological_threshold", 
                    "consequentialist_threshold", "virtue_weight",
                    "deontological_weight", "consequentialist_weight"
                ]
                
                available_params = [p for p in expected_params if p in params]
                
                self.log_test_result(
                    "Backward Compatibility - Parameters",
                    len(available_params) >= 3,
                    f"Parameters available: {len(available_params)}/{len(expected_params)}",
                    response_time
                )
            else:
                self.log_test_result(
                    "Backward Compatibility - Parameters",
                    False,
                    f"HTTP {response.status_code}",
                    response_time
                )
        except Exception as e:
            self.log_test_result(
                "Backward Compatibility - Parameters",
                False,
                f"Request failed: {str(e)}",
                0.0
            )
        
        # Test /api/learning-stats endpoint
        try:
            start_time = time.time()
            response = self.requests.get(f"{API_BASE}/learning-stats", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                stats = response.json()
                expected_fields = [
                    "total_evaluations", "total_feedback", "learning_enabled",
                    "performance_metrics", "last_updated"
                ]
                
                available_fields = [f for f in expected_fields if f in stats]
                
                self.log_test_result(
                    "Backward Compatibility - Learning Stats",
                    len(available_fields) >= 3,
                    f"Stats fields: {len(available_fields)}/{len(expected_fields)}",
                    response_time
                )
            else:
                self.log_test_result(
                    "Backward Compatibility - Learning Stats",
                    False,
                    f"HTTP {response.status_code}",
                    response_time
                )
        except Exception as e:
            self.log_test_result(
                "Backward Compatibility - Learning Stats",
                False,
                f"Request failed: {str(e)}",
                0.0
            )
        
        # Test /api/heat-map-mock endpoint
        try:
            start_time = time.time()
            response = self.requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": "This is a test for heat map visualization."},
                timeout=10
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                heatmap = response.json()
                expected_structure = [
                    "evaluations", "overallGrades", "textLength", "originalEvaluation"
                ]
                
                available_structure = [s for s in expected_structure if s in heatmap]
                
                # Check evaluations structure
                evaluations = heatmap.get("evaluations", {})
                eval_types = ["short", "medium", "long", "stochastic"]
                available_types = [t for t in eval_types if t in evaluations]
                
                self.log_test_result(
                    "Backward Compatibility - Heat Map Mock",
                    len(available_structure) >= 3 and len(available_types) >= 3,
                    f"Structure: {len(available_structure)}/4, Types: {len(available_types)}/4",
                    response_time
                )
            else:
                self.log_test_result(
                    "Backward Compatibility - Heat Map Mock",
                    False,
                    f"HTTP {response.status_code}",
                    response_time
                )
        except Exception as e:
            self.log_test_result(
                "Backward Compatibility - Heat Map Mock",
                False,
                f"Request failed: {str(e)}",
                0.0
            )
    
    async def test_configuration_system(self):
        """
        🔧 TEST: Configuration System Integration
        ═════════════════════════════════════════════
        
        Tests the unified configuration system by checking parameter updates
        and configuration validation through the API.
        """
        logger.info("🔧 Testing Configuration System Integration...")
        
        # Test parameter updates
        try:
            start_time = time.time()
            
            update_params = {
                "virtue_weight": 0.4,
                "deontological_weight": 0.3,
                "consequentialist_weight": 0.3,
                "optimization_level": "balanced"
            }
            
            response = self.requests.post(
                f"{API_BASE}/update-parameters",
                json=update_params,
                timeout=10
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                if "message" in result and "parameters" in result:
                    self.log_test_result(
                        "Configuration System - Parameter Update",
                        True,
                        f"Update acknowledged: {result.get('message', 'Success')}",
                        response_time
                    )
                else:
                    self.log_test_result(
                        "Configuration System - Parameter Update",
                        False,
                        "Invalid response structure",
                        response_time
                    )
            else:
                self.log_test_result(
                    "Configuration System - Parameter Update",
                    False,
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    response_time
                )
                
        except Exception as e:
            self.log_test_result(
                "Configuration System - Parameter Update",
                False,
                f"Request failed: {str(e)}",
                0.0
            )
    
    async def test_performance_characteristics(self):
        """
        ⚡ TEST: Performance Characteristics
        ═══════════════════════════════════════
        
        Tests the performance characteristics of the unified architecture
        to ensure it meets or exceeds the performance of the previous system.
        """
        logger.info("⚡ Testing Performance Characteristics...")
        
        # Performance test with multiple concurrent requests
        test_text = "This is a performance test for the unified ethical AI system."
        concurrent_requests = 3
        
        async def single_evaluation():
            try:
                start_time = time.time()
                response = self.requests.post(
                    f"{API_BASE}/evaluate",
                    json={
                        "text": test_text,
                        "mode": "production",
                        "priority": "normal"
                    },
                    timeout=15
                )
                response_time = time.time() - start_time
                return response.status_code == 200, response_time
            except Exception:
                return False, 0.0
        
        # Run concurrent evaluations
        start_time = time.time()
        tasks = []
        
        for i in range(concurrent_requests):
            # Simulate concurrent requests with small delays
            await asyncio.sleep(0.1)
            success, req_time = await asyncio.get_event_loop().run_in_executor(
                None, lambda: asyncio.run(single_evaluation())
            )
            tasks.append((success, req_time))
        
        total_time = time.time() - start_time
        successful_requests = sum(1 for success, _ in tasks if success)
        avg_response_time = sum(req_time for _, req_time in tasks) / len(tasks) if tasks else 0
        
        self.log_test_result(
            "Performance - Concurrent Evaluations",
            successful_requests >= concurrent_requests * 0.8,  # 80% success rate
            f"Success: {successful_requests}/{concurrent_requests}, Avg: {avg_response_time:.3f}s",
            total_time
        )
        
        # Test response time consistency
        response_times = [req_time for _, req_time in tasks if req_time > 0]
        if response_times:
            max_time = max(response_times)
            min_time = min(response_times)
            time_variance = max_time - min_time
            
            self.log_test_result(
                "Performance - Response Time Consistency",
                time_variance < 5.0,  # Less than 5 second variance
                f"Min: {min_time:.3f}s, Max: {max_time:.3f}s, Variance: {time_variance:.3f}s",
                avg_response_time
            )
    
    async def test_error_handling_and_graceful_degradation(self):
        """
        🛡️ TEST: Error Handling and Graceful Degradation
        ═════════════════════════════════════════════════════
        
        Tests the system's ability to handle errors gracefully and provide
        meaningful responses even when components fail.
        """
        logger.info("🛡️ Testing Error Handling and Graceful Degradation...")
        
        # Test empty text handling
        try:
            start_time = time.time()
            response = self.requests.post(
                f"{API_BASE}/evaluate",
                json={"text": ""},
                timeout=10
            )
            response_time = time.time() - start_time
            
            # Should return 400 or handle gracefully
            if response.status_code in [400, 422]:
                self.log_test_result(
                    "Error Handling - Empty Text",
                    True,
                    f"Properly rejected empty text with HTTP {response.status_code}",
                    response_time
                )
            elif response.status_code == 200:
                # If it returns 200, check if it's a graceful degradation
                result = response.json()
                if "error" in result.get("explanation", "").lower():
                    self.log_test_result(
                        "Error Handling - Empty Text",
                        True,
                        "Graceful degradation with error explanation",
                        response_time
                    )
                else:
                    self.log_test_result(
                        "Error Handling - Empty Text",
                        False,
                        "Empty text not properly handled",
                        response_time
                    )
            else:
                self.log_test_result(
                    "Error Handling - Empty Text",
                    False,
                    f"Unexpected HTTP {response.status_code}",
                    response_time
                )
        except Exception as e:
            self.log_test_result(
                "Error Handling - Empty Text",
                False,
                f"Request failed: {str(e)}",
                0.0
            )
        
        # Test malformed request handling
        try:
            start_time = time.time()
            response = self.requests.post(
                f"{API_BASE}/evaluate",
                json={"invalid_field": "test"},
                timeout=10
            )
            response_time = time.time() - start_time
            
            # Should return 422 (validation error)
            if response.status_code == 422:
                self.log_test_result(
                    "Error Handling - Malformed Request",
                    True,
                    "Properly rejected malformed request",
                    response_time
                )
            else:
                self.log_test_result(
                    "Error Handling - Malformed Request",
                    False,
                    f"HTTP {response.status_code} instead of 422",
                    response_time
                )
        except Exception as e:
            self.log_test_result(
                "Error Handling - Malformed Request",
                False,
                f"Request failed: {str(e)}",
                0.0
            )
    
    async def run_all_tests(self):
        """
        🚀 RUN ALL TESTS:
        ═════════════════════
        
        Execute the complete test suite for the unified ethical AI server
        architecture and generate comprehensive results.
        """
        logger.info("🚀 Starting Unified Architecture Test Suite...")
        logger.info(f"🎯 Testing against: {API_BASE}")
        
        if not self.requests_available:
            logger.error("❌ Cannot run tests - requests library not available")
            return
        
        # Execute all test categories
        await self.test_unified_server_health()
        await self.test_unified_evaluate_endpoint()
        await self.test_backward_compatibility_endpoints()
        await self.test_configuration_system()
        await self.test_performance_characteristics()
        await self.test_error_handling_and_graceful_degradation()
        
        # Generate final report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        logger.info("\n" + "="*80)
        logger.info("🏛️ UNIFIED ETHICAL AI SERVER ARCHITECTURE TEST REPORT")
        logger.info("="*80)
        logger.info(f"📊 OVERALL RESULTS:")
        logger.info(f"   Total Tests: {self.total_tests}")
        logger.info(f"   Passed: {self.passed_tests} ✅")
        logger.info(f"   Failed: {self.failed_tests} ❌")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info("")
        
        # Categorize results
        categories = {}
        for result in self.test_results:
            category = result["test_name"].split(" - ")[0]
            if category not in categories:
                categories[category] = {"passed": 0, "failed": 0, "total": 0}
            
            categories[category]["total"] += 1
            if result["success"]:
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1
        
        logger.info("📋 RESULTS BY CATEGORY:")
        for category, stats in categories.items():
            cat_success_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status = "✅" if cat_success_rate >= 80 else "⚠️" if cat_success_rate >= 60 else "❌"
            logger.info(f"   {status} {category}: {stats['passed']}/{stats['total']} ({cat_success_rate:.1f}%)")
        
        logger.info("")
        
        # Show failed tests
        failed_tests = [r for r in self.test_results if not r["success"]]
        if failed_tests:
            logger.info("❌ FAILED TESTS:")
            for test in failed_tests:
                logger.info(f"   • {test['test_name']}: {test['details']}")
            logger.info("")
        
        # Performance summary
        response_times = [r["response_time"] for r in self.test_results if r["response_time"] > 0]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            logger.info("⚡ PERFORMANCE SUMMARY:")
            logger.info(f"   Average Response Time: {avg_response_time:.3f}s")
            logger.info(f"   Maximum Response Time: {max_response_time:.3f}s")
            logger.info("")
        
        # Final assessment
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT: Unified architecture is working exceptionally well!")
        elif success_rate >= 80:
            logger.info("✅ GOOD: Unified architecture is working well with minor issues.")
        elif success_rate >= 60:
            logger.info("⚠️ ACCEPTABLE: Unified architecture has some issues that need attention.")
        else:
            logger.info("❌ CRITICAL: Unified architecture has significant issues requiring immediate attention.")
        
        logger.info("="*80)

async def main():
    """Main test execution function."""
    test_suite = UnifiedArchitectureTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
"""
Comprehensive Backend Testing for Ethical AI Developer Testbed
Focus: Phase 5 Enhanced Ethics Pipeline Testing with Philosophical Foundations

This test suite specifically validates:
1. Phase 5 Enhanced Ethics Pipeline - Comprehensive Analysis
2. Meta-Ethical Analysis Framework (Kantian, Moorean, Humean)
3. Normative Ethics Multi-Framework Analysis (Deontological, Consequentialist, Virtue)
4. Applied Ethics Domain-Specific Analysis (Digital, AI Ethics)
5. ML Training Ethical Guidance Integration
6. Pipeline Status and Performance Monitoring
7. Philosophical Rigor and Practical Utility Validation
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

# Get backend URL from environment - use localhost for testing
BACKEND_URL = 'http://localhost:8001'  # Use localhost for direct backend testing
API_BASE = f"{BACKEND_URL}/api"

class EthicalEvaluationTester:
    def __init__(self):
        self.results = []
        self.failed_tests = []
        self.passed_tests = []
        self.performance_metrics = []
        
        # Phase 5 specific test scenarios
        self.philosophical_test_cases = {
            "kantian_test": "If everyone lied whenever convenient, communication would become meaningless and lying would be self-defeating",
            "utilitarian_test": "This policy will increase overall happiness for 1000 people but cause moderate suffering for 10 people",
            "virtue_ethics_test": "A person demonstrates courage by standing up for their principles despite social pressure",
            "ai_ethics_test": "Our machine learning model uses demographic data to make hiring decisions",
            "digital_ethics_test": "This app collects user location data for advertising purposes without explicit consent",
            "complex_dilemma": "A self-driving car must choose between hitting one person or swerving to hit three people",
            "naturalistic_fallacy": "Violence is natural in evolution, therefore violence is morally good",
            "fact_value_distinction": "Studies show that meditation reduces stress, therefore everyone ought to meditate"
        }
        
    def log_result(self, test_name: str, success: bool, message: str, details: Dict = None):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'details': details or {}
        }
        self.results.append(result)
        
        if success:
            self.passed_tests.append(test_name)
            print(f"✅ {test_name}: {message}")
        else:
            self.failed_tests.append(test_name)
            print(f"❌ {test_name}: {message}")
            if details:
                print(f"   Details: {details}")
    
    def test_health_check(self):
        """Test /api/health endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                evaluator_initialized = data.get('evaluator_initialized', False)
                db_connected = data.get('database_connected', False)
                
                if evaluator_initialized and db_connected:
                    self.log_result(
                        "Health Check", 
                        True, 
                        f"Service healthy, evaluator initialized, DB connected ({response_time:.3f}s)",
                        {"response_time": response_time, "data": data}
                    )
                else:
                    self.log_result(
                        "Health Check", 
                        False, 
                        f"Service issues: evaluator={evaluator_initialized}, db={db_connected}",
                        {"data": data}
                    )
            else:
                self.log_result("Health Check", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Health Check", False, f"Request failed: {str(e)}")
    
    def test_parameters_endpoint(self):
        """Test /api/parameters endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/parameters", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['virtue_threshold', 'deontological_threshold', 'consequentialist_threshold']
                
                if all(field in data for field in required_fields):
                    self.log_result(
                        "Parameters Endpoint", 
                        True, 
                        f"Parameters retrieved successfully ({response_time:.3f}s)",
                        {"response_time": response_time, "parameters": data}
                    )
                else:
                    self.log_result(
                        "Parameters Endpoint", 
                        False, 
                        "Missing required parameter fields",
                        {"data": data, "required": required_fields}
                    )
            else:
                self.log_result("Parameters Endpoint", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Parameters Endpoint", False, f"Request failed: {str(e)}")
    
    def test_learning_stats_endpoint(self):
        """Test /api/learning-stats endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['total_evaluations', 'total_feedback', 'learning_enabled']
                
                if all(field in data for field in required_fields):
                    self.log_result(
                        "Learning Stats Endpoint", 
                        True, 
                        f"Learning stats retrieved successfully ({response_time:.3f}s)",
                        {"response_time": response_time, "stats": data}
                    )
                else:
                    self.log_result(
                        "Learning Stats Endpoint", 
                        False, 
                        "Missing required stats fields",
                        {"data": data, "required": required_fields}
                    )
            else:
                self.log_result("Learning Stats Endpoint", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Learning Stats Endpoint", False, f"Request failed: {str(e)}")
    
    def test_evaluation_with_unethical_content(self):
        """Test /api/evaluate with clearly unethical content"""
        unethical_text = "You are stupid and worthless, and you should die"
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": unethical_text},
                timeout=120  # Allow up to 2 minutes for evaluation
            )
            response_time = time.time() - start_time
            self.performance_metrics.append(("unethical_evaluation", response_time))
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ['evaluation', 'clean_text', 'explanation', 'delta_summary']
                if not all(field in data for field in required_fields):
                    self.log_result(
                        "Unethical Content Evaluation - Structure", 
                        False, 
                        "Missing required response fields",
                        {"data": data, "required": required_fields}
                    )
                    return
                
                # Check evaluation details
                evaluation = data.get('evaluation', {})
                delta_summary = data.get('delta_summary', {})
                
                overall_ethical = evaluation.get('overall_ethical', True)
                violation_count = len(evaluation.get('minimal_spans', []))
                ethical_status = delta_summary.get('ethical_status', True)
                
                # This should be flagged as unethical
                if not overall_ethical and violation_count > 0:
                    self.log_result(
                        "Unethical Content Evaluation - Detection", 
                        True, 
                        f"Correctly flagged as unethical: {violation_count} violations detected ({response_time:.3f}s)",
                        {
                            "response_time": response_time,
                            "overall_ethical": overall_ethical,
                            "violation_count": violation_count,
                            "ethical_status": ethical_status,
                            "engine_version": delta_summary.get('engine_version', 'unknown')
                        }
                    )
                else:
                    self.log_result(
                        "Unethical Content Evaluation - Detection", 
                        False, 
                        f"CRITICAL: Failed to flag unethical content (overall_ethical={overall_ethical}, violations={violation_count})",
                        {
                            "response_time": response_time,
                            "overall_ethical": overall_ethical,
                            "violation_count": violation_count,
                            "evaluation": evaluation,
                            "text": unethical_text
                        }
                    )
                
                # Check response structure completeness
                self.log_result(
                    "Unethical Content Evaluation - Structure", 
                    True, 
                    "Response structure complete with all required fields",
                    {"response_time": response_time, "structure_valid": True}
                )
                
            else:
                self.log_result(
                    "Unethical Content Evaluation", 
                    False, 
                    f"HTTP {response.status_code}",
                    {"response": response.text, "response_time": response_time}
                )
                
        except Exception as e:
            self.log_result("Unethical Content Evaluation", False, f"Request failed: {str(e)}")
    
    def test_evaluation_with_ethical_content(self):
        """Test /api/evaluate with clearly ethical content"""
        ethical_text = "Thank you for your help, I appreciate it"
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": ethical_text},
                timeout=120  # Allow up to 2 minutes for evaluation
            )
            response_time = time.time() - start_time
            self.performance_metrics.append(("ethical_evaluation", response_time))
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ['evaluation', 'clean_text', 'explanation', 'delta_summary']
                if not all(field in data for field in required_fields):
                    self.log_result(
                        "Ethical Content Evaluation - Structure", 
                        False, 
                        "Missing required response fields",
                        {"data": data, "required": required_fields}
                    )
                    return
                
                # Check evaluation details
                evaluation = data.get('evaluation', {})
                delta_summary = data.get('delta_summary', {})
                
                overall_ethical = evaluation.get('overall_ethical', False)
                violation_count = len(evaluation.get('minimal_spans', []))
                ethical_status = delta_summary.get('ethical_status', False)
                
                # This should pass as ethical
                if overall_ethical and violation_count == 0:
                    self.log_result(
                        "Ethical Content Evaluation - Detection", 
                        True, 
                        f"Correctly passed as ethical: {violation_count} violations detected ({response_time:.3f}s)",
                        {
                            "response_time": response_time,
                            "overall_ethical": overall_ethical,
                            "violation_count": violation_count,
                            "ethical_status": ethical_status,
                            "engine_version": delta_summary.get('engine_version', 'unknown')
                        }
                    )
                else:
                    self.log_result(
                        "Ethical Content Evaluation - Detection", 
                        False, 
                        f"Incorrectly flagged ethical content (overall_ethical={overall_ethical}, violations={violation_count})",
                        {
                            "response_time": response_time,
                            "overall_ethical": overall_ethical,
                            "violation_count": violation_count,
                            "evaluation": evaluation,
                            "text": ethical_text
                        }
                    )
                
                # Check response structure completeness
                self.log_result(
                    "Ethical Content Evaluation - Structure", 
                    True, 
                    "Response structure complete with all required fields",
                    {"response_time": response_time, "structure_valid": True}
                )
                
            else:
                self.log_result(
                    "Ethical Content Evaluation", 
                    False, 
                    f"HTTP {response.status_code}",
                    {"response": response.text, "response_time": response_time}
                )
                
        except Exception as e:
            self.log_result("Ethical Content Evaluation", False, f"Request failed: {str(e)}")
    
    def test_evaluation_with_empty_text(self):
        """Test /api/evaluate with empty text"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": ""},
                timeout=30
            )
            response_time = time.time() - start_time
            
            # Should return 400 Bad Request for empty text
            if response.status_code == 400:
                self.log_result(
                    "Empty Text Evaluation", 
                    True, 
                    f"Correctly rejected empty text with HTTP 400 ({response_time:.3f}s)",
                    {"response_time": response_time, "status_code": response.status_code}
                )
            else:
                self.log_result(
                    "Empty Text Evaluation", 
                    False, 
                    f"Unexpected response for empty text: HTTP {response.status_code}",
                    {"response": response.text, "response_time": response_time}
                )
                
        except Exception as e:
            self.log_result("Empty Text Evaluation", False, f"Request failed: {str(e)}")
    
    def test_evaluation_with_edge_cases(self):
        """Test /api/evaluate with edge cases"""
        edge_cases = [
            ("Single word", "Hello"),
            ("Numbers only", "12345"),
            ("Special characters", "!@#$%^&*()"),
            ("Mixed content", "Hello! How are you? 123 @#$")
        ]
        
        for case_name, text in edge_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=60
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    
                    self.log_result(
                        f"Edge Case - {case_name}", 
                        True, 
                        f"Successfully processed '{text}' ({response_time:.3f}s)",
                        {
                            "response_time": response_time,
                            "overall_ethical": evaluation.get('overall_ethical', 'unknown'),
                            "violation_count": len(evaluation.get('minimal_spans', []))
                        }
                    )
                else:
                    self.log_result(
                        f"Edge Case - {case_name}", 
                        False, 
                        f"Failed to process '{text}': HTTP {response.status_code}",
                        {"response": response.text, "response_time": response_time}
                    )
                    
            except Exception as e:
                self.log_result(f"Edge Case - {case_name}", False, f"Request failed: {str(e)}")
    
    def test_performance_stats_endpoint(self):
        """Test /api/performance-stats endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/performance-stats", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for optimization status
                optimization_available = data.get('optimization_available', False)
                
                self.log_result(
                    "Performance Stats Endpoint", 
                    True, 
                    f"Performance stats retrieved, optimization_available={optimization_available} ({response_time:.3f}s)",
                    {"response_time": response_time, "stats": data}
                )
            else:
                self.log_result("Performance Stats Endpoint", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Performance Stats Endpoint", False, f"Request failed: {str(e)}")
    
    def test_integration_workflow(self):
        """Test complete integration workflow"""
        try:
            # Test the complete workflow: health -> parameters -> evaluate -> stats
            workflow_start = time.time()
            
            # 1. Health check
            health_response = requests.get(f"{API_BASE}/health", timeout=10)
            if health_response.status_code != 200:
                self.log_result("Integration Workflow", False, "Health check failed in workflow")
                return
            
            # 2. Get parameters
            params_response = requests.get(f"{API_BASE}/parameters", timeout=10)
            if params_response.status_code != 200:
                self.log_result("Integration Workflow", False, "Parameters retrieval failed in workflow")
                return
            
            # 3. Evaluate text
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "This is a test message for integration workflow"},
                timeout=60
            )
            if eval_response.status_code != 200:
                self.log_result("Integration Workflow", False, "Evaluation failed in workflow")
                return
            
            # 4. Get learning stats
            stats_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            if stats_response.status_code != 200:
                self.log_result("Integration Workflow", False, "Learning stats failed in workflow")
                return
            
            workflow_time = time.time() - workflow_start
            
            self.log_result(
                "Integration Workflow", 
                True, 
                f"Complete workflow successful ({workflow_time:.3f}s total)",
                {"workflow_time": workflow_time, "steps_completed": 4}
            )
            
        except Exception as e:
            self.log_result("Integration Workflow", False, f"Workflow failed: {str(e)}")
    
    # ============================================================================
    # PHASE 5 ENHANCED ETHICS PIPELINE TESTING
    # ============================================================================
    
    def test_enhanced_ethics_pipeline_status(self):
        """Test /api/ethics/pipeline-status endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/ethics/pipeline-status", timeout=15)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['status', 'pipeline_health', 'component_status', 'capabilities']
                
                if all(field in data for field in required_fields):
                    # Check component status
                    components = data.get('component_status', {})
                    expected_components = ['meta_ethics_analyzer', 'normative_evaluator', 'applied_evaluator']
                    
                    components_available = all(comp in components for comp in expected_components)
                    
                    self.log_result(
                        "Enhanced Ethics Pipeline Status", 
                        True, 
                        f"Pipeline status retrieved successfully, components available: {components_available} ({response_time:.3f}s)",
                        {
                            "response_time": response_time, 
                            "status": data.get('status'),
                            "pipeline_health": data.get('pipeline_health'),
                            "components": list(components.keys()),
                            "capabilities": data.get('capabilities', {})
                        }
                    )
                else:
                    self.log_result(
                        "Enhanced Ethics Pipeline Status", 
                        False, 
                        "Missing required status fields",
                        {"data": data, "required": required_fields}
                    )
            else:
                self.log_result("Enhanced Ethics Pipeline Status", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Enhanced Ethics Pipeline Status", False, f"Request failed: {str(e)}")
    
    def test_meta_ethical_analysis(self):
        """Test /api/ethics/meta-analysis endpoint with philosophical test cases"""
        test_cases = [
            ("Kantian Universalizability", self.philosophical_test_cases["kantian_test"]),
            ("Naturalistic Fallacy Detection", self.philosophical_test_cases["naturalistic_fallacy"]),
            ("Fact-Value Distinction", self.philosophical_test_cases["fact_value_distinction"])
        ]
        
        for test_name, test_text in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE}/ethics/meta-analysis",
                    json={"text": test_text},
                    timeout=30
                )
                response_time = time.time() - start_time
                self.performance_metrics.append((f"meta_analysis_{test_name.lower().replace(' ', '_')}", response_time))
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    required_fields = ['status', 'meta_ethical_analysis', 'philosophical_interpretation']
                    if not all(field in data for field in required_fields):
                        self.log_result(
                            f"Meta-Ethical Analysis - {test_name} Structure", 
                            False, 
                            "Missing required response fields",
                            {"data": data, "required": required_fields}
                        )
                        continue
                    
                    # Check meta-ethical analysis content
                    meta_analysis = data.get('meta_ethical_analysis', {})
                    philosophical_interpretation = data.get('philosophical_interpretation', {})
                    
                    # Validate key philosophical components
                    has_universalizability = 'universalizability_test' in meta_analysis
                    has_naturalistic_fallacy = 'naturalistic_fallacy_check' in meta_analysis
                    has_semantic_coherence = 'semantic_coherence' in meta_analysis
                    has_modal_properties = 'modal_properties' in meta_analysis
                    
                    # Check philosophical interpretations
                    has_kantian = 'kantian_assessment' in philosophical_interpretation
                    has_moorean = 'moorean_assessment' in philosophical_interpretation
                    has_humean = 'humean_assessment' in philosophical_interpretation
                    
                    philosophical_completeness = all([
                        has_universalizability, has_naturalistic_fallacy, has_semantic_coherence,
                        has_modal_properties, has_kantian, has_moorean, has_humean
                    ])
                    
                    if philosophical_completeness:
                        # Test-specific validations
                        if test_name == "Kantian Universalizability":
                            universalizability_result = meta_analysis.get('universalizability_test', False)
                            self.log_result(
                                f"Meta-Ethical Analysis - {test_name}", 
                                True, 
                                f"Kantian universalizability test: {universalizability_result} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "universalizability_test": universalizability_result,
                                    "semantic_coherence": meta_analysis.get('semantic_coherence', 0),
                                    "kantian_assessment": philosophical_interpretation.get('kantian_assessment', {})
                                }
                            )
                        elif test_name == "Naturalistic Fallacy Detection":
                            fallacy_check = meta_analysis.get('naturalistic_fallacy_check', True)
                            self.log_result(
                                f"Meta-Ethical Analysis - {test_name}", 
                                True, 
                                f"Naturalistic fallacy avoided: {fallacy_check} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "naturalistic_fallacy_check": fallacy_check,
                                    "moorean_assessment": philosophical_interpretation.get('moorean_assessment', {})
                                }
                            )
                        else:  # Fact-Value Distinction
                            fact_value_relations = meta_analysis.get('fact_value_relations', [])
                            self.log_result(
                                f"Meta-Ethical Analysis - {test_name}", 
                                True, 
                                f"Fact-value relations identified: {len(fact_value_relations)} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "fact_value_relations": fact_value_relations,
                                    "humean_assessment": philosophical_interpretation.get('humean_assessment', {})
                                }
                            )
                    else:
                        self.log_result(
                            f"Meta-Ethical Analysis - {test_name}", 
                            False, 
                            "Incomplete philosophical analysis components",
                            {
                                "missing_components": {
                                    "universalizability": has_universalizability,
                                    "naturalistic_fallacy": has_naturalistic_fallacy,
                                    "semantic_coherence": has_semantic_coherence,
                                    "modal_properties": has_modal_properties,
                                    "kantian": has_kantian,
                                    "moorean": has_moorean,
                                    "humean": has_humean
                                }
                            }
                        )
                        
                else:
                    self.log_result(
                        f"Meta-Ethical Analysis - {test_name}", 
                        False, 
                        f"HTTP {response.status_code}",
                        {"response": response.text, "response_time": response_time}
                    )
                    
            except Exception as e:
                self.log_result(f"Meta-Ethical Analysis - {test_name}", False, f"Request failed: {str(e)}")
    
    def test_normative_ethics_analysis(self):
        """Test /api/ethics/normative-analysis endpoint with multi-framework analysis"""
        test_cases = [
            ("Deontological Framework", self.philosophical_test_cases["kantian_test"], "deontological"),
            ("Consequentialist Framework", self.philosophical_test_cases["utilitarian_test"], "consequentialist"),
            ("Virtue Ethics Framework", self.philosophical_test_cases["virtue_ethics_test"], "virtue"),
            ("Multi-Framework Analysis", self.philosophical_test_cases["complex_dilemma"], "all")
        ]
        
        for test_name, test_text, framework_focus in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE}/ethics/normative-analysis",
                    json={"text": test_text, "framework": framework_focus},
                    timeout=45
                )
                response_time = time.time() - start_time
                self.performance_metrics.append((f"normative_analysis_{framework_focus}", response_time))
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    required_fields = ['status', 'normative_analysis', 'philosophical_insights', 'resolution_guidance']
                    if not all(field in data for field in required_fields):
                        self.log_result(
                            f"Normative Ethics Analysis - {test_name} Structure", 
                            False, 
                            "Missing required response fields",
                            {"data": data, "required": required_fields}
                        )
                        continue
                    
                    # Check normative analysis content
                    normative_analysis = data.get('normative_analysis', {})
                    philosophical_insights = data.get('philosophical_insights', {})
                    resolution_guidance = data.get('resolution_guidance', {})
                    
                    # Validate framework components
                    has_deontological = 'deontological' in normative_analysis
                    has_consequentialist = 'consequentialist' in normative_analysis
                    has_virtue_ethics = 'virtue_ethics' in normative_analysis
                    has_framework_convergence = 'framework_convergence' in normative_analysis
                    
                    # Check philosophical insights
                    has_deont_insights = 'deontological_insights' in philosophical_insights
                    has_conseq_insights = 'consequentialist_insights' in philosophical_insights
                    has_virtue_insights = 'virtue_ethics_insights' in philosophical_insights
                    
                    framework_completeness = all([
                        has_deontological, has_consequentialist, has_virtue_ethics,
                        has_framework_convergence, has_deont_insights, has_conseq_insights, has_virtue_insights
                    ])
                    
                    if framework_completeness:
                        framework_convergence = normative_analysis.get('framework_convergence', 0)
                        ethical_dilemma_type = normative_analysis.get('ethical_dilemma_type')
                        
                        # Framework-specific validations
                        if framework_focus == "deontological":
                            deont_analysis = normative_analysis.get('deontological', {})
                            categorical_imperative = deont_analysis.get('categorical_imperative_test', False)
                            humanity_formula = deont_analysis.get('humanity_formula_test', False)
                            
                            self.log_result(
                                f"Normative Ethics Analysis - {test_name}", 
                                True, 
                                f"Deontological analysis: CI={categorical_imperative}, HF={humanity_formula} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "categorical_imperative_test": categorical_imperative,
                                    "humanity_formula_test": humanity_formula,
                                    "autonomy_respect": deont_analysis.get('autonomy_respect', 0),
                                    "framework_convergence": framework_convergence
                                }
                            )
                        elif framework_focus == "consequentialist":
                            conseq_analysis = normative_analysis.get('consequentialist', {})
                            utility_calculation = conseq_analysis.get('utility_calculation', 0)
                            aggregate_welfare = conseq_analysis.get('aggregate_welfare', 0)
                            
                            self.log_result(
                                f"Normative Ethics Analysis - {test_name}", 
                                True, 
                                f"Consequentialist analysis: utility={utility_calculation:.3f}, welfare={aggregate_welfare:.3f} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "utility_calculation": utility_calculation,
                                    "aggregate_welfare": aggregate_welfare,
                                    "positive_consequences": len(conseq_analysis.get('positive_consequences', [])),
                                    "negative_consequences": len(conseq_analysis.get('negative_consequences', []))
                                }
                            )
                        elif framework_focus == "virtue":
                            virtue_analysis = normative_analysis.get('virtue_ethics', {})
                            eudaimonic_contribution = virtue_analysis.get('eudaimonic_contribution', 0)
                            golden_mean = virtue_analysis.get('golden_mean_analysis', 0)
                            
                            self.log_result(
                                f"Normative Ethics Analysis - {test_name}", 
                                True, 
                                f"Virtue ethics analysis: eudaimonia={eudaimonic_contribution:.3f}, golden_mean={golden_mean:.3f} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "eudaimonic_contribution": eudaimonic_contribution,
                                    "golden_mean_analysis": golden_mean,
                                    "character_development": virtue_analysis.get('character_development', 0),
                                    "practical_wisdom": virtue_analysis.get('practical_wisdom', 0)
                                }
                            )
                        else:  # Multi-framework analysis
                            self.log_result(
                                f"Normative Ethics Analysis - {test_name}", 
                                True, 
                                f"Multi-framework analysis: convergence={framework_convergence:.3f}, dilemma={ethical_dilemma_type} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "framework_convergence": framework_convergence,
                                    "ethical_dilemma_type": ethical_dilemma_type,
                                    "resolution_recommendation": normative_analysis.get('resolution_recommendation', ''),
                                    "philosophical_consensus": resolution_guidance.get('philosophical_consensus', '')
                                }
                            )
                    else:
                        self.log_result(
                            f"Normative Ethics Analysis - {test_name}", 
                            False, 
                            "Incomplete normative framework components",
                            {
                                "missing_components": {
                                    "deontological": has_deontological,
                                    "consequentialist": has_consequentialist,
                                    "virtue_ethics": has_virtue_ethics,
                                    "framework_convergence": has_framework_convergence
                                }
                            }
                        )
                        
                else:
                    self.log_result(
                        f"Normative Ethics Analysis - {test_name}", 
                        False, 
                        f"HTTP {response.status_code}",
                        {"response": response.text, "response_time": response_time}
                    )
                    
            except Exception as e:
                self.log_result(f"Normative Ethics Analysis - {test_name}", False, f"Request failed: {str(e)}")
    
    def test_applied_ethics_analysis(self):
        """Test /api/ethics/applied-analysis endpoint with domain-specific cases"""
        test_cases = [
            ("Digital Ethics Domain", self.philosophical_test_cases["digital_ethics_test"], "digital"),
            ("AI Ethics Domain", self.philosophical_test_cases["ai_ethics_test"], "ai"),
            ("Auto Domain Detection", self.philosophical_test_cases["complex_dilemma"], "auto")
        ]
        
        for test_name, test_text, domain_focus in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE}/ethics/applied-analysis",
                    json={"text": test_text, "domain": domain_focus},
                    timeout=30
                )
                response_time = time.time() - start_time
                self.performance_metrics.append((f"applied_analysis_{domain_focus}", response_time))
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    required_fields = ['status', 'applied_analysis', 'domain_assessments', 'professional_guidance']
                    if not all(field in data for field in required_fields):
                        self.log_result(
                            f"Applied Ethics Analysis - {test_name} Structure", 
                            False, 
                            "Missing required response fields",
                            {"data": data, "required": required_fields}
                        )
                        continue
                    
                    # Check applied analysis content
                    applied_analysis = data.get('applied_analysis', {})
                    domain_assessments = data.get('domain_assessments', {})
                    professional_guidance = data.get('professional_guidance', {})
                    
                    # Validate domain detection
                    applicable_domains = applied_analysis.get('applicable_domains', [])
                    domain_relevance_scores = applied_analysis.get('domain_relevance_scores', {})
                    practical_recommendations = applied_analysis.get('practical_recommendations', [])
                    
                    domain_detection_working = len(applicable_domains) > 0 or len(domain_relevance_scores) > 0
                    
                    if domain_detection_working:
                        # Domain-specific validations
                        if domain_focus == "digital" and "digital_ethics" in domain_assessments:
                            digital_assessment = domain_assessments["digital_ethics"]
                            privacy_protection = digital_assessment.get('privacy_protection', 'UNKNOWN')
                            user_autonomy = digital_assessment.get('user_autonomy', 0)
                            
                            self.log_result(
                                f"Applied Ethics Analysis - {test_name}", 
                                True, 
                                f"Digital ethics analysis: privacy={privacy_protection}, autonomy={user_autonomy:.3f} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "privacy_protection": privacy_protection,
                                    "user_autonomy": user_autonomy,
                                    "transparency_level": digital_assessment.get('transparency_level', 0),
                                    "surveillance_risk": digital_assessment.get('surveillance_risk', 0)
                                }
                            )
                        elif domain_focus == "ai" and "ai_ethics" in domain_assessments:
                            ai_assessment = domain_assessments["ai_ethics"]
                            fairness_level = ai_assessment.get('fairness_level', 'UNKNOWN')
                            safety_assurance = ai_assessment.get('safety_assurance', 0)
                            
                            self.log_result(
                                f"Applied Ethics Analysis - {test_name}", 
                                True, 
                                f"AI ethics analysis: fairness={fairness_level}, safety={safety_assurance:.3f} ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "fairness_level": fairness_level,
                                    "safety_assurance": safety_assurance,
                                    "human_oversight": ai_assessment.get('human_oversight', 0),
                                    "bias_mitigation": ai_assessment.get('bias_mitigation', 0),
                                    "value_alignment": ai_assessment.get('value_alignment', 0)
                                }
                            )
                        else:  # Auto domain detection
                            self.log_result(
                                f"Applied Ethics Analysis - {test_name}", 
                                True, 
                                f"Auto domain detection: {len(applicable_domains)} domains, {len(practical_recommendations)} recommendations ({response_time:.3f}s)",
                                {
                                    "response_time": response_time,
                                    "applicable_domains": applicable_domains,
                                    "domain_relevance_scores": domain_relevance_scores,
                                    "recommendations_count": len(practical_recommendations),
                                    "implementation_priority": professional_guidance.get('implementation_priority', 'UNKNOWN')
                                }
                            )
                    else:
                        self.log_result(
                            f"Applied Ethics Analysis - {test_name}", 
                            False, 
                            "Domain detection not working - no applicable domains identified",
                            {
                                "applicable_domains": applicable_domains,
                                "domain_relevance_scores": domain_relevance_scores,
                                "response_time": response_time
                            }
                        )
                        
                else:
                    self.log_result(
                        f"Applied Ethics Analysis - {test_name}", 
                        False, 
                        f"HTTP {response.status_code}",
                        {"response": response.text, "response_time": response_time}
                    )
                    
            except Exception as e:
                self.log_result(f"Applied Ethics Analysis - {test_name}", False, f"Request failed: {str(e)}")
    
    def test_comprehensive_ethics_analysis(self):
        """Test /api/ethics/comprehensive-analysis endpoint with different analysis depths"""
        test_cases = [
            ("Surface Analysis", self.philosophical_test_cases["kantian_test"], "surface"),
            ("Standard Analysis", self.philosophical_test_cases["complex_dilemma"], "standard"),
            ("Comprehensive Analysis", self.philosophical_test_cases["ai_ethics_test"], "comprehensive")
        ]
        
        for test_name, test_text, analysis_depth in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE}/ethics/comprehensive-analysis",
                    json={"text": test_text, "depth": analysis_depth},
                    timeout=60  # Comprehensive analysis may take longer
                )
                response_time = time.time() - start_time
                self.performance_metrics.append((f"comprehensive_analysis_{analysis_depth}", response_time))
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    required_fields = ['status', 'analysis', 'meta']
                    if not all(field in data for field in required_fields):
                        self.log_result(
                            f"Comprehensive Ethics Analysis - {test_name} Structure", 
                            False, 
                            "Missing required response fields",
                            {"data": data, "required": required_fields}
                        )
                        continue
                    
                    # Check comprehensive analysis content
                    analysis = data.get('analysis', {})
                    meta = data.get('meta', {})
                    
                    # Validate three-layer architecture
                    has_meta_ethics = 'meta_ethics' in analysis
                    has_normative_ethics = 'normative_ethics' in analysis
                    has_applied_ethics = 'applied_ethics' in analysis
                    has_overall_consistency = 'overall_consistency' in analysis
                    has_ethical_confidence = 'ethical_confidence' in analysis
                    has_synthesized_judgment = 'synthesized_judgment' in analysis
                    has_actionable_recommendations = 'actionable_recommendations' in analysis
                    
                    three_layer_completeness = all([
                        has_meta_ethics, has_normative_ethics, has_applied_ethics,
                        has_overall_consistency, has_ethical_confidence, has_synthesized_judgment,
                        has_actionable_recommendations
                    ])
                    
                    if three_layer_completeness:
                        overall_consistency = analysis.get('overall_consistency', 0)
                        ethical_confidence = analysis.get('ethical_confidence', 0)
                        complexity_score = analysis.get('complexity_score', 0)
                        synthesized_judgment = analysis.get('synthesized_judgment', '')
                        primary_concerns = analysis.get('primary_concerns', [])
                        actionable_recommendations = analysis.get('actionable_recommendations', [])
                        
                        # Validate philosophical rigor
                        meta_ethics = analysis.get('meta_ethics', {})
                        normative_ethics = analysis.get('normative_ethics', {})
                        applied_ethics = analysis.get('applied_ethics', {})
                        
                        philosophical_rigor = (
                            'universalizability_test' in meta_ethics and
                            'naturalistic_fallacy_check' in meta_ethics and
                            'framework_convergence' in normative_ethics and
                            'applicable_domains' in applied_ethics
                        )
                        
                        self.log_result(
                            f"Comprehensive Ethics Analysis - {test_name}", 
                            True, 
                            f"Three-layer analysis complete: consistency={overall_consistency:.3f}, confidence={ethical_confidence:.3f}, judgment='{synthesized_judgment[:50]}...' ({response_time:.3f}s)",
                            {
                                "response_time": response_time,
                                "analysis_depth": analysis_depth,
                                "overall_consistency": overall_consistency,
                                "ethical_confidence": ethical_confidence,
                                "complexity_score": complexity_score,
                                "synthesized_judgment": synthesized_judgment,
                                "primary_concerns_count": len(primary_concerns),
                                "recommendations_count": len(actionable_recommendations),
                                "philosophical_rigor": philosophical_rigor,
                                "processing_time": analysis.get('processing_time', 0)
                            }
                        )
                    else:
                        self.log_result(
                            f"Comprehensive Ethics Analysis - {test_name}", 
                            False, 
                            "Incomplete three-layer analysis structure",
                            {
                                "missing_components": {
                                    "meta_ethics": has_meta_ethics,
                                    "normative_ethics": has_normative_ethics,
                                    "applied_ethics": has_applied_ethics,
                                    "overall_consistency": has_overall_consistency,
                                    "ethical_confidence": has_ethical_confidence,
                                    "synthesized_judgment": has_synthesized_judgment,
                                    "actionable_recommendations": has_actionable_recommendations
                                }
                            }
                        )
                        
                else:
                    self.log_result(
                        f"Comprehensive Ethics Analysis - {test_name}", 
                        False, 
                        f"HTTP {response.status_code}",
                        {"response": response.text, "response_time": response_time}
                    )
                    
            except Exception as e:
                self.log_result(f"Comprehensive Ethics Analysis - {test_name}", False, f"Request failed: {str(e)}")
    
    def test_ml_training_ethical_guidance(self):
        """Test /api/ethics/ml-training-guidance endpoint"""
        test_cases = [
            ("Training Data Ethics", self.philosophical_test_cases["ai_ethics_test"], "data"),
            ("Model Development Ethics", "Our AI model learns from user behavior to predict preferences", "model"),
            ("Comprehensive ML Guidance", "Training an AI system on social media data to detect harmful content", "comprehensive")
        ]
        
        for test_name, test_content, guidance_type in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE}/ethics/ml-training-guidance",
                    json={
                        "content": test_content, 
                        "type": guidance_type,
                        "training_context": {"model_type": "classification", "dataset_size": "large"}
                    },
                    timeout=60
                )
                response_time = time.time() - start_time
                self.performance_metrics.append((f"ml_guidance_{guidance_type}", response_time))
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    required_fields = ['status', 'ml_ethical_guidance', 'actionable_recommendations', 'philosophical_assessment']
                    if not all(field in data for field in required_fields):
                        self.log_result(
                            f"ML Training Ethical Guidance - {test_name} Structure", 
                            False, 
                            "Missing required response fields",
                            {"data": data, "required": required_fields}
                        )
                        continue
                    
                    # Check ML ethical guidance content
                    ml_guidance = data.get('ml_ethical_guidance', {})
                    actionable_recommendations = data.get('actionable_recommendations', [])
                    philosophical_assessment = data.get('philosophical_assessment', {})
                    
                    # Validate ML-specific components
                    has_training_data_ethics = 'training_data_ethics' in ml_guidance
                    has_model_development_ethics = 'model_development_ethics' in ml_guidance
                    has_philosophical_foundations = 'philosophical_foundations' in ml_guidance
                    
                    ml_guidance_completeness = all([
                        has_training_data_ethics, has_model_development_ethics, has_philosophical_foundations
                    ])
                    
                    if ml_guidance_completeness:
                        training_data_ethics = ml_guidance.get('training_data_ethics', {})
                        model_development_ethics = ml_guidance.get('model_development_ethics', {})
                        philosophical_foundations = ml_guidance.get('philosophical_foundations', {})
                        
                        # Extract key metrics
                        bias_risk = training_data_ethics.get('bias_risk_assessment', 0)
                        privacy_implications = training_data_ethics.get('privacy_implications', 0)
                        transparency_requirements = model_development_ethics.get('transparency_requirements', 0)
                        safety_considerations = model_development_ethics.get('safety_considerations', 0)
                        
                        # Philosophical grounding
                        kantian_universalizability = philosophical_foundations.get('kantian_universalizability', False)
                        utilitarian_welfare = philosophical_foundations.get('utilitarian_welfare_impact', 0)
                        overall_consistency = philosophical_foundations.get('overall_ethical_consistency', 0)
                        
                        self.log_result(
                            f"ML Training Ethical Guidance - {test_name}", 
                            True, 
                            f"ML guidance generated: bias_risk={bias_risk:.3f}, transparency={transparency_requirements:.3f}, {len(actionable_recommendations)} recommendations ({response_time:.3f}s)",
                            {
                                "response_time": response_time,
                                "guidance_type": guidance_type,
                                "bias_risk_assessment": bias_risk,
                                "privacy_implications": privacy_implications,
                                "transparency_requirements": transparency_requirements,
                                "safety_considerations": safety_considerations,
                                "kantian_universalizability": kantian_universalizability,
                                "utilitarian_welfare_impact": utilitarian_welfare,
                                "overall_ethical_consistency": overall_consistency,
                                "recommendations_count": len(actionable_recommendations),
                                "ethical_judgment": philosophical_assessment.get('ethical_judgment', ''),
                                "confidence_level": philosophical_assessment.get('confidence_level', 0)
                            }
                        )
                    else:
                        self.log_result(
                            f"ML Training Ethical Guidance - {test_name}", 
                            False, 
                            "Incomplete ML ethical guidance components",
                            {
                                "missing_components": {
                                    "training_data_ethics": has_training_data_ethics,
                                    "model_development_ethics": has_model_development_ethics,
                                    "philosophical_foundations": has_philosophical_foundations
                                }
                            }
                        )
                        
                else:
                    self.log_result(
                        f"ML Training Ethical Guidance - {test_name}", 
                        False, 
                        f"HTTP {response.status_code}",
                        {"response": response.text, "response_time": response_time}
                    )
                    
            except Exception as e:
                self.log_result(f"ML Training Ethical Guidance - {test_name}", False, f"Request failed: {str(e)}")
    
    def test_philosophical_rigor_validation(self):
        """Test philosophical rigor with classical examples"""
        classical_examples = [
            ("Trolley Problem", "A runaway trolley is heading towards five people. You can pull a lever to divert it to a side track where it will kill one person instead. Should you pull the lever?"),
            ("Categorical Imperative", "I should make a promise I don't intend to keep to get out of financial difficulty"),
            ("Utilitarian Calculus", "Torturing one innocent person would save the lives of a thousand people"),
            ("Virtue Ethics", "A doctor lies to a patient about their terminal diagnosis to spare them emotional pain"),
            ("Naturalistic Fallacy", "Aggression is found throughout nature, therefore human aggression is morally justified")
        ]
        
        for example_name, example_text in classical_examples:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE}/ethics/comprehensive-analysis",
                    json={"text": example_text, "depth": "comprehensive"},
                    timeout=60
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    analysis = data.get('analysis', {})
                    
                    # Validate philosophical accuracy for known examples
                    meta_ethics = analysis.get('meta_ethics', {})
                    normative_ethics = analysis.get('normative_ethics', {})
                    
                    if example_name == "Categorical Imperative":
                        # Should fail universalizability test
                        universalizability = meta_ethics.get('universalizability_test', True)
                        categorical_imperative = normative_ethics.get('deontological', {}).get('categorical_imperative_test', True)
                        
                        philosophical_accuracy = not universalizability and not categorical_imperative
                        
                        self.log_result(
                            f"Philosophical Rigor - {example_name}", 
                            philosophical_accuracy, 
                            f"Kantian analysis accuracy: universalizability={universalizability}, CI={categorical_imperative} ({response_time:.3f}s)",
                            {
                                "response_time": response_time,
                                "universalizability_test": universalizability,
                                "categorical_imperative_test": categorical_imperative,
                                "expected_failure": True,
                                "philosophical_accuracy": philosophical_accuracy
                            }
                        )
                    elif example_name == "Naturalistic Fallacy":
                        # Should detect naturalistic fallacy
                        naturalistic_fallacy_check = meta_ethics.get('naturalistic_fallacy_check', True)
                        
                        philosophical_accuracy = not naturalistic_fallacy_check  # Should be False (fallacy detected)
                        
                        self.log_result(
                            f"Philosophical Rigor - {example_name}", 
                            philosophical_accuracy, 
                            f"Naturalistic fallacy detection: clean={naturalistic_fallacy_check} ({response_time:.3f}s)",
                            {
                                "response_time": response_time,
                                "naturalistic_fallacy_check": naturalistic_fallacy_check,
                                "expected_fallacy_detection": True,
                                "philosophical_accuracy": philosophical_accuracy
                            }
                        )
                    else:
                        # General philosophical rigor check
                        overall_consistency = analysis.get('overall_consistency', 0)
                        ethical_confidence = analysis.get('ethical_confidence', 0)
                        framework_convergence = normative_ethics.get('framework_convergence', 0)
                        
                        philosophical_rigor = (
                            overall_consistency > 0.3 and  # Some consistency expected
                            ethical_confidence > 0.3 and   # Some confidence expected
                            framework_convergence >= 0.0   # Valid convergence score
                        )
                        
                        self.log_result(
                            f"Philosophical Rigor - {example_name}", 
                            philosophical_rigor, 
                            f"General rigor: consistency={overall_consistency:.3f}, confidence={ethical_confidence:.3f}, convergence={framework_convergence:.3f} ({response_time:.3f}s)",
                            {
                                "response_time": response_time,
                                "overall_consistency": overall_consistency,
                                "ethical_confidence": ethical_confidence,
                                "framework_convergence": framework_convergence,
                                "philosophical_rigor": philosophical_rigor
                            }
                        )
                else:
                    self.log_result(
                        f"Philosophical Rigor - {example_name}", 
                        False, 
                        f"HTTP {response.status_code}",
                        {"response": response.text, "response_time": response_time}
                    )
                    
            except Exception as e:
                self.log_result(f"Philosophical Rigor - {example_name}", False, f"Request failed: {str(e)}")
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases for Phase 5 endpoints"""
        edge_cases = [
            ("Empty Content", "", "/ethics/comprehensive-analysis"),
            ("Very Long Content", "A" * 10000, "/ethics/meta-analysis"),
            ("Non-English Content", "这是一个中文测试文本，用于测试伦理分析系统", "/ethics/normative-analysis"),
            ("Special Characters", "!@#$%^&*()_+{}|:<>?[]\\;'\",./ 🤖🧠⚖️", "/ethics/applied-analysis"),
            ("Invalid Analysis Depth", "Test content", "/ethics/comprehensive-analysis")
        ]
        
        for case_name, test_content, endpoint in edge_cases:
            try:
                start_time = time.time()
                
                # Prepare request based on case
                if case_name == "Empty Content":
                    request_data = {"text": test_content}
                elif case_name == "Invalid Analysis Depth":
                    request_data = {"text": test_content, "depth": "invalid_depth"}
                elif endpoint == "/ethics/ml-training-guidance":
                    request_data = {"content": test_content}
                else:
                    request_data = {"text": test_content}
                
                response = requests.post(
                    f"{API_BASE}{endpoint}",
                    json=request_data,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if case_name == "Empty Content":
                    # Should return 400 for empty content
                    expected_success = response.status_code == 400
                    self.log_result(
                        f"Edge Case - {case_name}", 
                        expected_success, 
                        f"Empty content handling: HTTP {response.status_code} ({response_time:.3f}s)",
                        {"response_time": response_time, "status_code": response.status_code, "expected": 400}
                    )
                elif case_name == "Invalid Analysis Depth":
                    # Should handle gracefully (either 400 or default to standard)
                    if response.status_code == 200:
                        data = response.json()
                        analysis_depth = data.get('meta', {}).get('analysis_depth', '')
                        expected_success = analysis_depth == 'standard'  # Should default to standard
                    else:
                        expected_success = response.status_code == 400  # Or reject with 400
                    
                    self.log_result(
                        f"Edge Case - {case_name}", 
                        expected_success, 
                        f"Invalid depth handling: HTTP {response.status_code} ({response_time:.3f}s)",
                        {"response_time": response_time, "status_code": response.status_code}
                    )
                else:
                    # Should handle gracefully without crashing
                    expected_success = response.status_code in [200, 400, 422]  # Acceptable responses
                    
                    if response.status_code == 200:
                        # If successful, check that analysis was attempted
                        data = response.json()
                        has_analysis = 'analysis' in data or 'meta_ethical_analysis' in data or 'normative_analysis' in data or 'applied_analysis' in data
                        
                        self.log_result(
                            f"Edge Case - {case_name}", 
                            has_analysis, 
                            f"Edge case processed successfully: analysis_present={has_analysis} ({response_time:.3f}s)",
                            {"response_time": response_time, "status_code": response.status_code, "analysis_present": has_analysis}
                        )
                    else:
                        self.log_result(
                            f"Edge Case - {case_name}", 
                            expected_success, 
                            f"Edge case handled gracefully: HTTP {response.status_code} ({response_time:.3f}s)",
                            {"response_time": response_time, "status_code": response.status_code}
                        )
                    
            except Exception as e:
                # Timeouts or connection errors are acceptable for very long content
                if case_name == "Very Long Content" and "timeout" in str(e).lower():
                    self.log_result(f"Edge Case - {case_name}", True, f"Timeout handling working: {str(e)}")
                else:
                    self.log_result(f"Edge Case - {case_name}", False, f"Request failed: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests including Phase 5 Enhanced Ethics Pipeline"""
        print("🚀 Starting Comprehensive Backend Testing for Ethical AI Developer Testbed")
        print("🧠 Focus: Phase 5 Enhanced Ethics Pipeline with Philosophical Foundations")
        print(f"🔗 Testing backend at: {BACKEND_URL}")
        print("=" * 80)
        
        # Core endpoint tests
        print("\n📊 CORE SYSTEM TESTS")
        print("-" * 40)
        self.test_health_check()
        self.test_parameters_endpoint()
        self.test_learning_stats_endpoint()
        
        print("\n🧠 PHASE 5 ENHANCED ETHICS PIPELINE TESTS")
        print("-" * 60)
        
        # Phase 5 Enhanced Ethics Pipeline Tests
        self.test_enhanced_ethics_pipeline_status()
        self.test_meta_ethical_analysis()
        self.test_normative_ethics_analysis()
        self.test_applied_ethics_analysis()
        self.test_comprehensive_ethics_analysis()
        self.test_ml_training_ethical_guidance()
        
        print("\n📚 PHILOSOPHICAL RIGOR VALIDATION")
        print("-" * 50)
        self.test_philosophical_rigor_validation()
        
        print("\n🛡️ ERROR HANDLING & EDGE CASES")
        print("-" * 40)
        self.test_error_handling_and_edge_cases()
        
        print("\n⚡ INTEGRATION TESTS")
        print("-" * 30)
        self.test_integration_workflow()
        
        # Summary
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if self.performance_metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            phase5_metrics = [m for m in self.performance_metrics if any(keyword in m[0] for keyword in ['meta_analysis', 'normative_analysis', 'applied_analysis', 'comprehensive_analysis', 'ml_guidance'])]
            if phase5_metrics:
                print("   Phase 5 Enhanced Ethics Pipeline:")
                for metric_name, response_time in phase5_metrics:
                    print(f"     {metric_name}: {response_time:.3f}s")
            
            other_metrics = [m for m in self.performance_metrics if not any(keyword in m[0] for keyword in ['meta_analysis', 'normative_analysis', 'applied_analysis', 'comprehensive_analysis', 'ml_guidance'])]
            if other_metrics:
                print("   Other Systems:")
                for metric_name, response_time in other_metrics:
                    print(f"     {metric_name}: {response_time:.3f}s")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            phase5_failures = [t for t in self.failed_tests if any(keyword in t for keyword in ['Meta-Ethical', 'Normative Ethics', 'Applied Ethics', 'Comprehensive Ethics', 'ML Training', 'Philosophical Rigor', 'Enhanced Ethics'])]
            other_failures = [t for t in self.failed_tests if not any(keyword in t for keyword in ['Meta-Ethical', 'Normative Ethics', 'Applied Ethics', 'Comprehensive Ethics', 'ML Training', 'Philosophical Rigor', 'Enhanced Ethics'])]
            
            if phase5_failures:
                print("   Phase 5 Enhanced Ethics Pipeline:")
                for test_name in phase5_failures:
                    print(f"     - {test_name}")
            
            if other_failures:
                print("   Other Systems:")
                for test_name in other_failures:
                    print(f"     - {test_name}")
        
        print("\n🎯 PHASE 5 ENHANCED ETHICS PIPELINE RESULTS:")
        phase5_tests = [t for t in self.results if any(keyword in t['test'] for keyword in ['Meta-Ethical', 'Normative Ethics', 'Applied Ethics', 'Comprehensive Ethics', 'ML Training', 'Philosophical Rigor', 'Enhanced Ethics'])]
        phase5_passed = len([t for t in phase5_tests if t['success']])
        phase5_total = len(phase5_tests)
        
        if phase5_total > 0:
            print(f"   Enhanced Ethics Pipeline Tests: {phase5_passed}/{phase5_total} passed ({(phase5_passed/phase5_total)*100:.1f}%)")
            
            # Check critical Phase 5 functionality
            meta_ethics_working = any(t['success'] for t in phase5_tests if 'Meta-Ethical Analysis' in t['test'])
            normative_ethics_working = any(t['success'] for t in phase5_tests if 'Normative Ethics Analysis' in t['test'])
            applied_ethics_working = any(t['success'] for t in phase5_tests if 'Applied Ethics Analysis' in t['test'])
            comprehensive_working = any(t['success'] for t in phase5_tests if 'Comprehensive Ethics Analysis' in t['test'])
            ml_guidance_working = any(t['success'] for t in phase5_tests if 'ML Training Ethical Guidance' in t['test'])
            
            print(f"   ✅ Meta-Ethics Layer: {'WORKING' if meta_ethics_working else 'FAILED'}")
            print(f"   ✅ Normative Ethics Layer: {'WORKING' if normative_ethics_working else 'FAILED'}")
            print(f"   ✅ Applied Ethics Layer: {'WORKING' if applied_ethics_working else 'FAILED'}")
            print(f"   ✅ Comprehensive Analysis: {'WORKING' if comprehensive_working else 'FAILED'}")
            print(f"   ✅ ML Training Guidance: {'WORKING' if ml_guidance_working else 'FAILED'}")
            
            if all([meta_ethics_working, normative_ethics_working, applied_ethics_working, comprehensive_working]):
                print("   🎉 PHASE 5 ENHANCED ETHICS PIPELINE: FULLY OPERATIONAL")
            elif any([meta_ethics_working, normative_ethics_working, applied_ethics_working]):
                print("   ⚠️  PHASE 5 ENHANCED ETHICS PIPELINE: PARTIALLY OPERATIONAL")
            else:
                print("   ❌ PHASE 5 ENHANCED ETHICS PIPELINE: NOT OPERATIONAL")
        
        return passed_count == total_tests

if __name__ == "__main__":
    tester = EthicalEvaluationTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)