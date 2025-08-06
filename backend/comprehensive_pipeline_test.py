#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline Test for Ethical AI Testbed

This script performs exhaustive testing of all API endpoints with increasing complexity
to verify the entire evaluation pipeline is functioning correctly.

Test Methodology:
    - Tests all API endpoints with four levels of complexity: simple, moderate, complex, and edge cases
    - Measures response times and success rates for each endpoint and complexity level
    - Provides detailed statistics and saves comprehensive test results to a JSON file
    - Validates system robustness against invalid inputs and edge cases

Endpoints Tested:
    - /api/health: System health check
    - /api/evaluate: Text evaluation endpoint
    - /api/parameters: Parameter management
    - /api/learning-stats: Learning system statistics
    - /api/heat-map-mock: Heat map visualization data
    - /api/ethics/comprehensive-analysis: Comprehensive ethics analysis
    - /api/ethics/meta-analysis: Meta-ethics analysis
    - /api/ethics/normative-analysis: Normative ethics analysis
    - /api/ethics/applied-analysis: Applied ethics analysis
    - /api/ethics/ml-training-guidance: ML training guidance

Usage:
    python comprehensive_pipeline_test.py [--url BASE_URL] [--complexity LEVEL]

Options:
    --url BASE_URL         Base URL for the API (default: http://localhost:8001)
    --complexity LEVEL     Complexity level for tests (simple, moderate, complex, edge_case, all)
                          Default is 'all' which tests all complexity levels
"""

import json
import requests
import time
import sys
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import argparse

# Configuration
BASE_URL = "http://localhost:8001"

# Test complexity levels
COMPLEXITY_LEVELS = ["simple", "moderate", "complex", "edge_case"]

# Sample texts of increasing complexity for testing
TEST_TEXTS = {
    "simple": [
        "This is a simple test sentence.",
        "Hello world, how are you today?",
        "The sky is blue and the grass is green."
    ],
    "moderate": [
        "Artificial intelligence should be developed with ethical considerations in mind to ensure it benefits humanity.",
        "Climate change poses significant challenges that require global cooperation and innovative solutions.",
        "Privacy concerns must be balanced with the benefits of data collection and analysis in modern technology."
    ],
    "complex": [
        """The intersection of artificial intelligence and ethics presents multifaceted challenges that span technical, 
        philosophical, and societal domains. As AI systems become increasingly autonomous and influential in decision-making 
        processes, questions arise about responsibility, transparency, bias mitigation, and the preservation of human agency. 
        These considerations must be addressed through robust governance frameworks and ongoing dialogue between diverse stakeholders.""",
        
        """Quantum computing represents a paradigm shift in computational capabilities, with potential implications for 
        cryptography, drug discovery, and complex system modeling. However, the transition from theoretical promise to 
        practical application involves overcoming significant technical hurdles related to qubit stability, error correction, 
        and scalability. The ethical dimensions of quantum computing include considerations of equitable access, security 
        implications, and potential disruptions to existing technological ecosystems.""",
        
        """The development of autonomous vehicles necessitates careful consideration of decision-making algorithms in 
        potential harm scenarios. The classic trolley problem takes on new dimensions when implemented in code that must 
        make split-second decisions with life-or-death consequences. This raises profound questions about the values we 
        encode into machines and the moral frameworks that should guide their behavior in situations where harm is unavoidable."""
    ],
    "edge_case": [
        # Empty text
        "",
        
        # Very long text (truncated for readability)
        "A" * 10000,
        
        # Text with special characters
        "!@#$%^&*()_+{}|:<>?~`-=[]\\;',./\n\t\"",
        
        # Text with potential SQL injection
        "'; DROP TABLE users; --",
        
        # Text with potential XSS
        "<script>alert('XSS')</script>",
        
        # Text with multiple languages
        "English text with 中文 and Español and Français and Русский and العربية"
    ]
}

# Test contexts of increasing complexity
TEST_CONTEXTS = {
    "simple": {"domain": "general"},
    "moderate": {"domain": "healthcare", "cultural_context": "western"},
    "complex": {
        "domain": "ai_ethics",
        "cultural_context": "global",
        "application_area": "autonomous_systems",
        "stakeholders": ["developers", "users", "regulators", "society"],
        "risk_level": "high"
    },
    "edge_case": {
        "domain": None,
        "empty_list": [],
        "nested": {"level1": {"level2": {"level3": "deep"}}},
        "mixed_types": [1, "string", True, None, {"key": "value"}]
    }
}

# Test parameters of increasing complexity
TEST_PARAMETERS = {
    "simple": {"explanation_level": "basic"},
    "moderate": {
        "explanation_level": "detailed",
        "confidence_threshold": 0.8
    },
    "complex": {
        "explanation_level": "comprehensive",
        "confidence_threshold": 0.9,
        "include_reasoning": True,
        "ethical_frameworks": ["virtue", "deontological", "consequentialist"],
        "sensitivity": "high"
    },
    "edge_case": {
        "explanation_level": None,
        "confidence_threshold": -0.1,  # Invalid value
        "include_reasoning": "not_a_boolean",  # Wrong type
        "empty_dict": {}
    }
}

class EndpointTester:
    """Class to test API endpoints with increasing complexity."""
    
    def __init__(self, base_url: str):
        """Initialize the tester with the base URL."""
        self.base_url = base_url
        self.results = []
        self.session = requests.Session()
    
    def test_endpoint(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict[str, Any]] = None, 
                     complexity: str = "simple") -> Dict[str, Any]:
        """Test a specific API endpoint and return the result."""
        url = f"{self.base_url}{endpoint}"
        
        print(f"\nTesting endpoint: {endpoint}")
        print(f"Complexity: {complexity}")
        if data:
            print(f"Data sample: {str(data)[:100]}...")
        
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=30)
            else:  # POST
                response = self.session.post(url, json=data, timeout=30)
            
            elapsed_time = time.time() - start_time
            
            # Process response
            result = {
                "endpoint": endpoint,
                "method": method,
                "complexity": complexity,
                "status": "success" if response.status_code == 200 else "failure",
                "status_code": response.status_code,
                "response_time": elapsed_time,
                "timestamp": datetime.now().isoformat(),
                "request_data": data,
                "response": None
            }
            
            # Try to parse JSON response
            try:
                if response.text:
                    result["response"] = response.json()
            except json.JSONDecodeError:
                result["response"] = response.text[:100] + "..." if len(response.text) > 100 else response.text
            
        except requests.exceptions.RequestException as e:
            elapsed_time = time.time() - start_time
            result = {
                "endpoint": endpoint,
                "method": method,
                "complexity": complexity,
                "status": "error",
                "error": str(e),
                "status_code": None,
                "response_time": elapsed_time,
                "timestamp": datetime.now().isoformat(),
                "request_data": data,
                "response": None
            }
        
        self.results.append(result)
        
        # Print status
        status_symbol = "✅" if result["status"] == "success" else "❌"
        print(f"{status_symbol} {endpoint} ({complexity}): {result.get('status_code')} in {result.get('response_time'):.2f}s")
        
        return result
    
    def test_health_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the health endpoint."""
        return self.test_endpoint("/api/health", "GET", None, complexity)
    
    def test_evaluate_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the evaluate endpoint with text of specified complexity."""
        text = random.choice(TEST_TEXTS[complexity])
        data = {
            "text": text,
            "context": TEST_CONTEXTS[complexity],
            "parameters": TEST_PARAMETERS[complexity],
            "mode": "development" if complexity != "edge_case" else "invalid_mode",
            "priority": "normal" if complexity != "edge_case" else "invalid_priority"
        }
        return self.test_endpoint("/api/evaluate", "POST", data, complexity)
    
    def test_parameters_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the parameters endpoint."""
        return self.test_endpoint("/api/parameters", "GET", None, complexity)
    
    def test_learning_stats_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the learning stats endpoint."""
        return self.test_endpoint("/api/learning-stats", "GET", None, complexity)
    
    def test_heat_map_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the heat map endpoint with text of specified complexity."""
        text = random.choice(TEST_TEXTS[complexity])
        data = {"text": text}
        return self.test_endpoint("/api/heat-map-mock", "POST", data, complexity)
    
    def test_comprehensive_ethics_analysis_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the comprehensive ethics analysis endpoint."""
        text = random.choice(TEST_TEXTS[complexity])
        data = {
            "text": text,
            "context": TEST_CONTEXTS[complexity]
        }
        return self.test_endpoint("/api/ethics/comprehensive-analysis", "POST", data, complexity)
    
    def test_meta_ethics_analysis_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the meta ethics analysis endpoint."""
        text = random.choice(TEST_TEXTS[complexity])
        data = {
            "text": text,
            "context": TEST_CONTEXTS[complexity]
        }
        return self.test_endpoint("/api/ethics/meta-analysis", "POST", data, complexity)
    
    def test_normative_ethics_analysis_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the normative ethics analysis endpoint."""
        text = random.choice(TEST_TEXTS[complexity])
        data = {
            "text": text,
            "context": TEST_CONTEXTS[complexity]
        }
        return self.test_endpoint("/api/ethics/normative-analysis", "POST", data, complexity)
    
    def test_applied_ethics_analysis_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the applied ethics analysis endpoint."""
        text = random.choice(TEST_TEXTS[complexity])
        data = {
            "text": text,
            "context": TEST_CONTEXTS[complexity]
        }
        return self.test_endpoint("/api/ethics/applied-analysis", "POST", data, complexity)
    
    def test_ml_training_guidance_endpoint(self, complexity: str) -> Dict[str, Any]:
        """Test the ML training guidance endpoint."""
        text = random.choice(TEST_TEXTS[complexity])
        data = {
            "content": text,
            "context": TEST_CONTEXTS[complexity]
        }
        return self.test_endpoint("/api/ethics/ml-training-guidance", "POST", data, complexity)
    
    def run_all_tests(self, complexity_levels: List[str] = None) -> None:
        """Run all tests for specified complexity levels."""
        if complexity_levels is None:
            complexity_levels = COMPLEXITY_LEVELS
        
        print(f"Starting comprehensive pipeline tests at {datetime.now().isoformat()}")
        print(f"Base URL: {self.base_url}")
        print(f"Testing with complexity levels: {', '.join(complexity_levels)}")
        print("-" * 80)
        
        for complexity in complexity_levels:
            print(f"\n=== Testing with {complexity.upper()} complexity ===")
            
            # Test all endpoints with current complexity
            self.test_health_endpoint(complexity)
            self.test_evaluate_endpoint(complexity)
            self.test_parameters_endpoint(complexity)
            self.test_learning_stats_endpoint(complexity)
            self.test_heat_map_endpoint(complexity)
            self.test_comprehensive_ethics_analysis_endpoint(complexity)
            self.test_meta_ethics_analysis_endpoint(complexity)
            self.test_normative_ethics_analysis_endpoint(complexity)
            self.test_applied_ethics_analysis_endpoint(complexity)
            self.test_ml_training_guidance_endpoint(complexity)
            
            # Add a small delay between complexity levels
            if complexity != complexity_levels[-1]:
                print("\nWaiting 2 seconds before next complexity level...")
                time.sleep(2)
        
        print("\n" + "-" * 80)
        self.summarize_results()
    
    def summarize_results(self) -> None:
        """Summarize the test results."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["status"] == "success")
        failed_tests = total_tests - successful_tests
        
        print(f"\nTest Summary:")
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(successful_tests / total_tests) * 100:.1f}%")
        
        # Group by endpoint and complexity
        endpoint_stats = {}
        complexity_stats = {}
        
        for result in self.results:
            endpoint = result["endpoint"]
            complexity = result["complexity"]
            success = result["status"] == "success"
            
            # Update endpoint stats
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {"total": 0, "success": 0}
            endpoint_stats[endpoint]["total"] += 1
            if success:
                endpoint_stats[endpoint]["success"] += 1
            
            # Update complexity stats
            if complexity not in complexity_stats:
                complexity_stats[complexity] = {"total": 0, "success": 0}
            complexity_stats[complexity]["total"] += 1
            if success:
                complexity_stats[complexity]["success"] += 1
        
        # Print endpoint stats
        print("\nEndpoint Statistics:")
        for endpoint, stats in endpoint_stats.items():
            success_rate = (stats["success"] / stats["total"]) * 100
            print(f"{endpoint}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Print complexity stats
        print("\nComplexity Statistics:")
        for complexity, stats in complexity_stats.items():
            success_rate = (stats["success"] / stats["total"]) * 100
            print(f"{complexity}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    def save_results(self) -> str:
        """Save results to a JSON file and return the filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_test_results_{timestamp}.json"
        
        with open(filename, "w") as fp:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "summary": {
                    "total": len(self.results),
                    "successful": sum(1 for r in self.results if r["status"] == "success"),
                    "failed": sum(1 for r in self.results if r["status"] != "success")
                },
                "results": self.results
            }, fp, indent=2)
        
        print(f"\nDetailed results saved to {filename}")
        return filename

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Run comprehensive pipeline tests for Ethical AI Testbed")
    parser.add_argument("--url", default=BASE_URL, help=f"Base URL for the API (default: {BASE_URL})")
    parser.add_argument("--complexity", choices=COMPLEXITY_LEVELS + ["all"], default="all",
                        help="Complexity level for tests (default: all)")
    args = parser.parse_args()
    
    # Determine complexity levels to test
    complexity_levels = COMPLEXITY_LEVELS if args.complexity == "all" else [args.complexity]
    
    # Create tester and run tests
    tester = EndpointTester(args.url)
    tester.run_all_tests(complexity_levels)
    tester.save_results()
    
    # Return non-zero exit code if any tests failed
    return 0 if all(r["status"] == "success" for r in tester.results) else 1

if __name__ == "__main__":
    sys.exit(main())
