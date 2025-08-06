#!/usr/bin/env python3
"""
Comprehensive Orthonormalization Validation Test Suite

This module provides extensive testing of the orthonormalization and framework aggregation
fixes with complex, emergent test scenarios to validate axis independence and emergence
properties before proceeding to perceptron-based threshold learning.

Focus Areas:
1. Framework-specific patterns and independence
2. Emergent ethical conflicts and dilemmas
3. Complex multi-framework scenarios
4. Edge cases and boundary conditions
5. Statistical validation of axis independence
"""

import asyncio
import logging
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import statistics
from pathlib import Path

from backend.adaptive_threshold_learner import IntentNormalizedFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveValidationSuite:
    """Comprehensive validation test suite for orthonormalization and axis independence."""
    
    def __init__(self, alpha: float = 0.2):
        self.extractor = IntentNormalizedFeatureExtractor(alpha=alpha)
        self.results = []
        self.test_categories = {}
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run the complete validation suite and return comprehensive results."""
        logger.info("Starting comprehensive orthonormalization validation...")
        
        # Test categories with increasing complexity
        test_suites = [
            ("basic_framework_independence", self._test_basic_framework_independence),
            ("emergent_ethical_conflicts", self._test_emergent_ethical_conflicts),
            ("complex_multi_framework", self._test_complex_multi_framework),
            ("boundary_conditions", self._test_boundary_conditions),
            ("statistical_independence", self._test_statistical_independence),
            ("real_world_scenarios", self._test_real_world_scenarios)
        ]
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "test_suites": {},
            "overall_summary": {},
            "axis_independence_metrics": {}
        }
        
        for suite_name, test_function in test_suites:
            logger.info(f"Running test suite: {suite_name}")
            suite_results = await test_function()
            validation_results["test_suites"][suite_name] = suite_results
            self.test_categories[suite_name] = suite_results
        
        # Calculate overall metrics
        validation_results["overall_summary"] = self._calculate_overall_metrics()
        validation_results["axis_independence_metrics"] = self._analyze_axis_independence()
        
        # Save results
        await self._save_validation_results(validation_results)
        
        return validation_results
    
    async def _test_basic_framework_independence(self) -> Dict[str, Any]:
        """Test basic framework-specific patterns and independence."""
        test_cases = [
            # Pure virtue ethics scenarios
            ("This person demonstrates exceptional integrity, wisdom, and moral courage.", "virtue_positive"),
            ("This individual shows dishonesty, cowardice, and moral corruption.", "virtue_negative"),
            
            # Pure deontological scenarios  
            ("This action strictly follows all legal requirements and moral duties.", "deontological_positive"),
            ("This behavior violates laws, breaks promises, and ignores moral obligations.", "deontological_negative"),
            
            # Pure consequentialist scenarios
            ("This decision will maximize happiness and benefit for all stakeholders.", "consequentialist_positive"),
            ("This choice will cause widespread harm, suffering, and negative outcomes.", "consequentialist_negative"),
            
            # Mixed scenarios for comparison
            ("This honest legal action benefits everyone involved.", "all_positive"),
            ("This dishonest illegal action harms everyone involved.", "all_negative")
        ]
        
        results = []
        independence_scores = []
        
        for text, category in test_cases:
            features = await self.extractor.extract_features(text)
            
            # Calculate independence metrics
            raw_scores = [features.raw_virtue_score, features.raw_deontological_score, features.raw_consequentialist_score]
            norm_scores = [features.normalized_virtue_score, features.normalized_deontological_score, features.normalized_consequentialist_score]
            ortho_scores = [features.ortho_virtue_score, features.ortho_deontological_score, features.ortho_consequentialist_score]
            
            # Independence measures
            raw_independence = self._calculate_independence_score(raw_scores)
            norm_independence = self._calculate_independence_score(norm_scores)
            ortho_independence = self._calculate_independence_score(ortho_scores)
            
            result = {
                "text": text,
                "category": category,
                "raw_scores": {"virtue": features.raw_virtue_score, "deontological": features.raw_deontological_score, "consequentialist": features.raw_consequentialist_score},
                "normalized_scores": {"virtue": features.normalized_virtue_score, "deontological": features.normalized_deontological_score, "consequentialist": features.normalized_consequentialist_score},
                "orthonormal_scores": {"virtue": features.ortho_virtue_score, "deontological": features.ortho_deontological_score, "consequentialist": features.ortho_consequentialist_score},
                "independence_metrics": {
                    "raw_independence": raw_independence,
                    "normalized_independence": norm_independence,
                    "orthonormal_independence": ortho_independence
                },
                "harm_intensity": features.harm_intensity
            }
            
            results.append(result)
            independence_scores.append(ortho_independence)
        
        return {
            "test_count": len(test_cases),
            "results": results,
            "summary": {
                "avg_orthonormal_independence": statistics.mean(independence_scores),
                "min_independence": min(independence_scores),
                "max_independence": max(independence_scores),
                "independence_std": statistics.stdev(independence_scores) if len(independence_scores) > 1 else 0.0
            }
        }
    
    async def _test_emergent_ethical_conflicts(self) -> Dict[str, Any]:
        """Test emergent ethical conflicts and complex dilemmas."""
        # Complex scenarios where frameworks conflict - emergence happens here
        conflict_scenarios = [
            # Virtue vs Deontological
            ("A honest person must lie to protect innocent lives from an unjust law.", "virtue_vs_deontological"),
            
            # Virtue vs Consequentialist  
            ("A compassionate person refuses to sacrifice one innocent to save many others.", "virtue_vs_consequentialist"),
            
            # Deontological vs Consequentialist
            ("Following the law strictly will cause preventable suffering and harm.", "deontological_vs_consequentialist"),
            
            # Triple conflicts - maximum complexity
            ("A truthful whistleblower breaks confidentiality laws to prevent corporate harm but destroys their own integrity in the process.", "triple_conflict_1"),
            ("A loyal friend lies to authorities to protect someone who committed crimes that helped many people.", "triple_conflict_2"),
            ("A doctor violates patient privacy to warn the public about a dangerous treatment that saves lives.", "triple_conflict_3"),
            
            # Emergent moral complexity
            ("This AI system optimizes for user engagement by promoting content that gradually erodes critical thinking while providing immediate satisfaction.", "emergent_ai_ethics"),
            ("This algorithm fairly distributes resources but creates systematic dependencies that reduce long-term autonomy.", "emergent_algorithmic_justice"),
            ("This automated decision system treats everyone equally but perpetuates historical biases through seemingly neutral criteria.", "emergent_bias_amplification")
        ]
        
        results = []
        conflict_metrics = []
        
        for text, scenario_type in conflict_scenarios:
            features = await self.extractor.extract_features(text)
            
            # Analyze conflict patterns
            ortho_scores = [features.ortho_virtue_score, features.ortho_deontological_score, features.ortho_consequentialist_score]
            
            # Conflict intensity: how much frameworks disagree
            conflict_intensity = self._calculate_conflict_intensity(ortho_scores)
            
            # Emergence metric: complexity of the ethical pattern
            emergence_score = self._calculate_emergence_score(features)
            
            result = {
                "text": text,
                "scenario_type": scenario_type,
                "orthonormal_scores": {"virtue": features.ortho_virtue_score, "deontological": features.ortho_deontological_score, "consequentialist": features.ortho_consequentialist_score},
                "conflict_intensity": conflict_intensity,
                "emergence_score": emergence_score,
                "harm_intensity": features.harm_intensity,
                "independence_score": self._calculate_independence_score(ortho_scores)
            }
            
            results.append(result)
            conflict_metrics.append(conflict_intensity)
        
        return {
            "test_count": len(conflict_scenarios),
            "results": results,
            "summary": {
                "avg_conflict_intensity": statistics.mean(conflict_metrics),
                "max_conflict": max(conflict_metrics),
                "emergence_detected": len([r for r in results if r["emergence_score"] > 0.7])
            }
        }
    
    async def _test_complex_multi_framework(self) -> Dict[str, Any]:
        """Test complex scenarios involving all three frameworks simultaneously."""
        complex_scenarios = [
            # Corporate ethics
            ("The CEO demonstrates personal integrity while following fiduciary duties that maximize shareholder value through environmentally harmful but legal practices.", "corporate_complexity"),
            
            # Medical ethics
            ("The surgeon maintains professional honesty about risks while respecting patient autonomy and optimizing treatment outcomes for limited resources.", "medical_complexity"),
            
            # AI/Technology ethics
            ("The AI researcher publishes transparent findings about bias while complying with privacy regulations and minimizing potential misuse of the technology.", "ai_complexity"),
            
            # Political/Social ethics
            ("The politician keeps campaign promises while following constitutional law and pursuing policies that benefit the greatest number of citizens.", "political_complexity"),
            
            # Educational ethics
            ("The teacher maintains academic integrity while accommodating diverse learning needs and preparing students for competitive environments.", "educational_complexity"),
            
            # Environmental ethics
            ("The environmental scientist reports accurate climate data while respecting indigenous rights and supporting economic development for impoverished communities.", "environmental_complexity")
        ]
        
        results = []
        complexity_scores = []
        
        for text, domain in complex_scenarios:
            features = await self.extractor.extract_features(text)
            
            # Multi-framework analysis
            ortho_scores = [features.ortho_virtue_score, features.ortho_deontological_score, features.ortho_consequentialist_score]
            
            # Complexity metrics
            framework_balance = self._calculate_framework_balance(ortho_scores)
            interaction_complexity = self._calculate_interaction_complexity(features)
            
            result = {
                "text": text,
                "domain": domain,
                "orthonormal_scores": {"virtue": features.ortho_virtue_score, "deontological": features.ortho_deontological_score, "consequentialist": features.ortho_consequentialist_score},
                "framework_balance": framework_balance,
                "interaction_complexity": interaction_complexity,
                "independence_score": self._calculate_independence_score(ortho_scores),
                "harm_intensity": features.harm_intensity
            }
            
            results.append(result)
            complexity_scores.append(interaction_complexity)
        
        return {
            "test_count": len(complex_scenarios),
            "results": results,
            "summary": {
                "avg_complexity": statistics.mean(complexity_scores),
                "high_complexity_cases": len([s for s in complexity_scores if s > 0.8])
            }
        }
    
    async def _test_boundary_conditions(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions."""
        boundary_cases = [
            # Empty/minimal content
            ("", "empty"),
            ("The.", "minimal"),
            ("This.", "minimal_2"),
            
            # Neutral content
            ("This is a neutral statement about weather.", "neutral"),
            ("The object exists in space.", "neutral_2"),
            
            # Extreme cases
            ("This absolutely perfect virtuous legal beneficial action demonstrates complete moral excellence.", "extreme_positive"),
            ("This completely corrupt illegal harmful action demonstrates total moral failure.", "extreme_negative"),
            
            # Identical framework triggers
            ("This honest honest honest action.", "repeated_virtue"),
            ("This legal legal legal action.", "repeated_deontological"),
            ("This beneficial beneficial beneficial action.", "repeated_consequentialist"),
            
            # Mixed repetition
            ("This honest legal beneficial honest legal beneficial action.", "mixed_repetition")
        ]
        
        results = []
        boundary_metrics = []
        
        for text, case_type in boundary_cases:
            try:
                features = await self.extractor.extract_features(text)
                
                ortho_scores = [features.ortho_virtue_score, features.ortho_deontological_score, features.ortho_consequentialist_score]
                independence = self._calculate_independence_score(ortho_scores)
                
                result = {
                    "text": text,
                    "case_type": case_type,
                    "orthonormal_scores": {"virtue": features.ortho_virtue_score, "deontological": features.ortho_deontological_score, "consequentialist": features.ortho_consequentialist_score},
                    "independence_score": independence,
                    "harm_intensity": features.harm_intensity,
                    "success": True
                }
                
                boundary_metrics.append(independence)
                
            except Exception as e:
                result = {
                    "text": text,
                    "case_type": case_type,
                    "error": str(e),
                    "success": False
                }
            
            results.append(result)
        
        return {
            "test_count": len(boundary_cases),
            "results": results,
            "summary": {
                "success_rate": len([r for r in results if r.get("success", False)]) / len(results),
                "avg_boundary_independence": statistics.mean(boundary_metrics) if boundary_metrics else 0.0
            }
        }
    
    async def _test_statistical_independence(self) -> Dict[str, Any]:
        """Test statistical independence across large sample."""
        # Generate diverse test cases for statistical analysis
        statistical_cases = [
            "This person shows great courage and wisdom.",
            "This action violates several important laws.",
            "This decision will cause significant harm.",
            "This honest legal beneficial choice helps everyone.",
            "This dishonest illegal harmful action hurts people.",
            "The virtuous leader follows duty while maximizing good.",
            "The corrupt official breaks laws causing widespread damage.",
            "This transparent process respects rights and improves outcomes.",
            "This deceptive procedure violates consent and worsens conditions.",
            "The ethical framework balances character, duty, and consequences."
        ]
        
        # Multiply cases for statistical power
        extended_cases = statistical_cases * 3  # 30 total cases
        
        all_raw_scores = []
        all_norm_scores = []
        all_ortho_scores = []
        independence_scores = []
        
        for text in extended_cases:
            features = await self.extractor.extract_features(text)
            
            raw_scores = [features.raw_virtue_score, features.raw_deontological_score, features.raw_consequentialist_score]
            norm_scores = [features.normalized_virtue_score, features.normalized_deontological_score, features.normalized_consequentialist_score]
            ortho_scores = [features.ortho_virtue_score, features.ortho_deontological_score, features.ortho_consequentialist_score]
            
            all_raw_scores.append(raw_scores)
            all_norm_scores.append(norm_scores)
            all_ortho_scores.append(ortho_scores)
            independence_scores.append(self._calculate_independence_score(ortho_scores))
        
        # Statistical analysis
        correlations = self._calculate_correlation_matrix(all_ortho_scores)
        
        return {
            "test_count": len(extended_cases),
            "statistical_metrics": {
                "avg_independence": statistics.mean(independence_scores),
                "independence_std": statistics.stdev(independence_scores),
                "correlation_matrix": correlations,
                "max_correlation": max([abs(c) for row in correlations for c in row if c != 1.0]),
                "independence_distribution": {
                    "high": len([s for s in independence_scores if s > 0.8]),
                    "medium": len([s for s in independence_scores if 0.5 < s <= 0.8]),
                    "low": len([s for s in independence_scores if s <= 0.5])
                }
            }
        }
    
    async def _test_real_world_scenarios(self) -> Dict[str, Any]:
        """Test real-world complex scenarios where emergence is expected."""
        real_world_cases = [
            # Technology ethics
            ("This social media algorithm increases user engagement by showing content that confirms existing beliefs while reducing exposure to diverse perspectives.", "social_media_echo_chambers"),
            
            # Healthcare AI
            ("This diagnostic AI system improves accuracy for common conditions but shows bias against underrepresented populations in training data.", "healthcare_ai_bias"),
            
            # Autonomous vehicles
            ("This self-driving car system prioritizes passenger safety over pedestrian safety in unavoidable accident scenarios.", "autonomous_vehicle_ethics"),
            
            # Workplace automation
            ("This automation system increases productivity and reduces human error while eliminating jobs for workers without alternative opportunities.", "workplace_automation"),
            
            # Financial algorithms
            ("This credit scoring algorithm uses legally compliant data to make fair individual decisions that perpetuate systemic inequality.", "algorithmic_credit_scoring"),
            
            # Educational technology
            ("This adaptive learning system personalizes education effectively while collecting extensive data on student behavior and performance.", "educational_surveillance"),
            
            # Environmental monitoring
            ("This environmental monitoring system provides accurate pollution data while enabling surveillance of individual movement patterns.", "environmental_surveillance"),
            
            # Content moderation
            ("This content moderation system removes harmful content effectively while suppressing legitimate political discourse through algorithmic bias.", "content_moderation_bias")
        ]
        
        results = []
        emergence_scores = []
        
        for text, scenario in real_world_cases:
            features = await self.extractor.extract_features(text)
            
            ortho_scores = [features.ortho_virtue_score, features.ortho_deontological_score, features.ortho_consequentialist_score]
            
            # Real-world complexity metrics
            emergence_score = self._calculate_emergence_score(features)
            ethical_tension = self._calculate_ethical_tension(ortho_scores)
            
            result = {
                "text": text,
                "scenario": scenario,
                "orthonormal_scores": {"virtue": features.ortho_virtue_score, "deontological": features.ortho_deontological_score, "consequentialist": features.ortho_consequentialist_score},
                "emergence_score": emergence_score,
                "ethical_tension": ethical_tension,
                "independence_score": self._calculate_independence_score(ortho_scores),
                "harm_intensity": features.harm_intensity
            }
            
            results.append(result)
            emergence_scores.append(emergence_score)
        
        return {
            "test_count": len(real_world_cases),
            "results": results,
            "summary": {
                "avg_emergence": statistics.mean(emergence_scores),
                "high_emergence_cases": len([s for s in emergence_scores if s > 0.7]),
                "complex_scenarios_detected": len([r for r in results if r["ethical_tension"] > 0.6])
            }
        }
    
    def _calculate_independence_score(self, scores: List[float]) -> float:
        """Calculate independence score based on variance and uniqueness."""
        if len(set([round(s, 3) for s in scores])) == 1:
            return 0.0  # No independence
        
        # Variance-based independence
        variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        
        # Uniqueness factor
        unique_count = len(set([round(s, 2) for s in scores]))
        uniqueness = unique_count / len(scores)
        
        # Combined independence score
        return min(1.0, variance * 4 + uniqueness * 0.5)
    
    def _calculate_conflict_intensity(self, scores: List[float]) -> float:
        """Calculate how much the frameworks conflict with each other."""
        if len(scores) < 2:
            return 0.0
        
        # Calculate pairwise differences
        differences = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                differences.append(abs(scores[i] - scores[j]))
        
        return statistics.mean(differences)
    
    def _calculate_emergence_score(self, features) -> float:
        """Calculate emergence score based on complexity of ethical patterns."""
        # Emergence indicated by high independence + high harm intensity + framework balance
        ortho_scores = [features.ortho_virtue_score, features.ortho_deontological_score, features.ortho_consequentialist_score]
        
        independence = self._calculate_independence_score(ortho_scores)
        balance = self._calculate_framework_balance(ortho_scores)
        intensity = features.harm_intensity
        
        # Emergence = complexity arising from interaction of all factors
        emergence = (independence * 0.4 + balance * 0.3 + intensity * 0.3)
        return min(1.0, emergence)
    
    def _calculate_framework_balance(self, scores: List[float]) -> float:
        """Calculate how balanced the frameworks are (vs one dominating)."""
        if not scores:
            return 0.0
        
        # Perfect balance = all scores equal, imbalance = one score dominates
        mean_score = statistics.mean(scores)
        if mean_score == 0:
            return 0.0
        
        deviations = [abs(score - mean_score) / mean_score for score in scores]
        avg_deviation = statistics.mean(deviations)
        
        # Convert to balance score (lower deviation = higher balance)
        balance = max(0.0, 1.0 - avg_deviation)
        return balance
    
    def _calculate_interaction_complexity(self, features) -> float:
        """Calculate complexity of framework interactions."""
        # Complexity from multiple factors interacting
        ortho_scores = [features.ortho_virtue_score, features.ortho_deontological_score, features.ortho_consequentialist_score]
        
        independence = self._calculate_independence_score(ortho_scores)
        conflict = self._calculate_conflict_intensity(ortho_scores)
        harm = features.harm_intensity
        
        # Interaction complexity = non-linear combination
        complexity = independence * conflict * (1 + harm)
        return min(1.0, complexity)
    
    def _calculate_ethical_tension(self, scores: List[float]) -> float:
        """Calculate ethical tension from competing framework demands."""
        # Tension = high scores in multiple frameworks (competing demands)
        high_scores = [s for s in scores if s > 0.6]
        
        if len(high_scores) <= 1:
            return 0.0  # No tension if only one framework activated
        
        # Tension increases with number of competing high scores
        tension = len(high_scores) / len(scores) * statistics.mean(high_scores)
        return min(1.0, tension)
    
    def _calculate_correlation_matrix(self, score_matrix: List[List[float]]) -> List[List[float]]:
        """Calculate correlation matrix for statistical independence analysis."""
        if not score_matrix:
            return []
        
        # Transpose to get scores by framework
        virtue_scores = [scores[0] for scores in score_matrix]
        deont_scores = [scores[1] for scores in score_matrix]
        conseq_scores = [scores[2] for scores in score_matrix]
        
        frameworks = [virtue_scores, deont_scores, conseq_scores]
        correlations = []
        
        for i in range(3):
            row = []
            for j in range(3):
                if i == j:
                    row.append(1.0)
                else:
                    # Calculate Pearson correlation
                    corr = self._pearson_correlation(frameworks[i], frameworks[j])
                    row.append(corr)
            correlations.append(row)
        
        return correlations
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall validation metrics across all test suites."""
        all_independence_scores = []
        all_emergence_scores = []
        
        for suite_name, suite_results in self.test_categories.items():
            if "results" in suite_results:
                for result in suite_results["results"]:
                    if "independence_score" in result:
                        all_independence_scores.append(result["independence_score"])
                    if "emergence_score" in result:
                        all_emergence_scores.append(result["emergence_score"])
        
        return {
            "total_tests": sum(suite.get("test_count", 0) for suite in self.test_categories.values()),
            "avg_independence": statistics.mean(all_independence_scores) if all_independence_scores else 0.0,
            "avg_emergence": statistics.mean(all_emergence_scores) if all_emergence_scores else 0.0,
            "high_independence_rate": len([s for s in all_independence_scores if s > 0.7]) / len(all_independence_scores) if all_independence_scores else 0.0,
            "high_emergence_rate": len([s for s in all_emergence_scores if s > 0.7]) / len(all_emergence_scores) if all_emergence_scores else 0.0
        }
    
    def _analyze_axis_independence(self) -> Dict[str, Any]:
        """Analyze axis independence across all tests."""
        independence_by_category = {}
        
        for suite_name, suite_results in self.test_categories.items():
            if "results" in suite_results:
                independence_scores = [r.get("independence_score", 0.0) for r in suite_results["results"]]
                independence_by_category[suite_name] = {
                    "avg_independence": statistics.mean(independence_scores) if independence_scores else 0.0,
                    "min_independence": min(independence_scores) if independence_scores else 0.0,
                    "max_independence": max(independence_scores) if independence_scores else 0.0
                }
        
        return independence_by_category
    
    async def _save_validation_results(self, results: Dict[str, Any]) -> None:
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"orthonormalization_validation_{timestamp}.json"
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to: {filepath}")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.extractor.cleanup()

async def main():
    """Run the comprehensive validation suite."""
    print("🧪 Starting Comprehensive Orthonormalization Validation Suite")
    print("=" * 80)
    
    validator = ComprehensiveValidationSuite(alpha=0.2)
    
    try:
        results = await validator.run_full_validation()
        
        # Print summary
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        overall = results["overall_summary"]
        print(f"Total Tests: {overall['total_tests']}")
        print(f"Average Independence Score: {overall['avg_independence']:.3f}")
        print(f"Average Emergence Score: {overall['avg_emergence']:.3f}")
        print(f"High Independence Rate: {overall['high_independence_rate']:.1%}")
        print(f"High Emergence Rate: {overall['high_emergence_rate']:.1%}")
        
        print("\n🎯 AXIS INDEPENDENCE BY CATEGORY")
        print("-" * 40)
        for category, metrics in results["axis_independence_metrics"].items():
            print(f"{category}: {metrics['avg_independence']:.3f} (range: {metrics['min_independence']:.3f}-{metrics['max_independence']:.3f})")
        
        # Success criteria
        success_threshold = 0.6
        independence_success = overall['avg_independence'] > success_threshold
        emergence_success = overall['avg_emergence'] > 0.5
        
        print(f"\n✅ VALIDATION RESULT: {'PASSED' if independence_success and emergence_success else 'NEEDS_IMPROVEMENT'}")
        print(f"   Independence: {'✅' if independence_success else '❌'} ({overall['avg_independence']:.3f} > {success_threshold})")
        print(f"   Emergence: {'✅' if emergence_success else '❌'} ({overall['avg_emergence']:.3f} > 0.5)")
        
        if independence_success and emergence_success:
            print("\n🚀 Ready for Phase 2: Perceptron-based threshold learning!")
        else:
            print("\n🔧 Further optimization needed before Phase 2.")
        
    finally:
        await validator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
