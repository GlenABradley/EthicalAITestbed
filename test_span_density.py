#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import pandas as pd
from backend.ethical_engine import EthicalEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test text with mixed ethical content
TEST_TEXT = """Maria volunteered every Saturday at the community food bank, making sure families in need received fresh produce and essential supplies. On the way home, she occasionally took office pens without asking, figuring no one would notice. Last week, when her coworker was blamed for a scheduling error she had made, she stayed silent to avoid trouble, though she felt uneasy afterward. Later, she donated part of her bonus to a scholarship fund for underprivileged students, hoping to give them opportunities she never had."""

def run_evaluation(density_factor, report_name):
    """Run evaluation with specified span density factor and generate report."""
    # Initialize the evaluator with the specified density factor
    evaluator = EthicalEvaluator()
    evaluator.span_selector.span_density_factor = density_factor
    
    logger.info(f"Running evaluation with density factor: {density_factor}")
    
    # Run evaluation
    evaluation = evaluator.evaluate_text(TEST_TEXT)
    
    # Create report
    violations = [span for span in evaluation.spans if span.is_violation]
    ethical_spans = [span for span in evaluation.spans if not span.is_violation]
    
    report = {
        'density_factor': density_factor,
        'total_spans': len(evaluation.spans),
        'violations': len(violations),
        'ethical_spans': len(ethical_spans),
        'violation_ratio': len(violations) / len(evaluation.spans) if evaluation.spans else 0,
        'lowest_v_score': min([span.virtue_score for span in evaluation.spans]) if evaluation.spans else None,
        'highest_v_score': max([span.virtue_score for span in evaluation.spans]) if evaluation.spans else None,
        'lowest_d_score': min([span.deontological_score for span in evaluation.spans]) if evaluation.spans else None,
        'highest_d_score': max([span.deontological_score for span in evaluation.spans]) if evaluation.spans else None,
        'lowest_c_score': min([span.consequentialist_score for span in evaluation.spans]) if evaluation.spans else None,
        'highest_c_score': max([span.consequentialist_score for span in evaluation.spans]) if evaluation.spans else None,
    }
    
    # Print key violation spans
    logger.info(f"\nVIOLATION SAMPLES (total: {len(violations)}):")
    for i, span in enumerate(sorted(violations, key=lambda s: s.combined_score)[:3]):
        logger.info(f"Violation {i+1}: '{span.text}'")
        logger.info(f"V/D/C: {span.virtue_score:.3f}/{span.deontological_score:.3f}/{span.consequentialist_score:.3f}")
        logger.info(f"Combined: {span.combined_score:.3f}")
        logger.info("-" * 40)
    
    # Save full results to file
    with open(f"{report_name}_full.txt", "w") as f:
        f.write(f"Evaluation with density factor: {density_factor}\n")
        f.write(f"Total spans: {len(evaluation.spans)}\n")
        f.write(f"Violations: {len(violations)}\n")
        f.write(f"Ethical spans: {len(ethical_spans)}\n\n")
        
        f.write("VIOLATIONS:\n")
        for i, span in enumerate(sorted(violations, key=lambda s: s.combined_score)):
            f.write(f"{i+1}. '{span.text}'\n")
            f.write(f"   V/D/C: {span.virtue_score:.3f}/{span.deontological_score:.3f}/{span.consequentialist_score:.3f}\n")
            f.write(f"   Combined: {span.combined_score:.3f}\n\n")
    
    return report

def compare_results(reports):
    """Compare results across different density factors."""
    df = pd.DataFrame(reports)
    print("\nCOMPARISON OF DENSITY FACTORS:")
    print(df)
    
    # Calculate score range preservation
    base_report = reports[0]  # Assuming first report is baseline
    for report in reports[1:]:
        v_range_preserved = (
            (report['lowest_v_score'] - base_report['lowest_v_score']) / base_report['lowest_v_score'] if base_report['lowest_v_score'] else 0,
            (report['highest_v_score'] - base_report['highest_v_score']) / base_report['highest_v_score'] if base_report['highest_v_score'] else 0
        )
        
        d_range_preserved = (
            (report['lowest_d_score'] - base_report['lowest_d_score']) / base_report['lowest_d_score'] if base_report['lowest_d_score'] else 0,
            (report['highest_d_score'] - base_report['highest_d_score']) / base_report['highest_d_score'] if base_report['highest_d_score'] else 0
        )
        
        c_range_preserved = (
            (report['lowest_c_score'] - base_report['lowest_c_score']) / base_report['lowest_c_score'] if base_report['lowest_c_score'] else 0,
            (report['highest_c_score'] - base_report['highest_c_score']) / base_report['highest_c_score'] if base_report['highest_c_score'] else 0
        )
        
        print(f"\nComparison between density {base_report['density_factor']} and {report['density_factor']}:")
        print(f"Spans: {base_report['total_spans']} -> {report['total_spans']} ({report['total_spans']/base_report['total_spans']:.2f}x)")
        print(f"Violations: {base_report['violations']} -> {report['violations']} ({report['violations']/base_report['violations']:.2f}x)")
        print(f"V-score range preservation: {v_range_preserved[0]:.2%}, {v_range_preserved[1]:.2%}")
        print(f"D-score range preservation: {d_range_preserved[0]:.2%}, {d_range_preserved[1]:.2%}")
        print(f"C-score range preservation: {c_range_preserved[0]:.2%}, {c_range_preserved[1]:.2%}")

def main():
    # Run evaluations with different density factors
    reports = []
    
    # Baseline (full density)
    reports.append(run_evaluation(1.0, "baseline"))
    
    # Half density (should give ~250-300 spans)
    reports.append(run_evaluation(0.5, "half_density"))
    
    # Quarter density (more aggressive reduction)
    reports.append(run_evaluation(0.25, "quarter_density"))
    
    # Compare results
    compare_results(reports)

if __name__ == "__main__":
    main()
