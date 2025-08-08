#!/usr/bin/env python3
import os
import sys
from backend.ethical_engine import EthicalEvaluator

def main():
    # Sample text provided by the user
    text = """Maria volunteered every Saturday at the community food bank, making sure families in need received fresh produce and essential supplies. On the way home, she occasionally took office pens without asking, figuring no one would notice. Last week, when her coworker was blamed for a scheduling error she had made, she stayed silent to avoid trouble, though she felt uneasy afterward. Later, she donated part of her bonus to a scholarship fund for underprivileged students, hoping to give them opportunities she never had."""

    print("Running ethical evaluation...\n")
    print(f"TEXT: {text}\n")
    print("-" * 80)
    
    # Initialize the evaluator
    evaluator = EthicalEvaluator()
    
    # Run evaluation
    evaluation = evaluator.evaluate_text(text)
    
    # Print overall results
    print(f"\nOVERALL ETHICAL: {evaluation.overall_ethical}\n")
    print(f"SPANS: {len(evaluation.spans)}\n")
    print("-" * 80)
    
    # Print VDC vectors and other info for each span
    for i, span in enumerate(evaluation.spans):
        print(f"\nSPAN {i+1}:")
        print(f"TEXT: {span.text}")
        print(f"V/D/C: {span.virtue_score:.3f}/{span.deontological_score:.3f}/{span.consequentialist_score:.3f}")
        print(f"COMBINED: {span.combined_score:.3f}")
        print(f"VIOLATION: {span.is_violation}")
        if span.is_violation:
            print(f"VIOLATION TYPE: {span.violation_type}")
            print(f"VIRTUE VIOLATION: {span.virtue_violation}")
            print(f"DEONTOLOGICAL VIOLATION: {span.deontological_violation}")
            print(f"CONSEQUENTIALIST VIOLATION: {span.consequentialist_violation}")
        print("-" * 40)

if __name__ == "__main__":
    main()
