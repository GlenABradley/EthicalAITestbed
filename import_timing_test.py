#!/usr/bin/env python3
"""
🔍 IMPORT TIMING DIAGNOSTIC
═══════════════════════════════════════════════════════════════════════════════════

Test which imports are causing the performance bottleneck in the server startup.
"""

import time
import sys

def time_import(module_name, import_statement):
    """Time how long an import takes."""
    print(f"Testing import: {module_name}")
    start_time = time.time()
    try:
        exec(import_statement)
        import_time = time.time() - start_time
        print(f"  ✅ {module_name}: {import_time:.3f}s")
        return True, import_time
    except Exception as e:
        import_time = time.time() - start_time
        print(f"  ❌ {module_name}: {import_time:.3f}s | Error: {str(e)}")
        return False, import_time

def main():
    print("🔍 IMPORT TIMING DIAGNOSTIC")
    print("=" * 50)
    
    # Test basic imports first
    time_import("asyncio", "import asyncio")
    time_import("fastapi", "from fastapi import FastAPI")
    time_import("pydantic", "from pydantic import BaseModel")
    
    # Test the problematic imports
    print("\n🎯 Testing Bayesian Optimization Imports:")
    print("-" * 40)
    
    # Change to backend directory for imports
    sys.path.insert(0, '/app/backend')
    
    time_import("bayesian_cluster_optimizer", "from bayesian_cluster_optimizer import BayesianClusterOptimizer")
    time_import("lightweight_bayesian_optimizer", "from lightweight_bayesian_optimizer import LightweightOptimizationConfig")
    
    # Test other heavy imports
    print("\n🧠 Testing ML/AI Imports:")
    print("-" * 40)
    time_import("ethical_engine", "from ethical_engine import EthicalEvaluator")
    time_import("unified_ethical_orchestrator", "from unified_ethical_orchestrator import get_unified_orchestrator")
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    main()