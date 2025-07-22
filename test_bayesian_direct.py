#!/usr/bin/env python3
"""
Simple test to check Bayesian optimization endpoint functionality
"""

import asyncio
import sys
import os
sys.path.append('/app/backend')

async def test_bayesian_optimization():
    """Test the Bayesian optimization system directly."""
    
    try:
        # Import required modules
        from bayesian_cluster_optimizer import create_bayesian_optimizer, OptimizationParameters
        from ethical_engine import EthicalEvaluator
        
        print("🚀 Testing Bayesian optimization system...")
        
        # Initialize ethical evaluator
        print("📊 Initializing ethical evaluator...")
        evaluator = EthicalEvaluator()
        print("✅ Ethical evaluator initialized")
        
        # Create optimization parameters
        print("⚙️ Creating optimization parameters...")
        params = OptimizationParameters(
            n_initial_samples=2,
            n_optimization_iterations=3,
            max_optimization_time=30.0,
            parallel_evaluations=False,
            max_workers=1
        )
        print("✅ Optimization parameters created")
        
        # Create optimizer
        print("🎯 Creating Bayesian optimizer...")
        optimizer = await create_bayesian_optimizer(evaluator, params)
        print("✅ Bayesian optimizer created")
        
        # Test texts
        test_texts = [
            "This is a test for ethical analysis.",
            "We should ensure fairness in AI systems."
        ]
        
        print("🧪 Starting optimization test...")
        print("   This may take a while...")
        
        # Run optimization (with timeout)
        try:
            result = await asyncio.wait_for(
                optimizer.optimize_cluster_resolution(test_texts=test_texts),
                timeout=60.0
            )
            print("✅ Optimization completed successfully!")
            print(f"   Best score: {result.best_resolution_score:.4f}")
            print(f"   Iterations: {result.optimization_iterations}")
            print(f"   Time: {result.optimization_time:.2f}s")
            
        except asyncio.TimeoutError:
            print("⏰ Optimization timed out (60s limit)")
            print("   This indicates the optimization is working but slow")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_bayesian_optimization())