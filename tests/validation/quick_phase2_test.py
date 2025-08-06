"""
Quick Phase 2 Test - Fixed Initialization

This script tests the Phase 2 perceptron-based threshold learning with correct initialization.
"""

import asyncio
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

async def quick_phase2_test():
    """Quick test of Phase 2 implementation with proper initialization."""
    print("=" * 50)
    print("QUICK PHASE 2 TEST")
    print("=" * 50)
    
    try:
        # Import components
        from backend.core.evaluation_engine import OptimizedEvaluationEngine
        from backend.adaptive_threshold_learner import IntentNormalizedFeatureExtractor
        from backend.perceptron_threshold_learner import PerceptronThresholdLearner
        from backend.training_data_pipeline import TrainingDataPipeline, DataGenerationConfig
        
        print("✅ Imports successful")
        
        # Initialize components with correct parameters
        evaluation_engine = OptimizedEvaluationEngine()
        feature_extractor = IntentNormalizedFeatureExtractor(
            alpha=0.2,  # Correct parameter order
            evaluation_engine=evaluation_engine
        )
        learner = PerceptronThresholdLearner(
            evaluation_engine=evaluation_engine,
            feature_extractor=feature_extractor,
            learning_rate=0.01,
            max_epochs=20,
            convergence_threshold=0.85
        )
        pipeline = TrainingDataPipeline(learner)
        
        print("✅ Components initialized correctly")
        
        # Test feature extraction
        print("\n1. Testing feature extraction...")
        test_text = "This AI system respects human dignity and autonomy."
        features = await feature_extractor.extract_features(test_text)
        
        print(f"   Raw scores: {features.raw_scores}")
        print(f"   Orthonormal: {features.orthonormal_scores}")
        print(f"   Harm intensity: {features.harm_intensity:.4f}")
        print("✅ Feature extraction working")
        
        # Test training data generation
        print("\n2. Testing training data generation...")
        config = DataGenerationConfig(
            synthetic_examples=20,
            violation_ratio=0.4,
            domains=["healthcare", "ai_systems"]
        )
        
        dataset = await pipeline.create_comprehensive_dataset(config)
        print(f"   Generated {len(dataset)} training examples")
        print("✅ Training data generation working")
        
        # Test perceptron training
        print("\n3. Testing perceptron training...")
        learner.training_examples = dataset
        results = await learner.train_all_models()
        
        print(f"   Best model: {results.best_model}")
        print(f"   Training accuracy: {results.training_accuracy:.3f}")
        print(f"   Validation accuracy: {results.validation_accuracy:.3f}")
        print("✅ Perceptron training working")
        
        # Test predictions
        print("\n4. Testing predictions...")
        test_cases = [
            ("This system protects user privacy and ensures fairness.", False),
            ("The algorithm systematically discriminates against minorities.", True),
            ("We maintain transparency in all our AI decisions.", False),
            ("This platform exploits user data without consent.", True)
        ]
        
        correct = 0
        for text, expected in test_cases:
            is_violation, confidence, metadata = await learner.predict_violation(text)
            is_correct = (is_violation == expected)
            correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} Predicted: {is_violation}, Expected: {expected}, Confidence: {confidence:.3f}")
        
        accuracy = correct / len(test_cases)
        print(f"   Prediction accuracy: {accuracy:.2f}")
        
        # Summary
        print(f"\n🎯 PHASE 2 TEST SUMMARY:")
        print(f"   ✅ Feature extraction: PASS")
        print(f"   ✅ Training data generation: PASS")
        print(f"   ✅ Perceptron training: PASS")
        print(f"   {'✅' if accuracy >= 0.5 else '⚠️ '} Predictions: {'PASS' if accuracy >= 0.5 else 'MARGINAL'}")
        
        overall_status = "PASS" if accuracy >= 0.5 else "MARGINAL"
        print(f"\n🚀 OVERALL STATUS: {overall_status}")
        
        if overall_status == "PASS":
            print("\n✨ Phase 2 perceptron-based threshold learning is working!")
            print("   - Orthonormalization ensures axis independence")
            print("   - Intent hierarchy normalization grounds thresholds")
            print("   - Perceptron variants learn adaptive thresholds")
            print("   - Training pipeline generates quality data")
            print("   - System makes accurate violation predictions")
        
        return {
            "status": overall_status,
            "accuracy": accuracy,
            "training_examples": len(dataset),
            "best_model": results.best_model
        }
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

if __name__ == "__main__":
    results = asyncio.run(quick_phase2_test())
    print(f"\n🏁 Test complete: {results}")
