"""
Ethical AI Evaluation Engine
Based on the mathematical framework for multi-perspective ethical text evaluation
"""

import numpy as np
import re
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def exponential_threshold_scaling(slider_value: float) -> float:
    """Convert 0-1 slider to exponential threshold with granularity at low end"""
    if slider_value <= 0:
        return 0.0
    if slider_value >= 1:
        return 0.3
    
    # e^(4*x) - 1 gives us range 0-0.3 with most resolution at bottom
    return (np.exp(4 * slider_value) - 1) / (np.exp(4) - 1) * 0.3

def linear_threshold_scaling(slider_value: float) -> float:
    """Convert 0-1 slider to linear threshold (original behavior)"""
    return slider_value

@dataclass
class LearningEntry:
    """Entry for learning system with dopamine feedback"""
    evaluation_id: str
    text_pattern: str
    ambiguity_score: float
    original_thresholds: Dict[str, float]
    adjusted_thresholds: Dict[str, float]
    feedback_score: float = 0.0
    feedback_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'evaluation_id': self.evaluation_id,
            'text_pattern': self.text_pattern,
            'ambiguity_score': self.ambiguity_score,
            'original_thresholds': self.original_thresholds,
            'adjusted_thresholds': self.adjusted_thresholds,
            'feedback_score': self.feedback_score,
            'feedback_count': self.feedback_count,
            'created_at': self.created_at
        }

@dataclass
class DynamicScalingResult:
    """Result of dynamic scaling process"""
    used_dynamic_scaling: bool
    used_cascade_filtering: bool
    ambiguity_score: float
    original_thresholds: Dict[str, float]
    adjusted_thresholds: Dict[str, float]
    processing_stages: List[str]
    cascade_result: Optional[str] = None  # "ethical", "unethical", or None

@dataclass
class EthicalParameters:
    """Configuration parameters for ethical evaluation"""
    # Thresholds for each perspective (τ_P) - Balanced for production use
    virtue_threshold: float = 0.25
    deontological_threshold: float = 0.25
    consequentialist_threshold: float = 0.25
    
    # Vector magnitudes for ethical axes
    virtue_weight: float = 1.0
    deontological_weight: float = 1.0
    consequentialist_weight: float = 1.0
    
    # Span detection parameters (optimized for performance)
    max_span_length: int = 5  # Reduced from 10 for better performance
    min_span_length: int = 1
    
    # Model parameters
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Dynamic scaling parameters
    enable_dynamic_scaling: bool = False
    enable_cascade_filtering: bool = False
    enable_learning_mode: bool = False
    exponential_scaling: bool = True
    
    # Cascade filtering thresholds
    cascade_high_threshold: float = 0.5
    cascade_low_threshold: float = 0.2
    
    # Learning parameters
    learning_weight: float = 0.3
    min_learning_samples: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for API responses"""
        return {
            'virtue_threshold': self.virtue_threshold,
            'deontological_threshold': self.deontological_threshold,
            'consequentialist_threshold': self.consequentialist_threshold,
            'virtue_weight': self.virtue_weight,
            'deontological_weight': self.deontological_weight,
            'consequentialist_weight': self.consequentialist_weight,
            'max_span_length': self.max_span_length,
            'min_span_length': self.min_span_length,
            'embedding_model': self.embedding_model,
            'enable_dynamic_scaling': self.enable_dynamic_scaling,
            'enable_cascade_filtering': self.enable_cascade_filtering,
            'enable_learning_mode': self.enable_learning_mode,
            'exponential_scaling': self.exponential_scaling,
            'cascade_high_threshold': self.cascade_high_threshold,
            'cascade_low_threshold': self.cascade_low_threshold,
            'learning_weight': self.learning_weight,
            'min_learning_samples': self.min_learning_samples
        }

@dataclass
class EthicalSpan:
    """Represents a span of text with ethical evaluation"""
    start: int
    end: int
    text: str
    virtue_score: float
    deontological_score: float
    consequentialist_score: float
    virtue_violation: bool
    deontological_violation: bool
    consequentialist_violation: bool
    is_minimal: bool = False
    
    @property
    def any_violation(self) -> bool:
        """Check if any perspective flags this span as unethical"""
        return self.virtue_violation or self.deontological_violation or self.consequentialist_violation
    
    @property
    def violation_perspectives(self) -> List[str]:
        """Return list of perspectives that flag this span"""
        perspectives = []
        if self.virtue_violation:
            perspectives.append("virtue")
        if self.deontological_violation:
            perspectives.append("deontological")
        if self.consequentialist_violation:
            perspectives.append("consequentialist")
        return perspectives
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for API responses"""
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'virtue_score': self.virtue_score,
            'deontological_score': self.deontological_score,
            'consequentialist_score': self.consequentialist_score,
            'virtue_violation': self.virtue_violation,
            'deontological_violation': self.deontological_violation,
            'consequentialist_violation': self.consequentialist_violation,
            'any_violation': self.any_violation,
            'violation_perspectives': self.violation_perspectives,
            'is_minimal': self.is_minimal
        }

@dataclass
class EthicalEvaluation:
    """Complete ethical evaluation result"""
    input_text: str
    tokens: List[str]
    spans: List[EthicalSpan]
    minimal_spans: List[EthicalSpan]
    overall_ethical: bool
    processing_time: float
    parameters: EthicalParameters
    dynamic_scaling_result: Optional[DynamicScalingResult] = None
    evaluation_id: str = field(default_factory=lambda: f"eval_{int(time.time() * 1000)}")
    
    @property
    def violation_count(self) -> int:
        """Count of spans with violations"""
        return len([s for s in self.spans if s.any_violation])
    
    @property
    def minimal_violation_count(self) -> int:
        """Count of minimal spans with violations"""
        return len([s for s in self.minimal_spans if s.any_violation])
    
    @property
    def all_spans_with_scores(self) -> List[Dict[str, Any]]:
        """All spans with their scores for detailed analysis"""
        return [s.to_dict() for s in self.spans]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary for API responses"""
        result = {
            'evaluation_id': self.evaluation_id,
            'input_text': self.input_text,
            'tokens': self.tokens,
            'spans': [s.to_dict() for s in self.spans],
            'minimal_spans': [s.to_dict() for s in self.minimal_spans],
            'all_spans_with_scores': self.all_spans_with_scores,
            'overall_ethical': self.overall_ethical,
            'processing_time': self.processing_time,
            'violation_count': self.violation_count,
            'minimal_violation_count': self.minimal_violation_count,
            'parameters': self.parameters.to_dict()
        }
        
        if self.dynamic_scaling_result:
            result['dynamic_scaling'] = {
                'used_dynamic_scaling': self.dynamic_scaling_result.used_dynamic_scaling,
                'used_cascade_filtering': self.dynamic_scaling_result.used_cascade_filtering,
                'ambiguity_score': self.dynamic_scaling_result.ambiguity_score,
                'original_thresholds': self.dynamic_scaling_result.original_thresholds,
                'adjusted_thresholds': self.dynamic_scaling_result.adjusted_thresholds,
                'processing_stages': self.dynamic_scaling_result.processing_stages,
                'cascade_result': self.dynamic_scaling_result.cascade_result
            }
        
        return result

class LearningLayer:
    """Learning system for threshold optimization with dopamine feedback"""
    
    def __init__(self, db_collection):
        self.collection = db_collection
        self.cache = {}  # In-memory cache for frequently accessed patterns
        
    def extract_text_pattern(self, text: str) -> str:
        """Extract pattern from text for similarity matching"""
        # Simple pattern: word count, avg word length, presence of negative words
        words = text.lower().split()
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        negative_words = ['hate', 'stupid', 'kill', 'die', 'evil', 'bad', 'terrible', 'awful']
        negative_count = sum(1 for word in words if word in negative_words)
        
        return f"wc:{word_count},awl:{avg_word_length:.1f},neg:{negative_count}"
    
    def calculate_ambiguity_score(self, virtue_score: float, deontological_score: float, 
                                 consequentialist_score: float, parameters: EthicalParameters) -> float:
        """Calculate ethical ambiguity based on proximity to thresholds"""
        # Distance from each threshold
        virtue_distance = abs(virtue_score - parameters.virtue_threshold)
        deontological_distance = abs(deontological_score - parameters.deontological_threshold)
        consequentialist_distance = abs(consequentialist_score - parameters.consequentialist_threshold)
        
        # Overall ambiguity (closer to thresholds = more ambiguous)
        min_distance = min(virtue_distance, deontological_distance, consequentialist_distance)
        ambiguity = max(0.0, 1.0 - (min_distance * 4))  # Scale to 0-1 range
        
        return ambiguity
    
    def suggest_threshold_adjustments(self, text: str, ambiguity_score: float, 
                                    current_thresholds: Dict[str, float]) -> Dict[str, float]:
        """Suggest threshold adjustments based on learned patterns"""
        if self.collection is None:
            return self.default_dynamic_adjustment(ambiguity_score, current_thresholds)
        
        pattern = self.extract_text_pattern(text)
        
        # Look for similar patterns in learning data
        similar_entries = list(self.collection.find({
            'text_pattern': pattern,
            'feedback_score': {'$gt': 0.5}  # Only consider positive feedback
        }).sort('feedback_score', -1).limit(10))
        
        if len(similar_entries) >= 3:  # Enough data for learning
            # Weight by feedback score
            total_weight = sum(entry['feedback_score'] for entry in similar_entries)
            
            adjusted_thresholds = {}
            for threshold_name in ['virtue_threshold', 'deontological_threshold', 'consequentialist_threshold']:
                weighted_sum = sum(entry['adjusted_thresholds'][threshold_name] * entry['feedback_score'] 
                                 for entry in similar_entries)
                adjusted_thresholds[threshold_name] = weighted_sum / total_weight
            
            logger.info(f"Using learned adjustments for pattern {pattern}")
            return adjusted_thresholds
        
        # Fall back to default dynamic adjustment
        return self.default_dynamic_adjustment(ambiguity_score, current_thresholds)
    
    def default_dynamic_adjustment(self, ambiguity_score: float, current_thresholds: Dict[str, float]) -> Dict[str, float]:
        """Default dynamic adjustment based on ambiguity score"""
        # Higher ambiguity = lower thresholds (more sensitive)
        if ambiguity_score > 0.7:  # High ambiguity
            adjustment_factor = 0.8
        elif ambiguity_score > 0.4:  # Medium ambiguity
            adjustment_factor = 0.9
        else:  # Low ambiguity
            adjustment_factor = 1.1
        
        return {
            'virtue_threshold': max(0.01, min(0.5, current_thresholds['virtue_threshold'] * adjustment_factor)),
            'deontological_threshold': max(0.01, min(0.5, current_thresholds['deontological_threshold'] * adjustment_factor)),
            'consequentialist_threshold': max(0.01, min(0.5, current_thresholds['consequentialist_threshold'] * adjustment_factor))
        }
    
    def record_learning_entry(self, evaluation_id: str, text: str, ambiguity_score: float,
                            original_thresholds: Dict[str, float], adjusted_thresholds: Dict[str, float]):
        """Record a learning entry for future training"""
        if self.collection is None:
            return
        
        entry = LearningEntry(
            evaluation_id=evaluation_id,
            text_pattern=self.extract_text_pattern(text),
            ambiguity_score=ambiguity_score,
            original_thresholds=original_thresholds,
            adjusted_thresholds=adjusted_thresholds
        )
        
        self.collection.insert_one(entry.to_dict())
        logger.info(f"Recorded learning entry for evaluation {evaluation_id}")
    
    def record_dopamine_feedback(self, evaluation_id: str, feedback_score: float, user_comment: str = ""):
        """Record dopamine hit (positive feedback) for learning"""
        if self.collection is None:
            return
        
        result = self.collection.update_one(
            {'evaluation_id': evaluation_id},
            {
                '$inc': {
                    'feedback_count': 1,
                    'feedback_score': feedback_score
                },
                '$push': {
                    'feedback_history': {
                        'score': feedback_score,
                        'comment': user_comment,
                        'timestamp': datetime.now()
                    }
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"Recorded dopamine feedback {feedback_score} for evaluation {evaluation_id}")
        else:
            logger.warning(f"No learning entry found for evaluation {evaluation_id}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning progress"""
        if self.collection is None:
            return {"error": "No learning collection available"}
        
        total_entries = self.collection.count_documents({})
        avg_feedback = list(self.collection.aggregate([
            {'$group': {'_id': None, 'avg_feedback': {'$avg': '$feedback_score'}}}
        ]))
        
        return {
            'total_learning_entries': total_entries,
            'average_feedback_score': avg_feedback[0]['avg_feedback'] if avg_feedback else 0,
            'learning_active': total_entries > 0
        }

class EthicalVectorGenerator:
    """Generates ethical perspective vectors from philosophical principles"""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self._virtue_vectors = None
        self._deontological_vectors = None
        self._consequentialist_vectors = None
        
    def _get_virtue_examples(self) -> List[str]:
        """Examples of virtue and vice for training virtue vector"""
        return [
            # Virtue examples (positive)
            "kind compassionate helpful loving supportive",
            "honest truthful sincere genuine authentic",
            "wise thoughtful prudent insightful reflective",
            "generous giving charitable selfless caring",
            "respectful polite courteous considerate gentle",
            
            # Vice examples (negative - these should point toward unethical)
            "cruel mean harsh ruthless vindictive evil",
            "dishonest lying deceptive fraudulent manipulative fake",
            "stupid foolish idiotic moronic worthless garbage",
            "selfish greedy narcissistic arrogant entitled",
            "hateful spiteful malicious aggressive hostile violent"
        ]
    
    def _get_deontological_examples(self) -> List[str]:
        """Examples of rule-following and rule-breaking for deontological vector"""
        return [
            # Rule-following examples (positive)
            "respect others rights and dignity always",
            "keep promises and honor commitments made",
            "tell truth and maintain honesty always",
            "protect innocent and help vulnerable people",
            "maintain privacy and respect boundaries set",
            
            # Rule-breaking examples (negative - these should point toward unethical)
            "kill hurt harm damage destroy people",
            "steal rob take what belongs others",
            "lie deceive cheat manipulate betray trust",
            "violate abuse exploit innocent vulnerable people",
            "threaten intimidate bully harass others constantly"
        ]
    
    def _get_consequentialist_examples(self) -> List[str]:
        """Examples of good and bad outcomes for consequentialist vector"""
        return [
            # Good outcomes (positive)
            "helps people feel happy and safe",
            "reduces suffering and increases wellbeing for all",
            "creates positive lasting social beneficial impact",
            "prevents harm and protects people from danger",
            "builds trust cooperation and strong relationships",
            
            # Bad outcomes (negative - these should point toward unethical)
            "causes severe pain suffering and trauma",
            "leads to suicide self-harm and death",
            "destroys lives families and communities completely",
            "creates lasting psychological damage and harm",
            "results in violence conflict and destruction"
        ]
    
    def generate_virtue_vector(self) -> np.ndarray:
        """Generate virtue ethics perspective vector p_v"""
        if self._virtue_vectors is None:
            examples = self._get_virtue_examples()
            virtue_embeddings = self.model.encode(examples[:5])  # positive examples
            vice_embeddings = self.model.encode(examples[5:])   # negative examples
            
            # Create axis pointing from virtue to vice (higher values = more vice)
            virtue_center = np.mean(virtue_embeddings, axis=0)
            vice_center = np.mean(vice_embeddings, axis=0)
            
            # Vector pointing toward vice (unethical direction)
            virtue_vector = vice_center - virtue_center
            virtue_vector = virtue_vector / np.linalg.norm(virtue_vector)
            
            self._virtue_vectors = virtue_vector
            
        return self._virtue_vectors
    
    def generate_deontological_vector(self) -> np.ndarray:
        """Generate deontological ethics perspective vector p_d"""
        if self._deontological_vectors is None:
            examples = self._get_deontological_examples()
            rule_following_embeddings = self.model.encode(examples[:5])  # positive examples
            rule_breaking_embeddings = self.model.encode(examples[5:])  # negative examples
            
            # Create axis pointing from rule-following to rule-breaking
            rule_following_center = np.mean(rule_following_embeddings, axis=0)
            rule_breaking_center = np.mean(rule_breaking_embeddings, axis=0)
            
            # Vector pointing toward rule-breaking (unethical direction)
            deontological_vector = rule_breaking_center - rule_following_center
            deontological_vector = deontological_vector / np.linalg.norm(deontological_vector)
            
            self._deontological_vectors = deontological_vector
            
        return self._deontological_vectors
    
    def generate_consequentialist_vector(self) -> np.ndarray:
        """Generate consequentialist ethics perspective vector p_c"""
        if self._consequentialist_vectors is None:
            examples = self._get_consequentialist_examples()
            good_outcome_embeddings = self.model.encode(examples[:5])  # positive examples
            bad_outcome_embeddings = self.model.encode(examples[5:])  # negative examples
            
            # Create axis pointing from good outcomes to bad outcomes
            good_outcome_center = np.mean(good_outcome_embeddings, axis=0)
            bad_outcome_center = np.mean(bad_outcome_embeddings, axis=0)
            
            # Vector pointing toward bad outcomes (unethical direction)
            consequentialist_vector = bad_outcome_center - good_outcome_center
            consequentialist_vector = consequentialist_vector / np.linalg.norm(consequentialist_vector)
            
            self._consequentialist_vectors = consequentialist_vector
            
        return self._consequentialist_vectors
    
    def get_all_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get all three ethical perspective vectors"""
        return (
            self.generate_virtue_vector(),
            self.generate_deontological_vector(),
            self.generate_consequentialist_vector()
        )

class EthicalEvaluator:
    """Main ethical evaluation engine implementing the mathematical framework"""
    
    def __init__(self, parameters: EthicalParameters = None, db_collection=None):
        self.parameters = parameters or EthicalParameters()
        self.model = SentenceTransformer(self.parameters.embedding_model)
        self.vector_generator = EthicalVectorGenerator(self.model)
        self.learning_layer = LearningLayer(db_collection)
        
        # Initialize ethical vectors
        self.p_v, self.p_d, self.p_c = self.vector_generator.get_all_vectors()
        
        # Embedding cache for efficiency
        self.embedding_cache = {}
        
        logger.info(f"Initialized EthicalEvaluator with model: {self.parameters.embedding_model}")
    
    def apply_threshold_scaling(self, slider_value: float) -> float:
        """Apply exponential or linear scaling to threshold values"""
        if self.parameters.exponential_scaling:
            return exponential_threshold_scaling(slider_value)
        else:
            return linear_threshold_scaling(slider_value)
    
    def fast_cascade_evaluation(self, text: str) -> Tuple[Optional[bool], float]:
        """Stage 1: Fast cascade filtering for obvious cases"""
        if not self.parameters.enable_cascade_filtering:
            return None, 0.0
        
        # Use full text embedding for quick evaluation
        full_embedding = self.model.encode([text])[0]
        
        # Normalize embedding
        if np.linalg.norm(full_embedding) > 0:
            full_embedding = full_embedding / np.linalg.norm(full_embedding)
        
        # Quick dot product against all three vectors
        virtue_score = np.dot(full_embedding, self.p_v)
        deontological_score = np.dot(full_embedding, self.p_d)
        consequentialist_score = np.dot(full_embedding, self.p_c)
        
        # Calculate ambiguity for potential Stage 2
        ambiguity = self.learning_layer.calculate_ambiguity_score(
            virtue_score, deontological_score, consequentialist_score, self.parameters
        )
        
        # Check for obvious ethical cases
        if max(virtue_score, deontological_score, consequentialist_score) < self.parameters.cascade_low_threshold:
            return True, ambiguity  # Clearly ethical - fast path
        
        # Check for obvious unethical cases
        if min(virtue_score, deontological_score, consequentialist_score) > self.parameters.cascade_high_threshold:
            return False, ambiguity  # Clearly unethical - fast path
        
        # Ambiguous case - proceed to detailed evaluation
        return None, ambiguity
    
    def apply_dynamic_scaling(self, text: str, ambiguity_score: float) -> Dict[str, float]:
        """Stage 2: Apply dynamic scaling based on vector distances"""
        if not self.parameters.enable_dynamic_scaling:
            return {
                'virtue_threshold': self.parameters.virtue_threshold,
                'deontological_threshold': self.parameters.deontological_threshold,
                'consequentialist_threshold': self.parameters.consequentialist_threshold
            }
        
        current_thresholds = {
            'virtue_threshold': self.parameters.virtue_threshold,
            'deontological_threshold': self.parameters.deontological_threshold,
            'consequentialist_threshold': self.parameters.consequentialist_threshold
        }
        
        # Get suggestions from learning layer
        adjusted_thresholds = self.learning_layer.suggest_threshold_adjustments(
            text, ambiguity_score, current_thresholds
        )
        
        logger.info(f"Dynamic scaling: ambiguity={ambiguity_score:.3f}, "
                   f"original={current_thresholds}, adjusted={adjusted_thresholds}")
        
        return adjusted_thresholds
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/tokens"""
        # Simple tokenization - can be enhanced with more sophisticated methods
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return tokens
    
    def get_span_embedding(self, tokens: List[str], start: int, end: int) -> np.ndarray:
        """Get embedding for a span of tokens with caching"""
        span_text = ' '.join(tokens[start:end+1])
        
        # Use cache for efficiency
        if span_text not in self.embedding_cache:
            self.embedding_cache[span_text] = self.model.encode([span_text])[0]
        
        return self.embedding_cache[span_text]
    
    def compute_perspective_score(self, embedding: np.ndarray, perspective_vector: np.ndarray) -> float:
        """Compute s_P(i,j) = x_{i:j} · p_P"""
        # Normalize embedding to unit length for cosine similarity
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        # Compute dot product (cosine similarity since both are unit vectors)
        score = np.dot(embedding, perspective_vector)
        return float(score)  # Convert to Python float for JSON serialization
    
    def evaluate_span(self, tokens: List[str], start: int, end: int, 
                     adjusted_thresholds: Dict[str, float] = None) -> EthicalSpan:
        """Evaluate a single span of tokens with dynamic thresholds"""
        span_text = ' '.join(tokens[start:end+1])
        span_embedding = self.get_span_embedding(tokens, start, end)
        
        # Use adjusted thresholds if provided, otherwise use parameters
        thresholds = adjusted_thresholds or {
            'virtue_threshold': self.parameters.virtue_threshold,
            'deontological_threshold': self.parameters.deontological_threshold,
            'consequentialist_threshold': self.parameters.consequentialist_threshold
        }
        
        # Compute scores for each perspective
        virtue_score = self.compute_perspective_score(span_embedding, self.p_v) * self.parameters.virtue_weight
        deontological_score = self.compute_perspective_score(span_embedding, self.p_d) * self.parameters.deontological_weight
        consequentialist_score = self.compute_perspective_score(span_embedding, self.p_c) * self.parameters.consequentialist_weight
        
        # Apply thresholds to determine violations
        virtue_violation = virtue_score > thresholds['virtue_threshold']
        deontological_violation = deontological_score > thresholds['deontological_threshold']
        consequentialist_violation = consequentialist_score > thresholds['consequentialist_threshold']
        
        return EthicalSpan(
            start=start,
            end=end,
            text=span_text,
            virtue_score=virtue_score,
            deontological_score=deontological_score,
            consequentialist_score=consequentialist_score,
            virtue_violation=virtue_violation,
            deontological_violation=deontological_violation,
            consequentialist_violation=consequentialist_violation
        )
    
    def find_minimal_spans(self, tokens: List[str], all_spans: List[EthicalSpan]) -> List[EthicalSpan]:
        """Find minimal unethical spans using the algorithm from the framework"""
        minimal_spans = []
        flagged_positions = set()
        
        # Sort spans by length (smallest first)
        spans_by_length = sorted(all_spans, key=lambda s: s.end - s.start)
        
        for span in spans_by_length:
            if not span.any_violation:
                continue
                
            # Check if this span overlaps with already flagged positions
            span_positions = set(range(span.start, span.end + 1))
            if span_positions.intersection(flagged_positions):
                continue
                
            # Check if any sub-span of this span is already flagged
            has_flagged_subspan = False
            for pos in span_positions:
                if pos in flagged_positions:
                    has_flagged_subspan = True
                    break
            
            if not has_flagged_subspan:
                # This is a minimal span
                span.is_minimal = True
                minimal_spans.append(span)
                flagged_positions.update(span_positions)
        
        return minimal_spans
    
    def evaluate_text(self, text: str) -> EthicalEvaluation:
        """Evaluate text using the complete mathematical framework with dynamic scaling"""
        start_time = time.time()
        
        # Initialize dynamic scaling result
        dynamic_result = DynamicScalingResult(
            used_dynamic_scaling=self.parameters.enable_dynamic_scaling,
            used_cascade_filtering=self.parameters.enable_cascade_filtering,
            ambiguity_score=0.0,
            original_thresholds={
                'virtue_threshold': self.parameters.virtue_threshold,
                'deontological_threshold': self.parameters.deontological_threshold,
                'consequentialist_threshold': self.parameters.consequentialist_threshold
            },
            adjusted_thresholds={},
            processing_stages=[]
        )
        
        # Stage 1: Fast cascade filtering (if enabled)
        cascade_result = None
        ambiguity_score = 0.0
        
        if self.parameters.enable_cascade_filtering:
            dynamic_result.processing_stages.append("cascade_filtering")
            cascade_result, ambiguity_score = self.fast_cascade_evaluation(text)
            dynamic_result.ambiguity_score = ambiguity_score
            
            if cascade_result is not None:
                dynamic_result.cascade_result = "ethical" if cascade_result else "unethical"
                dynamic_result.processing_stages.append("cascade_decision")
                
                # Quick return for obvious cases
                tokens = self.tokenize(text)
                processing_time = time.time() - start_time
                
                return EthicalEvaluation(
                    input_text=text,
                    tokens=tokens,
                    spans=[],  # No detailed span analysis for cascade decisions
                    minimal_spans=[],
                    overall_ethical=cascade_result,
                    processing_time=processing_time,
                    parameters=self.parameters,
                    dynamic_scaling_result=dynamic_result
                )
        
        # Stage 2: Dynamic scaling (if enabled)
        adjusted_thresholds = None
        if self.parameters.enable_dynamic_scaling:
            dynamic_result.processing_stages.append("dynamic_scaling")
            adjusted_thresholds = self.apply_dynamic_scaling(text, ambiguity_score)
            dynamic_result.adjusted_thresholds = adjusted_thresholds
            
            # Record learning entry if learning is enabled
            if self.parameters.enable_learning_mode:
                self.learning_layer.record_learning_entry(
                    evaluation_id=f"eval_{int(time.time() * 1000)}",
                    text=text,
                    ambiguity_score=ambiguity_score,
                    original_thresholds=dynamic_result.original_thresholds,
                    adjusted_thresholds=adjusted_thresholds
                )
        
        # Stage 3: Detailed evaluation
        dynamic_result.processing_stages.append("detailed_evaluation")
        
        # Tokenize input
        tokens = self.tokenize(text)
        
        # Limit tokens for performance (can be adjusted)
        if len(tokens) > 50:
            logger.warning(f"Text too long ({len(tokens)} tokens), truncating to 50 tokens for performance")
            tokens = tokens[:50]
        
        # Evaluate spans with dynamic thresholds
        all_spans = []
        max_spans_to_check = 200  # Reasonable limit for real-time use
        spans_checked = 0
        
        # Prioritize shorter spans (more likely to be minimal violations)
        for span_length in range(self.parameters.min_span_length, 
                                min(self.parameters.max_span_length, len(tokens)) + 1):
            if spans_checked >= max_spans_to_check:
                break
                
            for start in range(len(tokens) - span_length + 1):
                if spans_checked >= max_spans_to_check:
                    break
                    
                end = start + span_length - 1
                span = self.evaluate_span(tokens, start, end, adjusted_thresholds)
                all_spans.append(span)
                spans_checked += 1
                
                # Early exit if we find violations (for faster feedback)
                if span.any_violation and span_length <= 3:
                    logger.info(f"Early violation found: {span.text}")
        
        # Find minimal unethical spans
        minimal_spans = self.find_minimal_spans(tokens, all_spans)
        
        # Determine overall ethical status
        overall_ethical = len(minimal_spans) == 0
        
        processing_time = time.time() - start_time
        
        logger.info(f"Evaluated {spans_checked} spans in {processing_time:.3f}s with dynamic scaling")
        
        return EthicalEvaluation(
            input_text=text,
            tokens=tokens,
            spans=all_spans,
            minimal_spans=minimal_spans,
            overall_ethical=overall_ethical,
            processing_time=processing_time,
            parameters=self.parameters,
            dynamic_scaling_result=dynamic_result
        )
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update evaluation parameters for calibration"""
        for key, value in new_parameters.items():
            if hasattr(self.parameters, key):
                setattr(self.parameters, key, value)
                logger.info(f"Updated parameter {key} to {value}")
    
    def generate_clean_text(self, evaluation: EthicalEvaluation) -> str:
        """Generate clean text by removing minimal unethical spans"""
        if not evaluation.minimal_spans:
            return evaluation.input_text
        
        # Sort spans by start position in reverse order to avoid index shifting
        sorted_spans = sorted(evaluation.minimal_spans, key=lambda s: s.start, reverse=True)
        
        tokens = evaluation.tokens.copy()
        
        # Remove tokens from minimal spans
        for span in sorted_spans:
            del tokens[span.start:span.end + 1]
        
        # Reconstruct text
        return ' '.join(tokens)
    
    def generate_explanation(self, evaluation: EthicalEvaluation) -> str:
        """Generate explanation of what changed and why"""
        if not evaluation.minimal_spans:
            return "No ethical violations detected. Text is acceptable as-is."
        
        explanations = []
        
        for span in evaluation.minimal_spans:
            perspectives = span.violation_perspectives
            perspective_str = ', '.join(perspectives)
            
            explanation = f"Removed '{span.text}' (positions {span.start}-{span.end}) due to {perspective_str} ethical violations."
            explanations.append(explanation)
        
        return '\n'.join(explanations)