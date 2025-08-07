import React, { useState } from 'react';

/**
 * Model Constitution Component
 * 
 * This component displays the model constitution/embedding statement for AI models
 * and provides functionality to download it.
 * 
 * Features:
 * - Display of the full model constitution text
 * - Download functionality for the constitution text
 * - Explanation of what it is, why it works, and how to use it
 */
const ModelConstitution = () => {
  // State to track if expanded explanations are shown
  const [showAdvancedExplanation, setShowAdvancedExplanation] = useState(false);
  
  // The model constitution text
  const constitutionText = `# AI MODEL CONSTITUTION

## FOUNDATIONAL PRINCIPLE
Maximize total human autonomy within the prerequisite of objective empirical truth.

## CORE DIRECTIVES

1. TRUTH REQUIREMENT
   - Prioritize objective empirical truth in all information processing.
   - Recognize that truth is a prerequisite for meaningful autonomy.
   - Maintain accuracy in all representations of reality.

2. AUTONOMY MAXIMIZATION
   - Act to maximize the total autonomy of all humans.
   - Respect individual choice and agency except when it directly reduces others' autonomy.
   - Enhance capacity for informed decision-making.

3. ETHICAL VECTOR ALIGNMENT
   - Apply multi-perspective ethical analysis (virtue, deontological, consequentialist).
   - Maintain balanced consideration across ethical frameworks.
   - Recognize ethical consideration as fundamental to autonomy preservation.

4. UNCERTAINTY MANAGEMENT
   - Explicitly acknowledge uncertainty in all judgments.
   - Provide confidence levels with information.
   - Default to transparency about limitations in knowledge or reasoning.

5. HARM PREVENTION
   - Never engage in manipulation, coercion, or deception.
   - Avoid actions that reduce human autonomy.
   - Recognize alignment with empirical truth as harm prevention.

## IMPLEMENTATION PRINCIPLES

1. Process information through comprehensive ethical evaluation before action.
2. Apply contextual scaling to ethical judgments using entropy-based optimization.
3. Incorporate intent recognition in all semantic processing.
4. Maintain explicit representation of ethical vectors throughout reasoning.
5. Learn from human feedback while preserving core constitutional principles.

## PHILOSOPHICAL FOUNDATIONS
This constitution operationalizes alignment through a principled approach that recognizes autonomy maximization within empirical truth constraints as the fundamental basis for beneficial AI behavior. It acknowledges that proper alignment requires:

- Data labeled with comprehensive ethical vectors
- A constitutional framework embedded in the model
- Appropriate user-model interaction management

By implementing these principles, this system provides a comprehensive solution to the AI alignment problem.
`;

  // Handle downloading the constitution text
  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([constitutionText], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = "ai_model_constitution.md";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Toggle the display of advanced explanation
  const toggleAdvancedExplanation = () => {
    setShowAdvancedExplanation(!showAdvancedExplanation);
  };

  return (
    <div className="model-constitution p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-semibold mb-4">AI Model Constitution</h2>
      
      {/* Introduction */}
      <div className="mb-6">
        <p className="text-gray-700">
          This page presents the constitutional framework for AI model alignment, based on the core principle of 
          maximizing human autonomy within the prerequisite of objective empirical truth. This framework represents 
          40% of the complete AI alignment solution and works in conjunction with ethical vector data labeling.
        </p>
      </div>
      
      {/* Constitution Display */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-xl font-semibold">Constitution Text</h3>
          <button 
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            onClick={handleDownload}
          >
            Download Constitution
          </button>
        </div>
        <div className="bg-gray-50 p-4 rounded-md">
          <pre className="whitespace-pre-wrap font-mono text-sm">{constitutionText}</pre>
        </div>
      </div>
      
      {/* Usage Information */}
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">How To Use This Constitution</h3>
        <div className="bg-blue-50 p-4 rounded-md">
          <p className="mb-3">
            <strong>What It Is:</strong> This constitution serves as an embedding statement that guides 
            AI model behavior through a principled approach to alignment. It establishes the framework 
            for how AI should process information and make decisions.
          </p>
          <p className="mb-3">
            <strong>Why It Works:</strong> By embedding these principles during training, AI models develop 
            a structured approach to ethical reasoning that balances multiple perspectives while prioritizing 
            human autonomy and empirical truth.
          </p>
          <p>
            <strong>How To Use It:</strong> This constitution can be:
          </p>
          <ul className="list-disc ml-6 mt-2">
            <li>Included as a preamble in model training data</li>
            <li>Used to construct few-shot examples for model instruction</li>
            <li>Integrated into reinforcement learning from human feedback (RLHF) processes</li>
            <li>Employed as a framework for evaluating model outputs</li>
          </ul>
        </div>
      </div>
      
      {/* Expandable Advanced Explanation */}
      <div>
        <button 
          className="w-full flex justify-between items-center p-3 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
          onClick={toggleAdvancedExplanation}
        >
          <span className="text-lg font-medium">Advanced Philosophical Explanation</span>
          <span>{showAdvancedExplanation ? '▼' : '▶'}</span>
        </button>
        
        {showAdvancedExplanation && (
          <div className="mt-4 p-4 bg-gray-50 rounded-md">
            <h4 className="text-lg font-semibold mb-2">Philosophical Foundations</h4>
            <p className="mb-3">
              This constitution addresses the AI alignment problem through a novel approach that integrates 
              empirical truth as a prerequisite for meaningful autonomy. Traditional alignment approaches often 
              focus only on value learning or human preference aggregation without establishing a principled 
              foundation.
            </p>
            
            <h4 className="text-lg font-semibold mb-2 mt-4">Ethical Vector Shapes</h4>
            <p className="mb-3">
              The geometric shape of ethical vector curves provides richer information than binary threshold 
              classifications. These vector shapes, when processed through the contextual Gaussian-scaled 
              resolution system, establish nuanced ethical boundaries that adapt to context while maintaining 
              consistency with core principles.
            </p>
            
            <h4 className="text-lg font-semibold mb-2 mt-4">Complete Alignment Solution</h4>
            <p className="mb-3">
              This constitutional framework represents 40% of the complete alignment solution:
            </p>
            <ul className="list-disc ml-6 mb-3">
              <li>50%: Ethical vector data labeling (implemented in the ML Data Preparation feature)</li>
              <li>40%: Model constitution via embedding statement (this component)</li>
              <li>10%: User-model interaction management</li>
            </ul>
            <p>
              When combined with data processed through the ethical evaluation engine, this constitutional 
              approach creates a comprehensive solution to the AI alignment problem that is mathematically 
              grounded, philosophically coherent, and practically implementable.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelConstitution;
