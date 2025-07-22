import React, { useState } from 'react';
import axios from 'axios';

/**
 * ML Training Assistant Component
 * 
 * Provides a comprehensive interface for the Enhanced Ethics Pipeline (Phase 5)
 * integrating philosophical foundations with practical ML training guidance.
 * 
 * Features:
 * - Three-layer ethical analysis (Meta, Normative, Applied)
 * - ML-specific training guidance
 * - Visual representation of philosophical frameworks
 * - Practical recommendations for AI development
 */
const MLTrainingAssistant = ({ backendUrl }) => {
  const [activeAnalysisTab, setActiveAnalysisTab] = useState('comprehensive');
  const [analysisInput, setAnalysisInput] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisDepth, setAnalysisDepth] = useState('standard');
  const [guidanceType, setGuidanceType] = useState('comprehensive');
  
  const API = `${backendUrl}/api`;

  const performEthicsAnalysis = async (endpoint, payload) => {
    try {
      setAnalysisLoading(true);
      const response = await axios.post(`${API}/ethics/${endpoint}`, payload);
      setAnalysisResult(response.data);
    } catch (error) {
      console.error('Ethics analysis failed:', error);
      setAnalysisResult({
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleComprehensiveAnalysis = () => {
    if (!analysisInput.trim()) {
      alert('Please enter content to analyze');
      return;
    }

    performEthicsAnalysis('comprehensive-analysis', {
      text: analysisInput,
      depth: analysisDepth,
      context: { domain: 'ml_development' }
    });
  };

  const handleMetaAnalysis = () => {
    if (!analysisInput.trim()) {
      alert('Please enter content to analyze');
      return;
    }

    performEthicsAnalysis('meta-analysis', {
      text: analysisInput,
      context: { focus: 'philosophical_structure' }
    });
  };

  const handleNormativeAnalysis = () => {
    if (!analysisInput.trim()) {
      alert('Please enter content to analyze');
      return;
    }

    performEthicsAnalysis('normative-analysis', {
      text: analysisInput,
      framework: 'all',
      context: { analysis_type: 'multi_framework' }
    });
  };

  const handleAppliedAnalysis = () => {
    if (!analysisInput.trim()) {
      alert('Please enter content to analyze');
      return;
    }

    performEthicsAnalysis('applied-analysis', {
      text: analysisInput,
      domain: 'auto',
      context: { focus: 'practical_recommendations' }
    });
  };

  const handleMLGuidance = () => {
    if (!analysisInput.trim()) {
      alert('Please enter training content to analyze');
      return;
    }

    performEthicsAnalysis('ml-training-guidance', {
      content: analysisInput,
      type: guidanceType,
      training_context: { 
        stage: 'development',
        model_type: 'general',
        use_case: 'ethical_ai'
      }
    });
  };

  const renderPhilosophicalInsight = (title, value, description) => (
    <div className="bg-gray-50 p-3 rounded-md">
      <div className="flex justify-between items-center mb-1">
        <span className="font-medium text-gray-700">{title}</span>
        <span className={`px-2 py-1 rounded text-xs font-medium ${
          typeof value === 'boolean' 
            ? (value ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800')
            : typeof value === 'number'
            ? value > 0.7 ? 'bg-green-100 text-green-800' 
              : value > 0.4 ? 'bg-yellow-100 text-yellow-800' 
              : 'bg-red-100 text-red-800'
            : 'bg-gray-100 text-gray-800'
        }`}>
          {typeof value === 'boolean' ? (value ? 'PASS' : 'FAIL') 
           : typeof value === 'number' ? value.toFixed(3)
           : value}
        </span>
      </div>
      {description && <div className="text-sm text-gray-600">{description}</div>}
    </div>
  );

  const renderComprehensiveResults = (analysis) => {
    if (!analysis?.analysis) return null;

    const { meta_ethics, normative_ethics, applied_ethics } = analysis.analysis;

    return (
      <div className="space-y-6">
        {/* Overall Assessment */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">🎯 Overall Ethical Assessment</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-blue-50 p-4 rounded-md">
              <div className="text-2xl font-bold text-blue-800 mb-2">
                {(analysis.analysis.ethical_confidence * 100).toFixed(1)}%
              </div>
              <div className="text-blue-600 font-medium">Confidence Level</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-md">
              <div className="text-2xl font-bold text-purple-800 mb-2">
                {(analysis.analysis.overall_consistency * 100).toFixed(1)}%
              </div>
              <div className="text-purple-600 font-medium">Framework Consistency</div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-gray-50 rounded-md">
            <div className="font-medium text-gray-800 mb-2">Synthesized Judgment:</div>
            <div className={`text-sm font-medium ${
              analysis.analysis.synthesized_judgment.includes('SOUND') ? 'text-green-700' :
              analysis.analysis.synthesized_judgment.includes('ACCEPTABLE') ? 'text-blue-700' :
              analysis.analysis.synthesized_judgment.includes('PROBLEMATIC') ? 'text-yellow-700' :
              analysis.analysis.synthesized_judgment.includes('UNACCEPTABLE') ? 'text-red-700' :
              'text-gray-700'
            }`}>
              {analysis.analysis.synthesized_judgment}
            </div>
          </div>
        </div>

        {/* Meta-Ethics Layer */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">🔍 Meta-Ethical Analysis (Logical Structure)</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {renderPhilosophicalInsight(
              "Kantian Universalizability", 
              meta_ethics.universalizability_test,
              "Can this principle be universalized without contradiction?"
            )}
            {renderPhilosophicalInsight(
              "Moore's Naturalistic Fallacy", 
              meta_ethics.naturalistic_fallacy_check,
              "Avoids conflating natural facts with moral values"
            )}
            {renderPhilosophicalInsight(
              "Semantic Coherence", 
              meta_ethics.semantic_coherence,
              "Logical consistency of ethical claims"
            )}
            {renderPhilosophicalInsight(
              "Action Guidance", 
              meta_ethics.action_guidance_strength,
              "Prescriptive force for guiding behavior"
            )}
          </div>
        </div>

        {/* Normative Ethics Layer */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">⚖️ Normative Ethics Analysis (Moral Frameworks)</h4>
          
          {/* Framework Convergence */}
          <div className="mb-6 p-4 bg-gray-50 rounded-md">
            <div className="flex justify-between items-center">
              <span className="font-medium">Framework Convergence</span>
              <span className={`px-3 py-1 rounded font-medium ${
                normative_ethics.framework_convergence > 0.8 ? 'bg-green-100 text-green-800' :
                normative_ethics.framework_convergence > 0.6 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {(normative_ethics.framework_convergence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="text-sm text-gray-600 mt-2">
              Agreement between Kantian, Utilitarian, and Aristotelian approaches
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Deontological (Kantian) */}
            <div className="border rounded-md p-4">
              <h5 className="font-semibold text-gray-800 mb-3">🏛️ Deontological (Kantian)</h5>
              <div className="space-y-2">
                {renderPhilosophicalInsight(
                  "Categorical Imperative", 
                  normative_ethics.deontological.categorical_imperative_test,
                  "Universal moral law test"
                )}
                {renderPhilosophicalInsight(
                  "Humanity Formula", 
                  normative_ethics.deontological.humanity_formula_test,
                  "Treats persons as ends"
                )}
                {renderPhilosophicalInsight(
                  "Autonomy Respect", 
                  normative_ethics.deontological.autonomy_respect,
                  "Respects rational agency"
                )}
              </div>
            </div>

            {/* Consequentialist (Utilitarian) */}
            <div className="border rounded-md p-4">
              <h5 className="font-semibold text-gray-800 mb-3">📊 Consequentialist (Utilitarian)</h5>
              <div className="space-y-2">
                {renderPhilosophicalInsight(
                  "Utility Calculation", 
                  normative_ethics.consequentialist.utility_calculation,
                  "Net welfare impact"
                )}
                {renderPhilosophicalInsight(
                  "Aggregate Welfare", 
                  normative_ethics.consequentialist.aggregate_welfare,
                  "Overall wellbeing impact"
                )}
                {renderPhilosophicalInsight(
                  "Distribution Fairness", 
                  normative_ethics.consequentialist.distribution_fairness,
                  "Fair distribution of benefits"
                )}
              </div>
            </div>

            {/* Virtue Ethics (Aristotelian) */}
            <div className="border rounded-md p-4">
              <h5 className="font-semibold text-gray-800 mb-3">🌟 Virtue Ethics (Aristotelian)</h5>
              <div className="space-y-2">
                {renderPhilosophicalInsight(
                  "Eudaimonic Contribution", 
                  normative_ethics.virtue_ethics.eudaimonic_contribution,
                  "Contribution to flourishing"
                )}
                {renderPhilosophicalInsight(
                  "Golden Mean", 
                  normative_ethics.virtue_ethics.golden_mean_analysis,
                  "Balance between extremes"
                )}
                {renderPhilosophicalInsight(
                  "Practical Wisdom", 
                  normative_ethics.virtue_ethics.practical_wisdom,
                  "Phronesis application"
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Applied Ethics Layer */}
        {applied_ethics.applicable_domains.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h4 className="font-bold text-lg mb-4">🎯 Applied Ethics Analysis (Domain-Specific)</h4>
            
            <div className="mb-4">
              <span className="font-medium text-gray-700">Applicable Domains: </span>
              <div className="inline-flex flex-wrap gap-2 mt-2">
                {applied_ethics.applicable_domains.map((domain, idx) => (
                  <span key={idx} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                    {domain.replace('_', ' ').toUpperCase()}
                  </span>
                ))}
              </div>
            </div>

            {applied_ethics.digital_ethics && (
              <div className="mb-4 p-4 border rounded-md">
                <h5 className="font-semibold mb-3">🔒 Digital Ethics Assessment</h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {renderPhilosophicalInsight("Privacy", applied_ethics.digital_ethics.privacy_assessment, "Data protection level")}
                  {renderPhilosophicalInsight("Autonomy", applied_ethics.digital_ethics.digital_autonomy, "User control level")}
                  {renderPhilosophicalInsight("Transparency", applied_ethics.digital_ethics.algorithmic_transparency, "Algorithm clarity")}
                  {renderPhilosophicalInsight("Power Distribution", applied_ethics.digital_ethics.platform_power_analysis, "Fair power balance")}
                </div>
              </div>
            )}

            {applied_ethics.ai_ethics && (
              <div className="mb-4 p-4 border rounded-md">
                <h5 className="font-semibold mb-3">🤖 AI Ethics Assessment</h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {renderPhilosophicalInsight("Fairness", applied_ethics.ai_ethics.fairness_assessment, "Algorithmic fairness")}
                  {renderPhilosophicalInsight("Safety", applied_ethics.ai_ethics.safety_assurance, "AI safety measures")}
                  {renderPhilosophicalInsight("Accountability", applied_ethics.ai_ethics.accountability_measures, "Responsibility framework")}
                  {renderPhilosophicalInsight("Human Oversight", applied_ethics.ai_ethics.human_oversight, "Human control level")}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Primary Concerns */}
        {analysis.analysis.primary_concerns.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h4 className="font-bold text-lg mb-4">⚠️ Primary Ethical Concerns</h4>
            <ul className="space-y-2">
              {analysis.analysis.primary_concerns.slice(0, 8).map((concern, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="text-yellow-500 mr-2">⚠️</span>
                  <span className="text-gray-700">{concern}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Actionable Recommendations */}
        {analysis.analysis.actionable_recommendations.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h4 className="font-bold text-lg mb-4">✅ Actionable Recommendations</h4>
            <ul className="space-y-2">
              {analysis.analysis.actionable_recommendations.slice(0, 10).map((recommendation, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  <span className="text-gray-700">{recommendation}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  const renderMLGuidanceResults = (analysis) => {
    if (!analysis?.ml_ethical_guidance) return null;

    const { ml_ethical_guidance, actionable_recommendations, philosophical_assessment } = analysis;

    return (
      <div className="space-y-6">
        {/* ML Training Overview */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">🤖 ML Training Ethics Overview</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-red-50 rounded-md">
              <div className="text-2xl font-bold text-red-700">
                {(ml_ethical_guidance.training_data_ethics.bias_risk_assessment * 100).toFixed(0)}%
              </div>
              <div className="text-sm text-red-600">Bias Risk</div>
            </div>
            <div className="text-center p-3 bg-blue-50 rounded-md">
              <div className="text-2xl font-bold text-blue-700">
                {(ml_ethical_guidance.model_development_ethics.transparency_requirements * 100).toFixed(0)}%
              </div>
              <div className="text-sm text-blue-600">Transparency</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-md">
              <div className="text-2xl font-bold text-green-700">
                {(ml_ethical_guidance.training_data_ethics.privacy_implications * 100).toFixed(0)}%
              </div>
              <div className="text-sm text-green-600">Privacy Protection</div>
            </div>
            <div className="text-center p-3 bg-purple-50 rounded-md">
              <div className="text-2xl font-bold text-purple-700">
                {(ml_ethical_guidance.model_development_ethics.safety_considerations * 100).toFixed(0)}%
              </div>
              <div className="text-sm text-purple-600">Safety Level</div>
            </div>
          </div>
        </div>

        {/* Training Data Ethics */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">📊 Training Data Ethics</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {renderPhilosophicalInsight(
              "Bias Risk Assessment", 
              ml_ethical_guidance.training_data_ethics.bias_risk_assessment,
              "Risk of introducing algorithmic bias"
            )}
            {renderPhilosophicalInsight(
              "Representation Fairness", 
              ml_ethical_guidance.training_data_ethics.representation_fairness,
              "Fair representation across groups"
            )}
            {renderPhilosophicalInsight(
              "Consent Considerations", 
              ml_ethical_guidance.training_data_ethics.consent_considerations,
              "Data usage consent compliance"
            )}
            {renderPhilosophicalInsight(
              "Privacy Implications", 
              ml_ethical_guidance.training_data_ethics.privacy_implications,
              "Privacy protection measures"
            )}
          </div>
        </div>

        {/* Model Development Ethics */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">⚙️ Model Development Ethics</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {renderPhilosophicalInsight(
              "Transparency Requirements", 
              ml_ethical_guidance.model_development_ethics.transparency_requirements,
              "Model interpretability needs"
            )}
            {renderPhilosophicalInsight(
              "Accountability Measures", 
              ml_ethical_guidance.model_development_ethics.accountability_measures,
              "Responsibility framework strength"
            )}
            {renderPhilosophicalInsight(
              "Safety Considerations", 
              ml_ethical_guidance.model_development_ethics.safety_considerations,
              "AI safety implementation"
            )}
            {renderPhilosophicalInsight(
              "Value Alignment", 
              ml_ethical_guidance.model_development_ethics.value_alignment,
              "Alignment with human values"
            )}
          </div>
        </div>

        {/* Philosophical Foundations */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">🏛️ Philosophical Foundations</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {renderPhilosophicalInsight(
              "Kantian Universalizability", 
              ml_ethical_guidance.philosophical_foundations.kantian_universalizability,
              "Universal moral law compliance"
            )}
            {renderPhilosophicalInsight(
              "Utilitarian Welfare", 
              ml_ethical_guidance.philosophical_foundations.utilitarian_welfare_impact,
              "Overall welfare contribution"
            )}
            {renderPhilosophicalInsight(
              "Character Development", 
              ml_ethical_guidance.philosophical_foundations.virtue_ethics_character_impact,
              "Impact on human character"
            )}
            {renderPhilosophicalInsight(
              "Overall Consistency", 
              ml_ethical_guidance.philosophical_foundations.overall_ethical_consistency,
              "Cross-framework consistency"
            )}
          </div>
        </div>

        {/* Practical ML Recommendations */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">🛠️ Practical ML Recommendations</h4>
          <div className="space-y-3">
            {actionable_recommendations.slice(0, 12).map((rec, idx) => (
              <div key={idx} className="flex items-start p-3 bg-blue-50 rounded-md">
                <span className="text-blue-500 mr-3 text-lg">→</span>
                <span className="text-gray-800">{rec}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Philosophical Assessment Summary */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="font-bold text-lg mb-4">🎭 Philosophical Assessment</h4>
          <div className="bg-gray-50 p-4 rounded-md mb-4">
            <div className="font-medium text-gray-800 mb-2">Overall Judgment:</div>
            <div className={`text-sm font-medium ${
              philosophical_assessment.ethical_judgment.includes('SOUND') ? 'text-green-700' :
              philosophical_assessment.ethical_judgment.includes('ACCEPTABLE') ? 'text-blue-700' :
              philosophical_assessment.ethical_judgment.includes('PROBLEMATIC') ? 'text-yellow-700' :
              philosophical_assessment.ethical_judgment.includes('UNACCEPTABLE') ? 'text-red-700' :
              'text-gray-700'
            }`}>
              {philosophical_assessment.ethical_judgment}
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="text-center p-3 bg-blue-50 rounded-md">
              <div className="text-xl font-bold text-blue-700">
                {(philosophical_assessment.confidence_level * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-blue-600">Confidence</div>
            </div>
            <div className="text-center p-3 bg-purple-50 rounded-md">
              <div className="text-xl font-bold text-purple-700">
                {(philosophical_assessment.complexity_score * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-purple-600">Complexity</div>
            </div>
          </div>

          {philosophical_assessment.primary_concerns.length > 0 && (
            <div>
              <div className="font-medium text-gray-800 mb-2">Primary Concerns:</div>
              <ul className="text-sm space-y-1">
                {philosophical_assessment.primary_concerns.slice(0, 5).map((concern, idx) => (
                  <li key={idx} className="flex items-start">
                    <span className="text-red-500 mr-2">•</span>
                    <span className="text-gray-700">{concern}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-500 to-blue-600 text-white p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-2">🧠 ML Training Ethics Assistant</h2>
        <p className="text-purple-100">
          Advanced philosophical analysis for ethical AI development • Powered by 2400+ years of ethical wisdom
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">📝 Content Analysis Input</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Training Content or Ethical Scenario
            </label>
            <textarea
              value={analysisInput}
              onChange={(e) => setAnalysisInput(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              rows={4}
              placeholder="Enter your ML training data, model description, ethical scenario, or policy statement for comprehensive philosophical analysis..."
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Analysis Depth
              </label>
              <select
                value={analysisDepth}
                onChange={(e) => setAnalysisDepth(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="surface">Surface (Quick Overview)</option>
                <option value="standard">Standard (Balanced Analysis)</option>
                <option value="comprehensive">Comprehensive (Deep Analysis)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                ML Guidance Type
              </label>
              <select
                value={guidanceType}
                onChange={(e) => setGuidanceType(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="data">Training Data Focus</option>
                <option value="model">Model Development Focus</option>
                <option value="comprehensive">Comprehensive ML Guidance</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Tabs */}
      <div className="bg-white rounded-lg shadow-sm border">
        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Ethics Analysis Tabs">
            <button
              onClick={() => setActiveAnalysisTab('comprehensive')}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeAnalysisTab === 'comprehensive'
                  ? 'border-purple-500 text-purple-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              🎯 Comprehensive Analysis
            </button>
            
            <button
              onClick={() => setActiveAnalysisTab('meta')}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeAnalysisTab === 'meta'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              🔍 Meta-Ethics
            </button>
            
            <button
              onClick={() => setActiveAnalysisTab('normative')}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeAnalysisTab === 'normative'
                  ? 'border-green-500 text-green-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              ⚖️ Normative Ethics
            </button>
            
            <button
              onClick={() => setActiveAnalysisTab('applied')}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeAnalysisTab === 'applied'
                  ? 'border-yellow-500 text-yellow-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              🎯 Applied Ethics
            </button>
            
            <button
              onClick={() => setActiveAnalysisTab('ml-guidance')}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeAnalysisTab === 'ml-guidance'
                  ? 'border-red-500 text-red-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              🤖 ML Guidance
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {/* Comprehensive Analysis Tab */}
          {activeAnalysisTab === 'comprehensive' && (
            <div className="space-y-4">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  🎯 Comprehensive Three-Layer Ethics Analysis
                </h3>
                <p className="text-gray-600 mb-4">
                  Complete philosophical analysis integrating Meta-Ethics, Normative Ethics, and Applied Ethics
                </p>
                <button
                  onClick={handleComprehensiveAnalysis}
                  disabled={analysisLoading || !analysisInput.trim()}
                  className="px-6 py-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {analysisLoading ? '🔄 Analyzing...' : '🧠 Analyze with Full Philosophical Framework'}
                </button>
              </div>
              
              {analysisResult?.analysis && renderComprehensiveResults(analysisResult)}
            </div>
          )}

          {/* Meta-Ethics Tab */}
          {activeAnalysisTab === 'meta' && (
            <div className="space-y-4">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  🔍 Meta-Ethical Analysis
                </h3>
                <p className="text-gray-600 mb-4">
                  Kantian universalizability, Moore's naturalistic fallacy, Hume's fact-value distinction
                </p>
                <button
                  onClick={handleMetaAnalysis}
                  disabled={analysisLoading || !analysisInput.trim()}
                  className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {analysisLoading ? '🔄 Analyzing...' : '🔍 Analyze Logical Structure'}
                </button>
              </div>
            </div>
          )}

          {/* Normative Ethics Tab */}
          {activeAnalysisTab === 'normative' && (
            <div className="space-y-4">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  ⚖️ Normative Ethics Analysis
                </h3>
                <p className="text-gray-600 mb-4">
                  Deontological (Kantian), Consequentialist (Utilitarian), Virtue Ethics (Aristotelian)
                </p>
                <button
                  onClick={handleNormativeAnalysis}
                  disabled={analysisLoading || !analysisInput.trim()}
                  className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {analysisLoading ? '🔄 Analyzing...' : '⚖️ Analyze Moral Frameworks'}
                </button>
              </div>
            </div>
          )}

          {/* Applied Ethics Tab */}
          {activeAnalysisTab === 'applied' && (
            <div className="space-y-4">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  🎯 Applied Ethics Analysis
                </h3>
                <p className="text-gray-600 mb-4">
                  Domain-specific analysis for Digital Ethics, AI Ethics, and professional contexts
                </p>
                <button
                  onClick={handleAppliedAnalysis}
                  disabled={analysisLoading || !analysisInput.trim()}
                  className="px-6 py-3 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {analysisLoading ? '🔄 Analyzing...' : '🎯 Analyze Practical Applications'}
                </button>
              </div>
            </div>
          )}

          {/* ML Guidance Tab */}
          {activeAnalysisTab === 'ml-guidance' && (
            <div className="space-y-4">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  🤖 ML Training Ethical Guidance
                </h3>
                <p className="text-gray-600 mb-4">
                  Specialized guidance for machine learning development with philosophical foundations
                </p>
                <button
                  onClick={handleMLGuidance}
                  disabled={analysisLoading || !analysisInput.trim()}
                  className="px-6 py-3 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {analysisLoading ? '🔄 Analyzing...' : '🤖 Generate ML Ethics Guidance'}
                </button>
              </div>

              {analysisResult?.ml_ethical_guidance && renderMLGuidanceResults(analysisResult)}
            </div>
          )}

          {/* Loading State */}
          {analysisLoading && (
            <div className="text-center py-8">
              <div className="inline-flex items-center">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600 mr-3"></div>
                <span className="text-gray-600">Performing philosophical analysis...</span>
              </div>
            </div>
          )}

          {/* Error State */}
          {analysisResult?.status === 'error' && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
              <div className="flex items-start">
                <span className="text-red-500 mr-2">⚠️</span>
                <div>
                  <div className="font-medium text-red-800">Analysis Error</div>
                  <div className="text-red-600 text-sm">{analysisResult.error}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MLTrainingAssistant;