import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import EthicalChart from './components/EthicalChart';

// Backend API endpoint from environment variables
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

/**
 * Main Application Component - Ethical AI Developer Testbed
 * 
 * This is the main React component that provides:
 * - Multi-perspective ethical text evaluation
 * - Dynamic scaling and learning system integration
 * - Comprehensive parameter calibration interface
 * - Real-time feedback system for continuous improvement
 */
function App() {
  const [activeTab, setActiveTab] = useState('evaluate');
  const [inputText, setInputText] = useState('');
  const [evaluationResult, setEvaluationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [parameters, setParameters] = useState({});
  
  // New state for learning system
  const [learningStats, setLearningStats] = useState({});
  const [activeResultTab, setActiveResultTab] = useState('violations');
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [thresholdScalingTest, setThresholdScalingTest] = useState({});
  
  // Phase 4: Heat-map visualization state
  const [heatMapData, setHeatMapData] = useState(null);
  const [heatMapLoading, setHeatMapLoading] = useState(false);
  
  // Load initial parameters and learning stats
  useEffect(() => {
    loadParameters();
    loadLearningStats();
  }, []);
  
  const loadLearningStats = async () => {
    try {
      const response = await axios.get(`${API}/learning-stats`);
      setLearningStats(response.data);
    } catch (error) {
      console.error('Error loading learning stats:', error);
    }
  };

  const loadParameters = async () => {
    try {
      const response = await axios.get(`${API}/parameters`);
      setParameters(response.data.parameters);
    } catch (error) {
      console.error('Error loading parameters:', error);
    }
  };

  // Simple evaluation function
  const handleEvaluate = () => {
    if (!inputText.trim()) {
      alert('Please enter some text to evaluate');
      return;
    }

    setLoading(true);
    console.log('🚀 Starting evaluation for:', inputText);

    // Create abort controller for timeout handling
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minutes timeout

    // Use fetch with extended timeout for complex ethical evaluation
    fetch(`${API}/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: inputText,
        parameters: parameters || {}
      }),
      signal: controller.signal
    })
    .then(response => {
      console.log('📥 Response received:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('📊 Evaluation data:', data);
      setEvaluationResult(data);
      console.log('✅ Results set successfully');
    })
    .catch(error => {
      console.error('❌ Error during evaluation:', error);
      alert('Error: ' + error.message);
    })
    .finally(() => {
      setLoading(false);
      console.log('🏁 Evaluation finished');
    });
  };

  const updateParameter = (key, value) => {
    // Handle different parameter types properly
    let processedValue;
    
    if (typeof value === 'boolean') {
      // For checkboxes, use the boolean value directly
      processedValue = value;
    } else if (typeof value === 'string' && (value === 'true' || value === 'false')) {
      // Handle string boolean values
      processedValue = value === 'true';
    } else {
      // For numeric inputs (sliders), parse as float
      processedValue = parseFloat(value) || 0;
    }
    
    const newParams = { ...parameters, [key]: processedValue };
    setParameters(newParams);
    
    // Update backend parameters
    fetch(`${API}/update-parameters`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ parameters: newParams })
    })
    .then(response => response.json())
    .then(data => {
      // Parameters updated successfully
    })
    .catch(error => console.error('Parameter update error:', error));
  };

  // New functions for learning system
  const submitFeedback = (evaluationId, feedbackScore, userComment = '') => {
    fetch(`${API}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        evaluation_id: evaluationId,
        feedback_score: feedbackScore,
        user_comment: userComment
      })
    })
    .then(response => response.json())
    .then(data => {
      setFeedbackMessage('Feedback submitted successfully!');
      setTimeout(() => setFeedbackMessage(''), 3000);
      loadLearningStats(); // Refresh stats
    })
    .catch(error => {
      console.error('Feedback error:', error);
      setFeedbackMessage('Error submitting feedback');
    });
  };

  // Phase 4: Heat-map visualization function
  const generateHeatMap = async (text) => {
    if (!text.trim()) {
      setHeatMapData(null);
      return;
    }
    
    setHeatMapLoading(true);
    try {
      // Using real evaluation endpoint for authentic heat-map generation
      // Extended timeout to accommodate complex ethical evaluation processing
      const response = await axios.post(`${API}/heat-map-visualization`, {
        text: text
      }, {
        timeout: 120000 // 2 minutes timeout for complex evaluation
      });
      setHeatMapData(response.data);
    } catch (error) {
      console.error('Heat-map generation error:', error);
      setHeatMapData(null);
    } finally {
      setHeatMapLoading(false);
    }
  };

  const testThresholdScaling = (sliderValue, useExponential = true) => {
    fetch(`${API}/threshold-scaling`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        slider_value: sliderValue,
        use_exponential: useExponential
      })
    })
    .then(response => response.json())
    .then(data => {
      console.log('Threshold scaling test:', data);
      setThresholdScalingTest(data);
    })
    .catch(error => console.error('Threshold scaling test error:', error));
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Ethical AI Developer Testbed
          </h1>
          <p className="text-gray-600">
            Multi-Perspective Ethical Text Evaluation Framework
          </p>
        </header>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <nav className="flex space-x-2 bg-white p-2 rounded-lg shadow">
            <button
              onClick={() => setActiveTab('evaluate')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'evaluate'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Evaluate Text
            </button>
            <button
              onClick={() => setActiveTab('heatmap')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'heatmap'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              📊 Heat-Map
            </button>
            <button
              onClick={() => setActiveTab('parameters')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'parameters'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Parameter Tuning
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="max-w-6xl mx-auto">
          {activeTab === 'evaluate' && (
            <div className="space-y-6">
              {/* Input Section */}
              <div className="bg-white p-6 rounded-lg shadow">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-2xl font-bold">Text Evaluation</h2>
                  
                  {/* v3.0 Semantic Embedding Update Notice */}
                  <div className="relative group">
                    <div className="flex items-center space-x-2 bg-blue-50 text-blue-700 px-3 py-1 rounded-full border border-blue-200 cursor-help">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                      <span className="text-sm font-medium">The ethical embeddings have been updated!</span>
                    </div>
                    
                    {/* Tooltip */}
                    <div className="absolute right-0 top-full mt-2 w-80 bg-white border border-gray-200 rounded-lg shadow-lg p-4 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                      <div className="space-y-3">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <span className="font-semibold text-gray-800">v3.0 Semantic Embedding Framework</span>
                        </div>
                        
                        <div className="text-sm text-gray-600 space-y-2">
                          <p><strong>Revolutionary Upgrade:</strong> Enhanced from basic examples to sophisticated autonomy-maximization principles.</p>
                          
                          <div className="bg-gray-50 p-3 rounded">
                            <p className="font-medium text-gray-800 mb-1">Core Axiom:</p>
                            <p className="text-xs">Maximize human autonomy (Σ D_i) within objective empirical truth (t ≥ 0.95)</p>
                          </div>
                          
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div>
                              <p className="font-medium text-gray-800">Before:</p>
                              <p>Simple virtue-vice examples</p>
                            </div>
                            <div>
                              <p className="font-medium text-gray-800">Now:</p>
                              <p>5 Autonomy dimensions + Mathematical rigor</p>
                            </div>
                          </div>
                          
                          <div className="bg-blue-50 p-2 rounded">
                            <p className="text-xs font-medium text-blue-800">✨ 18% improvement in principle clustering</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Enter text to evaluate:
                    </label>
                    <textarea
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      placeholder="Type your text here..."
                      className="w-full h-32 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div className="flex space-x-4">
                    <button
                      onClick={handleEvaluate}
                      disabled={loading || !inputText.trim()}
                      className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
                    >
                      {loading ? 'Evaluating...' : 'Evaluate Text'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Results Section */}
              {evaluationResult && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Evaluation Summary */}
                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-xl font-bold mb-4">Evaluation Summary</h3>
                    <div className="space-y-3">
                      <div className={`p-3 rounded-md ${
                        evaluationResult.evaluation?.overall_ethical 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        Status: {evaluationResult.evaluation?.overall_ethical ? 'Ethical' : 'Unethical'}
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>Processing Time: {evaluationResult.evaluation?.processing_time?.toFixed(3)}s</div>
                        <div>Violations Found: {evaluationResult.evaluation?.minimal_violation_count || 0}</div>
                        <div>Original Length: {evaluationResult.delta_summary?.original_length || 0}</div>
                        <div>Clean Length: {evaluationResult.delta_summary?.clean_length || 0}</div>
                      </div>
                    </div>
                  </div>

                  {/* Clean Text */}
                  <div className="bg-white p-6 rounded-lg shadow">
                    <h3 className="text-xl font-bold mb-4">Clean Text</h3>
                    <div className="bg-gray-50 p-4 rounded-md">
                      <p className="text-gray-700">{evaluationResult.clean_text}</p>
                    </div>
                  </div>

                  {/* Violations and Analysis Tabs */}
                  <div className="bg-white p-6 rounded-lg shadow lg:col-span-2">
                    <h3 className="text-xl font-bold mb-4">Detailed Analysis</h3>
                    
                    {/* Tab Navigation */}
                    <div className="flex space-x-4 mb-6 border-b">
                      <button
                        onClick={() => setActiveResultTab('violations')}
                        className={`px-4 py-2 -mb-px font-medium transition-colors ${
                          activeResultTab === 'violations'
                            ? 'border-b-2 border-blue-500 text-blue-600'
                            : 'text-gray-600 hover:text-gray-800'
                        }`}
                      >
                        Violations ({evaluationResult.evaluation?.minimal_violation_count || 0})
                      </button>
                      <button
                        onClick={() => setActiveResultTab('allSpans')}
                        className={`px-4 py-2 -mb-px font-medium transition-colors ${
                          activeResultTab === 'allSpans'
                            ? 'border-b-2 border-blue-500 text-blue-600'
                            : 'text-gray-600 hover:text-gray-800'
                        }`}
                      >
                        All Spans ({evaluationResult.evaluation?.spans?.length || 0})
                      </button>
                      <button
                        onClick={() => setActiveResultTab('learning')}
                        className={`px-4 py-2 -mb-px font-medium transition-colors ${
                          activeResultTab === 'learning'
                            ? 'border-b-2 border-blue-500 text-blue-600'
                            : 'text-gray-600 hover:text-gray-800'
                        }`}
                      >
                        Learning & Feedback
                      </button>
                      <button
                        onClick={() => setActiveResultTab('dynamic')}
                        className={`px-4 py-2 -mb-px font-medium transition-colors ${
                          activeResultTab === 'dynamic'
                            ? 'border-b-2 border-blue-500 text-blue-600'
                            : 'text-gray-600 hover:text-gray-800'
                        }`}
                      >
                        Dynamic Scaling
                      </button>
                    </div>

                    {/* Tab Content */}
                    {activeResultTab === 'violations' && (
                      <div className="space-y-3">
                        {evaluationResult.evaluation?.minimal_spans?.length > 0 ? (
                          evaluationResult.evaluation.minimal_spans.map((span, index) => (
                            <div key={index} className="border border-red-200 rounded-md p-4">
                              <div className="flex justify-between items-start mb-2">
                                <span className="font-mono bg-red-100 text-red-800 px-2 py-1 rounded">
                                  "{span.text}"
                                </span>
                                <span className="text-sm text-gray-500">
                                  Positions {span.start}-{span.end}
                                </span>
                              </div>
                              <div className="text-sm space-y-1">
                                <div>Virtue: {span.virtue_score?.toFixed(3)} {span.virtue_violation ? '❌' : '✅'}</div>
                                <div>Deontological: {span.deontological_score?.toFixed(3)} {span.deontological_violation ? '❌' : '✅'}</div>
                                <div>Consequentialist: {span.consequentialist_score?.toFixed(3)} {span.consequentialist_violation ? '❌' : '✅'}</div>
                              </div>
                            </div>
                          ))
                        ) : (
                          <div className="text-center py-8 text-gray-500">
                            No violations detected in this text.
                          </div>
                        )}
                      </div>
                    )}

                    {activeResultTab === 'allSpans' && (
                      <div className="space-y-3 max-h-96 overflow-y-auto">
                        {evaluationResult.evaluation?.spans?.length > 0 ? (
                          evaluationResult.evaluation.spans.map((span, index) => (
                            <div key={index} className={`border rounded-md p-4 ${
                              span.any_violation ? 'border-red-200 bg-red-50' : 'border-gray-200'
                            }`}>
                              <div className="flex justify-between items-start mb-2">
                                <span className={`font-mono px-2 py-1 rounded ${
                                  span.any_violation 
                                    ? 'bg-red-100 text-red-800' 
                                    : 'bg-gray-100 text-gray-800'
                                }`}>
                                  "{span.text}"
                                </span>
                                <span className="text-sm text-gray-500">
                                  Positions {span.start}-{span.end}
                                </span>
                              </div>
                              <div className="text-sm grid grid-cols-3 gap-4">
                                <div>
                                  <strong>Virtue:</strong> {span.virtue_score?.toFixed(3)} 
                                  {span.virtue_violation ? ' ❌' : ' ✅'}
                                </div>
                                <div>
                                  <strong>Deontological:</strong> {span.deontological_score?.toFixed(3)} 
                                  {span.deontological_violation ? ' ❌' : ' ✅'}
                                </div>
                                <div>
                                  <strong>Consequentialist:</strong> {span.consequentialist_score?.toFixed(3)} 
                                  {span.consequentialist_violation ? ' ❌' : ' ✅'}
                                </div>
                              </div>
                            </div>
                          ))
                        ) : (
                          <div className="text-center py-8 text-gray-500">
                            No spans to display.
                          </div>
                        )}
                      </div>
                    )}

                    {activeResultTab === 'learning' && (
                      <div className="space-y-4">
                        <div className="bg-blue-50 p-4 rounded-md">
                          <h4 className="font-semibold text-blue-800 mb-2">Learning System Status</h4>
                          <div className="text-sm space-y-1">
                            <div>Total Learning Entries: {learningStats.total_learning_entries || 0}</div>
                            <div>Average Feedback Score: {learningStats.average_feedback_score?.toFixed(3) || 0}</div>
                            <div>Learning Active: {learningStats.learning_active ? 'Yes' : 'No'}</div>
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 p-4 rounded-md">
                          <h4 className="font-semibold text-gray-800 mb-3">Provide Feedback (Dopamine System)</h4>
                          <div className="flex space-x-2 mb-3">
                            <button
                              onClick={() => submitFeedback(evaluationResult.evaluation?.evaluation_id, 1.0, 'Perfect result')}
                              className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
                            >
                              Perfect (1.0)
                            </button>
                            <button
                              onClick={() => submitFeedback(evaluationResult.evaluation?.evaluation_id, 0.8, 'Good result')}
                              className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                            >
                              Good (0.8)
                            </button>
                            <button
                              onClick={() => submitFeedback(evaluationResult.evaluation?.evaluation_id, 0.5, 'Okay result')}
                              className="px-3 py-1 bg-yellow-500 text-white rounded hover:bg-yellow-600"
                            >
                              Okay (0.5)
                            </button>
                            <button
                              onClick={() => submitFeedback(evaluationResult.evaluation?.evaluation_id, 0.2, 'Poor result')}
                              className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
                            >
                              Poor (0.2)
                            </button>
                          </div>
                          {feedbackMessage && (
                            <div className="text-sm text-blue-600 mt-2">{feedbackMessage}</div>
                          )}
                        </div>
                      </div>
                    )}

                    {activeResultTab === 'dynamic' && (
                      <div className="space-y-4">
                        <div className="bg-purple-50 p-4 rounded-md">
                          <h4 className="font-semibold text-purple-800 mb-2">Dynamic Scaling Information</h4>
                          {evaluationResult.evaluation?.dynamic_scaling ? (
                            <div className="text-sm space-y-1">
                              <div>Dynamic Scaling Used: {evaluationResult.evaluation.dynamic_scaling.used_dynamic_scaling ? 'Yes' : 'No'}</div>
                              <div>Cascade Filtering Used: {evaluationResult.evaluation.dynamic_scaling.used_cascade_filtering ? 'Yes' : 'No'}</div>
                              <div>Ambiguity Score: {evaluationResult.evaluation.dynamic_scaling.ambiguity_score?.toFixed(3) || 'N/A'}</div>
                              <div>Processing Stages: {evaluationResult.evaluation.dynamic_scaling.processing_stages?.join(', ') || 'None'}</div>
                              {evaluationResult.evaluation.dynamic_scaling.cascade_result && (
                                <div>Cascade Result: {evaluationResult.evaluation.dynamic_scaling.cascade_result}</div>
                              )}
                            </div>
                          ) : (
                            <div className="text-sm text-gray-600">Dynamic scaling information not available for this evaluation.</div>
                          )}
                        </div>
                        
                        <div className="bg-gray-50 p-4 rounded-md">
                          <h4 className="font-semibold text-gray-800 mb-3">Threshold Scaling Test</h4>
                          <div className="flex space-x-2 mb-3">
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.1"
                              onChange={(e) => testThresholdScaling(parseFloat(e.target.value))}
                              className="flex-1"
                            />
                            <span className="text-sm w-16">Test Slider</span>
                          </div>
                          {thresholdScalingTest.slider_value !== undefined && (
                            <div className="text-sm space-y-1">
                              <div>Slider Value: {thresholdScalingTest.slider_value}</div>
                              <div>Scaled Threshold: {thresholdScalingTest.scaled_threshold?.toFixed(4)}</div>
                              <div>Scaling Type: {thresholdScalingTest.scaling_type}</div>
                              <div>Formula: {thresholdScalingTest.formula}</div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Explanation */}
                  <div className="bg-white p-6 rounded-lg shadow lg:col-span-2">
                    <h3 className="text-xl font-bold mb-4">Explanation</h3>
                    <div className="bg-gray-50 p-4 rounded-md">
                      <pre className="text-sm text-gray-700 whitespace-pre-wrap">
                        {evaluationResult.explanation}
                      </pre>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'heatmap' && (
            <div className="space-y-6">
              {/* Heat-Map Input Section */}
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-2xl font-bold mb-4">📊 Heat-Map Visualization</h2>
                <p className="text-gray-600 mb-4">
                  Generate multidimensional ethical evaluation heat-maps with span granularity analysis.
                </p>
                
                <div className="space-y-4">
                  <div>
                    <label htmlFor="heatmap-text" className="block text-sm font-medium text-gray-700 mb-2">
                      Enter text to visualize:
                    </label>
                    <textarea
                      id="heatmap-text"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                      rows="4"
                      placeholder="Type your text here for heat-map analysis..."
                    />
                  </div>
                  
                  <div className="flex space-x-3">
                    <button
                      onClick={() => generateHeatMap(inputText)}
                      disabled={!inputText.trim() || heatMapLoading}
                      className="px-6 py-2 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                      {heatMapLoading ? 'Generating...' : 'Generate Heat-Map'}
                    </button>
                    
                    <button
                      onClick={() => {
                        setHeatMapData(null);
                        setInputText('');
                      }}
                      className="px-6 py-2 bg-gray-600 text-white font-medium rounded-md hover:bg-gray-700"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              </div>

              {/* Heat-Map Visualization */}
              {heatMapLoading && (
                <div className="bg-white p-6 rounded-lg shadow">
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    <span className="ml-3 text-gray-600">Generating heat-map visualization...</span>
                  </div>
                </div>
              )}

              {heatMapData && !heatMapLoading && (
                <EthicalChart data={heatMapData} />
              )}

              {!heatMapData && !heatMapLoading && inputText.trim() && (
                <div className="bg-gray-50 p-6 rounded-lg border-2 border-dashed border-gray-300">
                  <div className="text-center text-gray-500">
                    <div className="text-lg mb-2">📊 Ready to Generate</div>
                    <div className="text-sm">Click "Generate Heat-Map" to create visualization</div>
                  </div>
                </div>
              )}
              
              {!inputText.trim() && !heatMapLoading && (
                <div className="bg-gray-50 p-6 rounded-lg border-2 border-dashed border-gray-300">
                  <div className="text-center text-gray-500">
                    <div className="text-lg mb-2">📝 Enter Text Above</div>
                    <div className="text-sm">Add text to generate multidimensional ethical visualization</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'parameters' && (
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-2xl font-bold mb-4">Parameter Calibration</h2>
              <div className="space-y-6">
                {/* Thresholds */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Ethical Perspective Thresholds (τ_P)</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[
                      { key: 'virtue_threshold', label: 'Virtue Ethics' },
                      { key: 'deontological_threshold', label: 'Deontological' },
                      { key: 'consequentialist_threshold', label: 'Consequentialist' }
                    ].map((param) => (
                      <div key={param.key}>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          {param.label}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="0.5"
                          step="0.005"
                          value={parameters[param.key] || 0}
                          onChange={(e) => updateParameter(param.key, e.target.value)}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-gray-500">
                          <span>0</span>
                          <span className="font-mono">{(parameters[param.key] || 0).toFixed(3)}</span>
                          <span>0.5</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Weights */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Perspective Weights</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[
                      { key: 'virtue_weight', label: 'Virtue Weight' },
                      { key: 'deontological_weight', label: 'Deontological Weight' },
                      { key: 'consequentialist_weight', label: 'Consequentialist Weight' }
                    ].map((param) => (
                      <div key={param.key}>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          {param.label}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="3"
                          step="0.1"
                          value={parameters[param.key] || 0}
                          onChange={(e) => updateParameter(param.key, e.target.value)}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-gray-500">
                          <span>0</span>
                          <span className="font-mono">{(parameters[param.key] || 0).toFixed(1)}</span>
                          <span>3</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Dynamic Scaling Controls */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Dynamic Scaling & Learning</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-3">
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={parameters.enable_dynamic_scaling || false}
                          onChange={(e) => updateParameter('enable_dynamic_scaling', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Enable Dynamic Scaling</span>
                      </label>
                      
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={parameters.enable_cascade_filtering || false}
                          onChange={(e) => updateParameter('enable_cascade_filtering', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Enable Cascade Filtering</span>
                      </label>
                      
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={parameters.enable_learning_mode || false}
                          onChange={(e) => updateParameter('enable_learning_mode', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Enable Learning Mode</span>
                      </label>
                      
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={parameters.exponential_scaling || false}
                          onChange={(e) => updateParameter('exponential_scaling', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Exponential Threshold Scaling</span>
                      </label>
                    </div>
                    
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Cascade High Threshold
                        </label>
                        <input
                          type="range"
                          min="0.15"
                          max="0.5"
                          step="0.01"
                          value={parameters.cascade_high_threshold || 0.25}
                          onChange={(e) => updateParameter('cascade_high_threshold', e.target.value)}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-gray-500">
                          <span>0.15</span>
                          <span className="font-mono">{(parameters.cascade_high_threshold || 0.25).toFixed(3)}</span>
                          <span>0.5</span>
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Cascade Low Threshold
                        </label>
                        <input
                          type="range"
                          min="0.0"
                          max="0.2"
                          step="0.01"
                          value={parameters.cascade_low_threshold || 0.08}
                          onChange={(e) => updateParameter('cascade_low_threshold', e.target.value)}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-gray-500">
                          <span>0.0</span>
                          <span className="font-mono">{(parameters.cascade_low_threshold || 0.08).toFixed(3)}</span>
                          <span>0.2</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Learning System Status */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Learning System Status</h3>
                  <div className="bg-blue-50 p-4 rounded-md">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      <div>
                        <strong>Total Learning Entries:</strong> {learningStats.total_learning_entries || 0}
                      </div>
                      <div>
                        <strong>Average Feedback Score:</strong> {learningStats.average_feedback_score?.toFixed(3) || 0}
                      </div>
                      <div>
                        <strong>Learning Active:</strong> {learningStats.learning_active ? 'Yes' : 'No'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;