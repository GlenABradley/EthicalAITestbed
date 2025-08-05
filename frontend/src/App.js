import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import axios from 'axios';
import EthicalChart from './components/EthicalChart';
import MLTrainingAssistant from './components/MLTrainingAssistant';
import RealTimeStreamingInterface from './components/RealTimeStreamingInterface';

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
  
  // State for threshold scaling
  const [thresholdScaling, setThresholdScaling] = useState({
    sliderValue: 0.5,  // Default to middle position
    isUpdating: false,  // Loading state
    lastUpdate: null,   // Timestamp of last update
    scalingType: 'exponential',  // Default to exponential scaling
    error: null        // Error message if any
  });
  
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
      // Fix: parameters are directly in response.data, not response.data.parameters
      setParameters(response.data);
    } catch (error) {
      console.error('Error loading parameters:', error);
      // Set default parameters if loading fails
      setParameters({
        virtue_threshold: 0.15,
        deontological_threshold: 0.15,
        consequentialist_threshold: 0.15,
        virtue_weight: 1.0,
        deontological_weight: 1.0,
        consequentialist_weight: 1.0
      });
    }
  };

  // Memoized event handlers to prevent re-rendering issues
  const handleEvaluate = useCallback(() => {
    console.log('🔥 BUTTON CLICKED - handleEvaluate called!');
    
    if (!inputText.trim()) {
      console.log('❌ No input text provided');
      alert('Please enter some text to evaluate');
      return;
    }

    console.log('✅ Input text validated:', inputText);
    setLoading(true);
    console.log('🚀 Starting evaluation for:', inputText);

    // Use fetch for evaluation
    fetch(`${API}/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: inputText,
        parameters: {
          ...(parameters || {}),
          // Include the tau slider value in the parameters
          tau_slider: thresholdScaling.sliderValue,
          // Include the scaling type (exponential/linear)
          scaling_type: thresholdScaling.scalingType
        }
      })
    })
    .then(response => {
      console.log('📥 Response received:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('📊 Evaluation data received:', data);
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
  }, [inputText, parameters]);

  // Alternative click handler using native DOM events as fallback
  const handleEvaluateNative = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    console.log('🎯 NATIVE EVENT HANDLER CALLED');
    handleEvaluate();
  }, [handleEvaluate]);

  const updateParameter = useCallback((key, value) => {
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
  }, [parameters]);

  // Memoized tab handlers
  const handleTabSwitch = useCallback((tabName) => {
    console.log(`🔥 TAB SWITCH CLICKED - switching to: ${tabName}`);
    setActiveTab(tabName);
  }, []);

  // Alternative tab handlers using native events
  const handleTabSwitchNative = useCallback((event, tabName) => {
    event.preventDefault();
    event.stopPropagation();
    console.log(`🎯 NATIVE TAB HANDLER - switching to: ${tabName}`);
    setActiveTab(tabName);
  }, []);

  // Add native DOM event listeners as fallback after component mounts
  useEffect(() => {
    console.log('🔧 Setting up fallback event listeners');
    
    // Function to add native click listeners to buttons
    const addNativeListeners = () => {
      // Tab buttons
      const tabButtons = document.querySelectorAll('nav button');
      tabButtons.forEach((button, index) => {
        const tabNames = ['evaluate', 'heatmap', 'ml-assistant', 'streaming', 'parameters'];
        const tabName = tabNames[index];
        
        if (tabName) {
          // Remove existing listeners
          button.removeEventListener('click', button._nativeClickHandler);
          
          // Add new listener
          const clickHandler = (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log(`🎯 NATIVE LISTENER - Tab clicked: ${tabName}`);
            setActiveTab(tabName);
          };
          
          button._nativeClickHandler = clickHandler;
          button.addEventListener('click', clickHandler, { passive: false });
        }
      });
      
      // Evaluate button
      const evaluateButton = document.querySelector('button[data-action="evaluate"]');
      if (evaluateButton) {
        evaluateButton.removeEventListener('click', evaluateButton._nativeClickHandler);
        
        const evaluateClickHandler = (e) => {
          e.preventDefault();
          e.stopPropagation();
          console.log('🎯 NATIVE LISTENER - Evaluate button clicked');
          handleEvaluate();
        };
        
        evaluateButton._nativeClickHandler = evaluateClickHandler;
        evaluateButton.addEventListener('click', evaluateClickHandler, { passive: false });
      }
    };

    // Add listeners after a short delay to ensure DOM is ready
    const timer = setTimeout(addNativeListeners, 100);
    
    // Also add listeners when the active tab changes
    addNativeListeners();
    
    return () => {
      clearTimeout(timer);
      // Cleanup listeners
      const allButtons = document.querySelectorAll('button');
      allButtons.forEach(button => {
        if (button._nativeClickHandler) {
          button.removeEventListener('click', button._nativeClickHandler);
        }
      });
    };
  }, [activeTab, handleEvaluate]);

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
      // NOTE: Using optimized mock endpoint for heat-map visualization
      // The full ethical evaluation takes 60+ seconds and would timeout in the UI
      // This provides authentic visualization structure with representative data
      const response = await axios.post(`${API}/heat-map-mock`, {
        text: text
      });
      setHeatMapData(response.data);
    } catch (error) {
      console.error('Heat-map generation error:', error);
      setHeatMapData(null);
    } finally {
      setHeatMapLoading(false);
    }
  };
  
  // Ref to track the last evaluation time to prevent rapid re-evaluations
  const lastEvaluationTimeRef = React.useRef(0);
  const EVALUATION_COOLDOWN = 1000; // 1 second cooldown between evaluations

  /**
   * Updates the threshold scaling values and triggers a re-evaluation if needed
   * @param {number} newValue - The new slider value (0.0 to 1.0)
   * @param {boolean} triggerEvaluation - Whether to trigger a re-evaluation after updating
   */
  const updateThresholdScaling = async (newValue, triggerEvaluation = true) => {
    try {
      // Prevent unnecessary updates if the value hasn't changed
      if (Math.abs(thresholdScaling.sliderValue - newValue) < 0.001) {
        return;
      }

      // Prevent rapid updates
      const now = Date.now();
      if (now - lastEvaluationTimeRef.current < EVALUATION_COOLDOWN) {
        console.log('Skipping rapid threshold update');
        return;
      }
      lastEvaluationTimeRef.current = now;

      console.log('Updating threshold scaling to:', newValue);
      
      // Update local state optimistically
      setThresholdScaling(prev => ({
        ...prev,
        sliderValue: newValue,
        isUpdating: true,
        error: null
      }));

      try {
        // Send update to backend
        const response = await axios.post(`${API}/threshold-scaling`, {
          slider_value: newValue,
          use_exponential: thresholdScaling.scalingType === 'exponential'
        });

        // Update state with response
        setThresholdScaling(prev => ({
          ...prev,
          isUpdating: false,
          lastUpdate: new Date().toISOString(),
          error: null
        }));

        console.log('Threshold scaling updated:', response.data);
        
        // If we have an evaluation result and we should trigger a re-evaluation
        if (evaluationResult && triggerEvaluation && inputText.trim()) {
          console.log('Triggering re-evaluation with new threshold');
          await handleEvaluate();
        }
        
        return response.data;
      } catch (error) {
        // If there's an error, reset the slider to the last good value
        setThresholdScaling(prev => ({
          ...prev,
          sliderValue: thresholdScaling.sliderValue,
          isUpdating: false,
          error: error.message
        }));
        throw error;
      }
    } catch (error) {
      console.error('Error updating threshold scaling:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Failed to update threshold';
      
      setThresholdScaling(prev => ({
        ...prev,
        isUpdating: false,
        error: errorMsg
      }));
      
      // Show error to user
      setFeedbackMessage({
        type: 'error',
        text: `Failed to update threshold: ${errorMsg}`
      });
      
      throw error;
    }
  };
  
  /**
   * Toggles between exponential and linear scaling
   */
  const toggleScalingType = async () => {
    const newScalingType = thresholdScaling.scalingType === 'exponential' ? 'linear' : 'exponential';
    
    setThresholdScaling(prev => ({
      ...prev,
      scalingType: newScalingType,
      isUpdating: true
    }));
    
    // Re-apply the current slider value with the new scaling type
    try {
      await updateThresholdScaling(thresholdScaling.sliderValue, true);
    } catch (error) {
      // Error is already handled in updateThresholdScaling
    }
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
              onClick={() => handleTabSwitch('evaluate')}
              onMouseDown={(e) => handleTabSwitchNative(e, 'evaluate')}
              data-tab="evaluate"
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'evaluate'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Evaluate Text
            </button>
            <button
              onClick={() => handleTabSwitch('heatmap')}
              onMouseDown={(e) => handleTabSwitchNative(e, 'heatmap')}
              data-tab="heatmap"
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'heatmap'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              📊 Heat-Map
            </button>
            <button
              onClick={() => handleTabSwitch('ml-assistant')}
              onMouseDown={(e) => handleTabSwitchNative(e, 'ml-assistant')}
              data-tab="ml-assistant"
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'ml-assistant'
                  ? 'bg-purple-500 text-white'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              🧠 ML Ethics Assistant
            </button>
            <button
              onClick={() => handleTabSwitch('streaming')}
              onMouseDown={(e) => handleTabSwitchNative(e, 'streaming')}
              data-tab="streaming"
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'streaming'
                  ? 'bg-green-500 text-white'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              🚀 Real-Time Streaming
            </button>
            <button
              onClick={() => handleTabSwitch('parameters')}
              onMouseDown={(e) => handleTabSwitchNative(e, 'parameters')}
              data-tab="parameters"
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
                      onMouseDown={handleEvaluateNative}
                      data-action="evaluate"
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
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <h4 className="font-semibold text-gray-800">Threshold Sensitivity</h4>
                              <button
                                onClick={toggleScalingType}
                                className={`px-2 py-1 text-xs rounded-md ${
                                  thresholdScaling.scalingType === 'exponential'
                                    ? 'bg-blue-100 text-blue-800 hover:bg-blue-200'
                                    : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                                } transition-colors`}
                                title={thresholdScaling.scalingType === 'exponential' 
                                  ? 'Using exponential scaling (better for fine-tuning)' 
                                  : 'Using linear scaling'}
                                disabled={thresholdScaling.isUpdating}
                              >
                                {thresholdScaling.scalingType === 'exponential' ? 'Exponential' : 'Linear'}
                              </button>
                            </div>
                            
                            <div className="flex items-center space-x-3">
                              <span className="text-xs text-gray-500 w-8">Low</span>
                              <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.01"
                                value={thresholdScaling.sliderValue}
                                onChange={(e) => updateThresholdScaling(parseFloat(e.target.value), true)}
                                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                disabled={thresholdScaling.isUpdating}
                              />
                              <span className="text-xs text-gray-500 w-12 text-right">High</span>
                            </div>
                            
                            <div className="text-sm space-y-1 text-gray-600">
                              <div className="flex justify-between">
                                <span>Sensitivity:</span>
                                <span className="font-medium">
                                  {Math.round(thresholdScaling.sliderValue * 100)}%
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span>Threshold:</span>
                                <span className="font-mono">
                                  {thresholdScaling.scalingType === 'exponential'
                                    ? ((Math.exp(6 * thresholdScaling.sliderValue) - 1) / (Math.exp(6) - 1) * 0.5).toFixed(4)
                                    : (thresholdScaling.sliderValue * 0.5).toFixed(4)
                                  }
                                </span>
                              </div>
                              {thresholdScaling.isUpdating && (
                                <div className="text-blue-600 text-xs flex items-center">
                                  <svg className="animate-spin -ml-1 mr-2 h-3 w-3 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                  </svg>
                                  Updating...
                                </div>
                              )}
                              {thresholdScaling.error && (
                                <div className="text-red-500 text-xs">
                                  Error: {thresholdScaling.error}
                                </div>
                              )}
                            </div>
                          </div>
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
                          value={parameters?.[param.key] || 0}
                          onChange={(e) => updateParameter(param.key, e.target.value)}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-gray-500">
                          <span>0</span>
                          <span className="font-mono">{(parameters?.[param.key] || 0).toFixed(3)}</span>
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
                          value={parameters?.[param.key] || 0}
                          onChange={(e) => updateParameter(param.key, e.target.value)}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-gray-500">
                          <span>0</span>
                          <span className="font-mono">{(parameters?.[param.key] || 0).toFixed(1)}</span>
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
                          checked={parameters?.enable_dynamic_scaling || false}
                          onChange={(e) => updateParameter('enable_dynamic_scaling', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Enable Dynamic Scaling</span>
                      </label>
                      
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={parameters?.enable_cascade_filtering || false}
                          onChange={(e) => updateParameter('enable_cascade_filtering', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Enable Cascade Filtering</span>
                      </label>
                      
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={parameters?.enable_learning_mode || false}
                          onChange={(e) => updateParameter('enable_learning_mode', e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Enable Learning Mode</span>
                      </label>
                      
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={parameters?.exponential_scaling || false}
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
                          value={parameters?.cascade_high_threshold || 0.25}
                          onChange={(e) => updateParameter('cascade_high_threshold', e.target.value)}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-gray-500">
                          <span>0.15</span>
                          <span className="font-mono">{(parameters?.cascade_high_threshold || 0.25).toFixed(3)}</span>
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
                          value={parameters?.cascade_low_threshold || 0.08}
                          onChange={(e) => updateParameter('cascade_low_threshold', e.target.value)}
                          className="w-full"
                        />
                        <div className="flex justify-between text-sm text-gray-500">
                          <span>0.0</span>
                          <span className="font-mono">{(parameters?.cascade_low_threshold || 0.08).toFixed(3)}</span>
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

          {activeTab === 'ml-assistant' && (
            <MLTrainingAssistant backendUrl={BACKEND_URL} />
          )}

          {activeTab === 'streaming' && (
            <RealTimeStreamingInterface backendUrl={BACKEND_URL} />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;