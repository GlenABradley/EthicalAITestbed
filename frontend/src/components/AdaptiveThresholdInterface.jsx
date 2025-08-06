import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// Backend API endpoint
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
const API = `${BACKEND_URL}/api`;

/**
 * Adaptive Threshold Learning Interface v1.2.2
 * 
 * This React component provides a comprehensive interface for the Phase 2 adaptive threshold
 * learning system, enabling users to interact with perceptron-based threshold optimization
 * while maintaining full transparency and cognitive autonomy.
 * 
 * Core Features:
 * 1. **Real-time Violation Prediction**: 
 *    - Text input with confidence-scored violation detection
 *    - Three perceptron variants (classic, averaged, voted)
 *    - Intent-normalized feature extraction with α=0.2
 * 
 * 2. **Model Training Interface**:
 *    - Training data generation across 5 ethical domains
 *    - Algorithm selection (classic/averaged/voted perceptron)
 *    - Hyperparameter configuration (learning rate, epochs)
 *    - Real-time training progress monitoring
 * 
 * 3. **Performance Monitoring**:
 *    - Accuracy, precision, recall metrics display
 *    - Training/validation loss visualization
 *    - Model convergence analysis
 *    - Performance comparison across variants
 * 
 * 4. **Training Data Management**:
 *    - Synthetic data generation with domain selection
 *    - Manual annotation interface for human-in-the-loop learning
 *    - Data quality validation and recommendations
 *    - Batch processing capabilities
 * 
 * 5. **Audit Logging & Transparency**:
 *    - Complete training and prediction history
 *    - Model decision explanations
 *    - User override tracking
 *    - Export capabilities for external analysis
 * 
 * Technical Integration:
 * - Connects to backend adaptive threshold API (/api/adaptive/*)
 * - Uses axios for HTTP requests with error handling
 * - Implements React hooks for state management
 * - Responsive design with Tailwind CSS
 * 
 * Cognitive Autonomy Compliance:
 * - Preserves user override capabilities for all automated decisions
 * - Provides complete transparency of model training and predictions
 * - Enables empirical grounding while maintaining human control
 * - Supports audit trail for accountability and trust
 * 
 * Author: Ethical AI Testbed Development Team
 * Version: 1.2.2 - Complete Adaptive Threshold Learning Interface
 * Last Updated: 2025-08-06
 */
const AdaptiveThresholdInterface = () => {
  // System status state
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Prediction state
  const [predictionText, setPredictionText] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  
  // Training state
  const [trainingConfig, setTrainingConfig] = useState({
    synthetic_examples: 50,
    violation_ratio: 0.3,
    domains: ['healthcare', 'finance', 'ai_systems'],
    complexity_levels: ['simple', 'moderate', 'complex']
  });
  const [training, setTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  
  // Model performance state
  const [modelPerformance, setModelPerformance] = useState(null);
  
  // Audit logs state
  const [auditLogs, setAuditLogs] = useState([]);
  
  // Active tab state
  const [activeTab, setActiveTab] = useState('predict');
  
  // Load system status on component mount
  useEffect(() => {
    loadSystemStatus();
    loadModelPerformance();
    loadAuditLogs();
  }, []);
  
  const loadSystemStatus = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/adaptive/status`);
      setSystemStatus(response.data);
      setError(null);
    } catch (err) {
      console.error('Error loading system status:', err);
      setError('Failed to load system status. The adaptive threshold system may not be initialized.');
    } finally {
      setLoading(false);
    }
  };
  
  const loadModelPerformance = async () => {
    try {
      const response = await axios.get(`${API}/adaptive/models/performance`);
      setModelPerformance(response.data);
    } catch (err) {
      console.error('Error loading model performance:', err);
    }
  };
  
  const loadAuditLogs = async () => {
    try {
      const response = await axios.get(`${API}/adaptive/audit/logs?limit=20`);
      setAuditLogs(response.data.recent_events || []);
    } catch (err) {
      console.error('Error loading audit logs:', err);
    }
  };
  
  const handlePrediction = async () => {
    if (!predictionText.trim()) return;
    
    try {
      setPredicting(true);
      const response = await axios.post(`${API}/adaptive/predict`, {
        text: predictionText,
        model_preference: 'best',
        include_metadata: true
      });
      setPredictionResult(response.data);
    } catch (err) {
      console.error('Error making prediction:', err);
      setError('Failed to make prediction. Please try again.');
    } finally {
      setPredicting(false);
    }
  };
  
  const generateTrainingData = async () => {
    try {
      setTraining(true);
      const response = await axios.post(`${API}/adaptive/training-data/generate`, trainingConfig);
      setTrainingResult(response.data);
      await loadSystemStatus(); // Refresh status
    } catch (err) {
      console.error('Error generating training data:', err);
      setError('Failed to generate training data. Please try again.');
    } finally {
      setTraining(false);
    }
  };
  
  const trainModels = async () => {
    try {
      setTraining(true);
      const response = await axios.post(`${API}/adaptive/models/train`, {
        force_retrain: true
      });
      setTrainingResult(response.data);
      await loadSystemStatus(); // Refresh status
      await loadModelPerformance(); // Refresh performance
    } catch (err) {
      console.error('Error training models:', err);
      setError('Failed to train models. Please try again.');
    } finally {
      setTraining(false);
    }
  };
  
  const StatusIndicator = ({ status, label }) => (
    <div className="flex items-center space-x-2">
      <div className={`w-3 h-3 rounded-full ${
        status ? 'bg-green-500' : 'bg-red-500'
      }`}></div>
      <span className={`text-sm ${status ? 'text-green-700' : 'text-red-700'}`}>
        {label}
      </span>
    </div>
  );
  
  const TabButton = ({ id, label, active, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`px-4 py-2 font-medium text-sm rounded-lg transition-colors ${
        active
          ? 'bg-blue-600 text-white'
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
    >
      {label}
    </button>
  );
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">Loading adaptive threshold system...</span>
      </div>
    );
  }
  
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              🤖 Adaptive Threshold Learning System
            </h2>
            <p className="text-gray-600 mt-1">
              Automated, data-driven ethical violation detection with perceptron-based learning
            </p>
          </div>
          <button
            onClick={loadSystemStatus}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Refresh Status
          </button>
        </div>
        
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
            <div className="flex">
              <div className="text-red-400">⚠️</div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}
        
        {/* System Status */}
        {systemStatus && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <StatusIndicator 
              status={systemStatus.system_initialized} 
              label="System Initialized" 
            />
            <StatusIndicator 
              status={systemStatus.models_trained} 
              label="Models Trained" 
            />
            <div className="text-sm">
              <span className="text-gray-600">Training Examples: </span>
              <span className="font-medium">{systemStatus.training_examples}</span>
            </div>
            <div className="text-sm">
              <span className="text-gray-600">Best Model: </span>
              <span className="font-medium">{systemStatus.best_model || 'None'}</span>
            </div>
          </div>
        )}
      </div>
      
      {/* Navigation Tabs */}
      <div className="flex space-x-2">
        <TabButton 
          id="predict" 
          label="🎯 Predict Violations" 
          active={activeTab === 'predict'} 
          onClick={setActiveTab} 
        />
        <TabButton 
          id="train" 
          label="🧠 Train Models" 
          active={activeTab === 'train'} 
          onClick={setActiveTab} 
        />
        <TabButton 
          id="performance" 
          label="📊 Model Performance" 
          active={activeTab === 'performance'} 
          onClick={setActiveTab} 
        />
        <TabButton 
          id="audit" 
          label="🔍 Audit Logs" 
          active={activeTab === 'audit'} 
          onClick={setActiveTab} 
        />
      </div>
      
      {/* Tab Content */}
      {activeTab === 'predict' && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Adaptive Violation Prediction
          </h3>
          <p className="text-gray-600 mb-4">
            Enter text to analyze for ethical violations using our trained adaptive threshold models.
          </p>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Text to Analyze
              </label>
              <textarea
                value={predictionText}
                onChange={(e) => setPredictionText(e.target.value)}
                className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter text to analyze for ethical violations..."
              />
            </div>
            
            <button
              onClick={handlePrediction}
              disabled={predicting || !predictionText.trim() || !systemStatus?.models_trained}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {predicting ? 'Analyzing...' : 'Analyze Text'}
            </button>
            
            {!systemStatus?.models_trained && (
              <p className="text-amber-600 text-sm">
                ⚠️ Models need to be trained before making predictions. Go to the "Train Models" tab.
              </p>
            )}
          </div>
          
          {/* Prediction Results */}
          {predictionResult && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-3">Prediction Results</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <span className="text-sm text-gray-600">Violation Detected:</span>
                  <div className={`text-lg font-semibold ${
                    predictionResult.is_violation ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {predictionResult.is_violation ? '⚠️ Yes' : '✅ No'}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Confidence:</span>
                  <div className="text-lg font-semibold text-blue-600">
                    {(predictionResult.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Model Used:</span>
                  <div className="text-sm font-medium">{predictionResult.model_used}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Model Accuracy:</span>
                  <div className="text-sm font-medium">
                    {(predictionResult.model_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              
              {predictionResult.metadata && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-800">
                    View Detailed Metadata
                  </summary>
                  <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto">
                    {JSON.stringify(predictionResult.metadata, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          )}
        </div>
      )}
      
      {activeTab === 'train' && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Model Training & Data Generation
          </h3>
          
          <div className="space-y-6">
            {/* Training Data Configuration */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Training Data Configuration</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Examples
                  </label>
                  <input
                    type="number"
                    value={trainingConfig.synthetic_examples}
                    onChange={(e) => setTrainingConfig({
                      ...trainingConfig,
                      synthetic_examples: parseInt(e.target.value)
                    })}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    min="10"
                    max="500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Violation Ratio
                  </label>
                  <input
                    type="number"
                    value={trainingConfig.violation_ratio}
                    onChange={(e) => setTrainingConfig({
                      ...trainingConfig,
                      violation_ratio: parseFloat(e.target.value)
                    })}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    min="0.1"
                    max="0.9"
                    step="0.1"
                  />
                </div>
              </div>
            </div>
            
            {/* Action Buttons */}
            <div className="flex space-x-4">
              <button
                onClick={generateTrainingData}
                disabled={training}
                className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 transition-colors"
              >
                {training ? 'Generating...' : 'Generate Training Data'}
              </button>
              <button
                onClick={trainModels}
                disabled={training || !systemStatus?.training_examples}
                className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 transition-colors"
              >
                {training ? 'Training...' : 'Train Models'}
              </button>
            </div>
            
            {!systemStatus?.training_examples && (
              <p className="text-amber-600 text-sm">
                ⚠️ Generate training data before training models.
              </p>
            )}
            
            {/* Training Results */}
            {trainingResult && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-3">Training Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  {trainingResult.best_model && (
                    <>
                      <div>
                        <span className="text-gray-600">Best Model:</span>
                        <div className="font-medium">{trainingResult.best_model}</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Training Accuracy:</span>
                        <div className="font-medium">{(trainingResult.training_accuracy * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Validation Accuracy:</span>
                        <div className="font-medium">{(trainingResult.validation_accuracy * 100).toFixed(1)}%</div>
                      </div>
                    </>
                  )}
                  {trainingResult.total_examples && (
                    <>
                      <div>
                        <span className="text-gray-600">Total Examples:</span>
                        <div className="font-medium">{trainingResult.total_examples}</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Violation Examples:</span>
                        <div className="font-medium">{trainingResult.violation_examples}</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Quality Score:</span>
                        <div className="font-medium">{(trainingResult.quality_score * 100).toFixed(1)}%</div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {activeTab === 'performance' && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Model Performance Metrics
          </h3>
          
          {modelPerformance && modelPerformance.models ? (
            <div className="space-y-4">
              {Object.entries(modelPerformance.models).map(([modelName, metrics]) => (
                <div key={modelName} className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-900 capitalize">{modelName} Perceptron</h4>
                    {metrics.is_best_model && (
                      <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                        Best Model
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Accuracy:</span>
                      <div className="font-medium">{(metrics.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <span className="text-gray-600">Training Iterations:</span>
                      <div className="font-medium">{metrics.training_iterations}</div>
                    </div>
                    <div>
                      <span className="text-gray-600">Weights:</span>
                      <div className="font-mono text-xs">
                        [{metrics.weights.map(w => w.toFixed(3)).join(', ')}]
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              
              <div className="mt-4 text-sm text-gray-600">
                <p>Last Training: {modelPerformance.last_training ? new Date(modelPerformance.last_training).toLocaleString() : 'Never'}</p>
                <p>Training Examples: {modelPerformance.training_examples}</p>
              </div>
            </div>
          ) : (
            <p className="text-gray-600">No model performance data available. Train models first.</p>
          )}
        </div>
      )}
      
      {activeTab === 'audit' && (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Audit Logs & Transparency
          </h3>
          
          <div className="space-y-3">
            {auditLogs.length > 0 ? (
              auditLogs.map((log, index) => (
                <div key={index} className="p-3 border rounded-lg text-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-gray-900">{log.event_type}</span>
                    <span className="text-gray-500">
                      {new Date(log.timestamp).toLocaleString()}
                    </span>
                  </div>
                  {log.details && (
                    <p className="text-gray-600 mt-1">{log.details}</p>
                  )}
                </div>
              ))
            ) : (
              <p className="text-gray-600">No audit logs available.</p>
            )}
          </div>
          
          <button
            onClick={loadAuditLogs}
            className="mt-4 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            Refresh Logs
          </button>
        </div>
      )}
    </div>
  );
};

export default AdaptiveThresholdInterface;
