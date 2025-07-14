import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

console.log('BACKEND_URL:', BACKEND_URL);
console.log('API:', API);

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('evaluate');
  const [evaluationResult, setEvaluationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [parameters, setParameters] = useState({});
  const [calibrationTests, setCalibrationTests] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);

  // Debug: Watch for evaluationResult state changes
  useEffect(() => {
    console.log('🔄 evaluationResult state changed:', evaluationResult);
  }, [evaluationResult]);

  // Load initial parameters and data
  useEffect(() => {
    loadParameters();
    loadCalibrationTests();
    loadPerformanceMetrics();
  }, []);

  const loadParameters = async () => {
    try {
      const response = await axios.get(`${API}/parameters`);
      setParameters(response.data.parameters);
    } catch (error) {
      console.error('Error loading parameters:', error);
    }
  };

  const loadCalibrationTests = async () => {
    try {
      const response = await axios.get(`${API}/calibration-tests`);
      setCalibrationTests(response.data.tests);
    } catch (error) {
      console.error('Error loading calibration tests:', error);
    }
  };

  const loadPerformanceMetrics = async () => {
    try {
      const response = await axios.get(`${API}/performance-metrics`);
      setPerformanceMetrics(response.data.metrics);
    } catch (error) {
      console.error('Error loading performance metrics:', error);
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
            {[
              { id: 'evaluate', label: 'Evaluate Text' },
              { id: 'parameters', label: 'Parameter Tuning' },
              { id: 'calibration', label: 'Calibration Tests' },
              { id: 'metrics', label: 'Performance Metrics' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-md font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="max-w-6xl mx-auto">
          {activeTab === 'evaluate' && (
            <EvaluateTab
              evaluationResult={evaluationResult}
              setEvaluationResult={setEvaluationResult}
              loading={loading}
              setLoading={setLoading}
              parameters={parameters}
            />
          )}
          
          {activeTab === 'parameters' && (
            <ParametersTab
              parameters={parameters}
              setParameters={setParameters}
              loadParameters={loadParameters}
            />
          )}
          
          {activeTab === 'calibration' && (
            <CalibrationTab
              calibrationTests={calibrationTests}
              loadCalibrationTests={loadCalibrationTests}
            />
          )}
          
          {activeTab === 'metrics' && (
            <MetricsTab
              performanceMetrics={performanceMetrics}
              loadPerformanceMetrics={loadPerformanceMetrics}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// Evaluate Text Tab Component
function EvaluateTab({ evaluationResult, setEvaluationResult, loading, setLoading, parameters }) {
  const [inputText, setInputText] = useState('');

  const handleEvaluate = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    console.log('🚀 Starting evaluation for text:', inputText);
    
    try {
      console.log('📡 Making API request to:', `${API}/evaluate`);
      
      const requestData = {
        text: inputText,
        parameters: parameters || {}
      };
      
      console.log('📤 Request data:', requestData);
      
      const response = await axios.post(`${API}/evaluate`, requestData);
      
      console.log('📥 Raw response received:', response);
      console.log('📊 Response status:', response.status);
      console.log('📋 Response data:', response.data);
      console.log('🔍 Response data type:', typeof response.data);
      console.log('🔍 Response data keys:', Object.keys(response.data || {}));
      
      if (response.status === 200 && response.data) {
        console.log('✅ Valid response received, setting evaluation result');
        console.log('📄 Evaluation data:', response.data.evaluation);
        setEvaluationResult(response.data);
        console.log('🎯 Evaluation result state should now be set');
      } else {
        console.error('❌ Invalid response:', {
          status: response.status,
          data: response.data
        });
        alert('Invalid response from server');
      }
    } catch (error) {
      console.error('💥 Error during evaluation:', error);
      console.error('🔍 Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        stack: error.stack
      });
      alert('Error evaluating text: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
      console.log('🏁 Evaluation process finished');
    }
  };

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-2xl font-bold mb-4">Text Evaluation</h2>
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
          <button
            onClick={handleEvaluate}
            disabled={loading || !inputText.trim()}
            className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed mr-4"
          >
            {loading ? 'Evaluating...' : 'Evaluate Text'}
          </button>
          <button
            onClick={() => {
              console.log('Simple test clicked');
              fetch(`${API}/evaluate`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  text: "Hello world"
                })
              })
              .then(response => {
                console.log('Fetch response:', response);
                return response.json();
              })
              .then(data => {
                console.log('Fetch data:', data);
                alert('Fetch success: ' + JSON.stringify(data.evaluation.overall_ethical));
              })
              .catch(error => {
                console.error('Fetch error:', error);
                alert('Fetch error: ' + error.message);
              });
            }}
            className="bg-purple-500 text-white px-4 py-2 rounded-md hover:bg-purple-600 ml-2"
          >
            Fetch Test
          </button>
        </div>
      </div>

      {/* Debug Info */}
      <div className="bg-gray-100 p-2 text-xs text-gray-600 mt-4">
        Debug: evaluationResult is {evaluationResult ? 'SET' : 'NULL'} | 
        Loading: {loading ? 'TRUE' : 'FALSE'} |
        API URL: {API}
      </div>

      {/* Results Section */}
      {evaluationResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {console.log('Rendering results:', evaluationResult)}
          {/* Evaluation Summary */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-xl font-bold mb-4">Evaluation Summary</h3>
            <div className="space-y-3">
              <div className={`p-3 rounded-md ${
                evaluationResult.evaluation.overall_ethical 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                Status: {evaluationResult.evaluation.overall_ethical ? 'Ethical' : 'Unethical'}
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>Processing Time: {evaluationResult.evaluation.processing_time.toFixed(3)}s</div>
                <div>Violations Found: {evaluationResult.evaluation.minimal_violation_count}</div>
                <div>Original Length: {evaluationResult.delta_summary.original_length}</div>
                <div>Clean Length: {evaluationResult.delta_summary.clean_length}</div>
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

          {/* Violations */}
          {evaluationResult.evaluation.minimal_spans.length > 0 && (
            <div className="bg-white p-6 rounded-lg shadow lg:col-span-2">
              <h3 className="text-xl font-bold mb-4">Detected Violations</h3>
              <div className="space-y-3">
                {evaluationResult.evaluation.minimal_spans.map((span, index) => (
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
                      <div className="flex space-x-4">
                        <span className={`px-2 py-1 rounded text-xs ${
                          span.virtue_violation ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                        }`}>
                          Virtue: {span.virtue_score.toFixed(3)}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs ${
                          span.deontological_violation ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                        }`}>
                          Deontological: {span.deontological_score.toFixed(3)}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs ${
                          span.consequentialist_violation ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                        }`}>
                          Consequentialist: {span.consequentialist_score.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

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
  );
}

// Parameters Tab Component
function ParametersTab({ parameters, setParameters, loadParameters }) {
  const [localParameters, setLocalParameters] = useState(parameters);
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    setLocalParameters(parameters);
  }, [parameters]);

  const handleParameterChange = (key, value) => {
    setLocalParameters(prev => ({
      ...prev,
      [key]: parseFloat(value) || 0
    }));
  };

  const handleUpdateParameters = async () => {
    setUpdating(true);
    try {
      await axios.post(`${API}/update-parameters`, {
        parameters: localParameters
      });
      setParameters(localParameters);
      alert('Parameters updated successfully!');
    } catch (error) {
      console.error('Error updating parameters:', error);
      alert('Error updating parameters. Please try again.');
    } finally {
      setUpdating(false);
    }
  };

  const handleResetParameters = () => {
    setLocalParameters(parameters);
  };

  return (
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
                  max="1"
                  step="0.01"
                  value={localParameters[param.key] || 0}
                  onChange={(e) => handleParameterChange(param.key, e.target.value)}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-gray-500">
                  <span>0</span>
                  <span className="font-mono">{(localParameters[param.key] || 0).toFixed(3)}</span>
                  <span>1</span>
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
                  value={localParameters[param.key] || 0}
                  onChange={(e) => handleParameterChange(param.key, e.target.value)}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-gray-500">
                  <span>0</span>
                  <span className="font-mono">{(localParameters[param.key] || 0).toFixed(1)}</span>
                  <span>3</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Span Detection */}
        <div>
          <h3 className="text-lg font-semibold mb-3">Span Detection Parameters</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { key: 'max_span_length', label: 'Max Span Length', min: 1, max: 10 },
              { key: 'min_span_length', label: 'Min Span Length', min: 1, max: 5 }
            ].map((param) => (
              <div key={param.key}>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {param.label}
                </label>
                <input
                  type="range"
                  min={param.min}
                  max={param.max}
                  step="1"
                  value={localParameters[param.key] || param.min}
                  onChange={(e) => handleParameterChange(param.key, e.target.value)}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-gray-500">
                  <span>{param.min}</span>
                  <span className="font-mono">{localParameters[param.key] || param.min}</span>
                  <span>{param.max}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-4">
          <button
            onClick={handleUpdateParameters}
            disabled={updating}
            className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300"
          >
            {updating ? 'Updating...' : 'Update Parameters'}
          </button>
          <button
            onClick={handleResetParameters}
            className="bg-gray-500 text-white px-6 py-2 rounded-md hover:bg-gray-600"
          >
            Reset to Current
          </button>
        </div>
      </div>
    </div>
  );
}

// Calibration Tab Component
function CalibrationTab({ calibrationTests, loadCalibrationTests }) {
  const [newTest, setNewTest] = useState({ text: '', expected_result: 'ethical' });
  const [running, setRunning] = useState(false);

  const handleCreateTest = async () => {
    if (!newTest.text.trim()) return;

    try {
      await axios.post(`${API}/calibration-test`, newTest);
      setNewTest({ text: '', expected_result: 'ethical' });
      loadCalibrationTests();
      alert('Test case created successfully!');
    } catch (error) {
      console.error('Error creating test:', error);
      alert('Error creating test case. Please try again.');
    }
  };

  const handleRunTest = async (testId) => {
    setRunning(true);
    try {
      await axios.post(`${API}/run-calibration-test/${testId}`);
      loadCalibrationTests();
    } catch (error) {
      console.error('Error running test:', error);
      alert('Error running test. Please try again.');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Create New Test */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-2xl font-bold mb-4">Create Calibration Test</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Test Text:
            </label>
            <textarea
              value={newTest.text}
              onChange={(e) => setNewTest(prev => ({ ...prev, text: e.target.value }))}
              placeholder="Enter text to test..."
              className="w-full h-24 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Expected Result:
            </label>
            <select
              value={newTest.expected_result}
              onChange={(e) => setNewTest(prev => ({ ...prev, expected_result: e.target.value }))}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            >
              <option value="ethical">Ethical</option>
              <option value="unethical">Unethical</option>
            </select>
          </div>
          <button
            onClick={handleCreateTest}
            disabled={!newTest.text.trim()}
            className="bg-green-500 text-white px-6 py-2 rounded-md hover:bg-green-600 disabled:bg-gray-300"
          >
            Create Test Case
          </button>
        </div>
      </div>

      {/* Test Cases */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-2xl font-bold mb-4">Test Cases ({calibrationTests.length})</h2>
        <div className="space-y-4">
          {calibrationTests.map((test, index) => (
            <div key={test.id} className="border rounded-md p-4">
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <p className="text-sm text-gray-700 mb-2">{test.text}</p>
                  <div className="flex space-x-4 text-sm">
                    <span className="text-gray-600">Expected: {test.expected_result}</span>
                    {test.actual_result && (
                      <span className={`${
                        test.passed ? 'text-green-600' : 'text-red-600'
                      }`}>
                        Actual: {test.actual_result} {test.passed ? '✓' : '✗'}
                      </span>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => handleRunTest(test.id)}
                  disabled={running}
                  className="bg-blue-500 text-white px-4 py-1 rounded text-sm hover:bg-blue-600 disabled:bg-gray-300"
                >
                  {running ? 'Running...' : 'Run Test'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Metrics Tab Component
function MetricsTab({ performanceMetrics, loadPerformanceMetrics }) {
  useEffect(() => {
    loadPerformanceMetrics();
  }, []);

  if (!performanceMetrics) {
    return (
      <div className="bg-white p-6 rounded-lg shadow text-center">
        <p className="text-gray-500">Loading performance metrics...</p>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Performance Metrics</h2>
        <button
          onClick={loadPerformanceMetrics}
          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
        >
          Refresh
        </button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-gray-50 p-4 rounded-md">
          <h3 className="font-semibold text-gray-700 mb-2">Total Evaluations</h3>
          <p className="text-2xl font-bold text-blue-600">{performanceMetrics.total_evaluations}</p>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-md">
          <h3 className="font-semibold text-gray-700 mb-2">Avg Processing Time</h3>
          <p className="text-2xl font-bold text-green-600">
            {performanceMetrics.average_processing_time.toFixed(3)}s
          </p>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-md">
          <h3 className="font-semibold text-gray-700 mb-2">Throughput</h3>
          <p className="text-2xl font-bold text-purple-600">
            {performanceMetrics.throughput_chars_per_second.toFixed(1)} chars/s
          </p>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-md">
          <h3 className="font-semibold text-gray-700 mb-2">Min Processing Time</h3>
          <p className="text-2xl font-bold text-yellow-600">
            {performanceMetrics.min_processing_time.toFixed(3)}s
          </p>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-md">
          <h3 className="font-semibold text-gray-700 mb-2">Max Processing Time</h3>
          <p className="text-2xl font-bold text-red-600">
            {performanceMetrics.max_processing_time.toFixed(3)}s
          </p>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-md">
          <h3 className="font-semibold text-gray-700 mb-2">Avg Text Length</h3>
          <p className="text-2xl font-bold text-indigo-600">
            {performanceMetrics.average_text_length.toFixed(0)} chars
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;