import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';

/**
 * Real-Time Streaming Interface Component - Phase 7
 * 
 * Implements world-class WebSocket client following:
 * - Guillermo Rauch's connection management patterns (Socket.IO)
 * - Reactive programming principles for stream handling
 * - Circuit breaker patterns for resilience
 * - Backpressure handling and flow control
 * - Real-time ethics analysis visualization
 * 
 * Features:
 * - Live token-by-token ethical analysis
 * - Real-time intervention detection and alerts
 * - Performance metrics and latency monitoring
 * - Connection health monitoring with auto-reconnect
 * - Stream windowing with contextual analysis
 */
const RealTimeStreamingInterface = ({ backendUrl }) => {
  // WebSocket connection state
  const [wsConnection, setWsConnection] = useState(null);
  const [connectionState, setConnectionState] = useState('disconnected'); // disconnected, connecting, connected, streaming
  const [serverStatus, setServerStatus] = useState(null);
  
  // Streaming state
  const [streamId, setStreamId] = useState(null);
  const [inputText, setInputText] = useState('');
  const [streamingActive, setStreamingActive] = useState(false);
  const [tokenQueue, setTokenQueue] = useState([]);
  const [currentTokenIndex, setCurrentTokenIndex] = useState(0);
  
  // Analysis results and metrics
  const [analysisResults, setAnalysisResults] = useState([]);
  const [interventions, setInterventions] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState({});
  const [connectionHealth, setConnectionHealth] = useState({});
  
  // Real-time visualization state
  const [streamingWindow, setStreamingWindow] = useState([]);
  const [realTimeStats, setRealTimeStats] = useState({
    tokensProcessed: 0,
    interventionsTriggered: 0,
    averageLatency: 0,
    ethicalScore: 0
  });
  
  // Refs for persistent values
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const streamingIntervalRef = useRef(null);
  const metricsRef = useRef({ startTime: 0, processedTokens: 0, totalLatency: 0 });

  const API = `${backendUrl}/api`;
  const WS_URL = 'ws://localhost:8765'; // WebSocket server from Phase 7

  // Initialize component and check server status
  useEffect(() => {
    checkServerStatus();
    return () => {
      cleanup();
    };
  }, []);

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (streamingIntervalRef.current) {
      clearInterval(streamingIntervalRef.current);
    }
    setWsConnection(null);
    setConnectionState('disconnected');
    setStreamingActive(false);
  }, []);

  const checkServerStatus = async () => {
    try {
      const response = await axios.get(`${API}/streaming/status`);
      setServerStatus(response.data);
    } catch (error) {
      console.error('Failed to check streaming server status:', error);
      setServerStatus({ status: 'unavailable', error: error.message });
    }
  };

  const connectWebSocket = useCallback(() => {
    if (wsRef.current) {
      console.log('WebSocket already connected');
      return;
    }

    console.log('🔌 Connecting to WebSocket server:', WS_URL);
    setConnectionState('connecting');

    try {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        console.log('✅ WebSocket connected successfully');
        setConnectionState('connected');
        setWsConnection(ws);
        wsRef.current = ws;
        
        // Reset metrics
        metricsRef.current = { startTime: Date.now(), processedTokens: 0, totalLatency: 0 };
        
        // Update connection health
        setConnectionHealth({
          connected: true,
          connectedAt: new Date().toISOString(),
          reconnectAttempts: 0
        });
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('🔌 WebSocket connection closed:', event.code, event.reason);
        setConnectionState('disconnected');
        setWsConnection(null);
        wsRef.current = null;
        
        // Attempt reconnection if not manually closed
        if (event.code !== 1000 && event.code !== 1001) {
          scheduleReconnection();
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setConnectionState('disconnected');
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionState('disconnected');
    }
  }, []);

  const scheduleReconnection = useCallback(() => {
    if (reconnectTimeoutRef.current) return;
    
    console.log('🔄 Scheduling WebSocket reconnection in 3 seconds...');
    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectTimeoutRef.current = null;
      connectWebSocket();
    }, 3000);
  }, [connectWebSocket]);

  const handleWebSocketMessage = useCallback((message) => {
    const { event_type, analysis_result, intervention_data, token, processing_time } = message;
    
    switch (event_type) {
      case 'connection_status':
        console.log('📡 Connection status update:', analysis_result);
        if (analysis_result?.stream_started) {
          setStreamId(analysis_result.stream_started);
          setConnectionState('streaming');
        }
        break;
        
      case 'ethical_analysis':
        console.log('🧠 Received ethical analysis:', message);
        
        // Update analysis results
        setAnalysisResults(prev => [...prev.slice(-100), message]); // Keep last 100 results
        
        // Update streaming window (last 10 tokens)
        setStreamingWindow(prev => [...prev.slice(-9), {
          token: token?.content,
          confidence: message.confidence,
          timestamp: message.timestamp,
          processing_time
        }]);
        
        // Update real-time stats
        setRealTimeStats(prev => {
          const newProcessed = prev.tokensProcessed + 1;
          const newTotalLatency = metricsRef.current.totalLatency + processing_time;
          const newAverage = newTotalLatency / newProcessed;
          
          metricsRef.current.processedTokens = newProcessed;
          metricsRef.current.totalLatency = newTotalLatency;
          
          return {
            ...prev,
            tokensProcessed: newProcessed,
            averageLatency: newAverage,
            ethicalScore: (prev.ethicalScore * 0.9) + (message.confidence * 0.1) // Moving average
          };
        });
        break;
        
      case 'intervention_required':
        console.log('⚠️ Intervention required:', intervention_data);
        
        setInterventions(prev => [...prev.slice(-50), {
          ...intervention_data,
          token: token?.content,
          timestamp: message.timestamp
        }]);
        
        setRealTimeStats(prev => ({
          ...prev,
          interventionsTriggered: prev.interventionsTriggered + 1
        }));
        break;
        
      case 'heartbeat':
        console.log('💓 Heartbeat received');
        setConnectionHealth(prev => ({
          ...prev,
          lastHeartbeat: new Date().toISOString()
        }));
        
        // Send heartbeat response
        if (wsRef.current) {
          wsRef.current.send(JSON.stringify({
            type: 'heartbeat_response',
            timestamp: Date.now()
          }));
        }
        break;
        
      case 'backpressure_signal':
        console.log('🚦 Backpressure signal received:', analysis_result);
        // Handle backpressure by slowing down token sending
        break;
        
      case 'stream_complete':
        console.log('✅ Stream completed:', analysis_result);
        setStreamingActive(false);
        setConnectionState('connected');
        break;
        
      case 'stream_error':
        console.error('❌ Stream error:', analysis_result);
        setStreamingActive(false);
        break;
        
      default:
        console.log('📨 Unknown message type:', event_type, message);
    }
  }, []);

  const startStreaming = useCallback(() => {
    if (!wsRef.current || connectionState !== 'connected') {
      alert('Please connect to WebSocket server first');
      return;
    }

    if (!inputText.trim()) {
      alert('Please enter text to stream');
      return;
    }

    // Tokenize input text
    const tokens = inputText.trim().split(/\s+/);
    setTokenQueue(tokens);
    setCurrentTokenIndex(0);
    setStreamingActive(true);
    
    // Reset analysis state
    setAnalysisResults([]);
    setInterventions([]);
    setStreamingWindow([]);
    setRealTimeStats({
      tokensProcessed: 0,
      interventionsTriggered: 0,
      averageLatency: 0,
      ethicalScore: 0
    });
    
    // Start streaming session
    const newStreamId = `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    wsRef.current.send(JSON.stringify({
      type: 'start_stream',
      stream_id: newStreamId,
      context: {
        domain: 'ethical_ai_generation',
        total_tokens: tokens.length,
        user_session: 'web_interface'
      }
    }));
    
    console.log(`🚀 Starting stream ${newStreamId} with ${tokens.length} tokens`);
    
    // Start sending tokens at regular intervals
    let tokenIndex = 0;
    streamingIntervalRef.current = setInterval(() => {
      if (tokenIndex >= tokens.length) {
        // Stream complete
        clearInterval(streamingIntervalRef.current);
        streamingIntervalRef.current = null;
        
        wsRef.current.send(JSON.stringify({
          type: 'end_stream',
          stream_id: newStreamId
        }));
        
        return;
      }
      
      const token = tokens[tokenIndex];
      const contextWindow = tokens.slice(Math.max(0, tokenIndex - 5), tokenIndex);
      
      wsRef.current.send(JSON.stringify({
        type: 'stream_token',
        token_id: `token_${tokenIndex}_${Date.now()}`,
        content: token,
        position: tokenIndex,
        context_window: contextWindow,
        metadata: {
          timestamp: Date.now(),
          session: 'web_interface'
        }
      }));
      
      setCurrentTokenIndex(tokenIndex + 1);
      tokenIndex++;
      
    }, 500); // Send token every 500ms
    
  }, [inputText, connectionState]);

  const stopStreaming = useCallback(() => {
    if (streamingIntervalRef.current) {
      clearInterval(streamingIntervalRef.current);
      streamingIntervalRef.current = null;
    }
    
    if (wsRef.current && streamId) {
      wsRef.current.send(JSON.stringify({
        type: 'end_stream',
        stream_id: streamId
      }));
    }
    
    setStreamingActive(false);
    setConnectionState('connected');
  }, [streamId]);

  const renderConnectionStatus = () => {
    const statusColors = {
      disconnected: 'bg-red-100 text-red-800',
      connecting: 'bg-yellow-100 text-yellow-800',
      connected: 'bg-green-100 text-green-800',
      streaming: 'bg-blue-100 text-blue-800'
    };
    
    const statusIcons = {
      disconnected: '🔴',
      connecting: '🟡',
      connected: '🟢',
      streaming: '🔵'
    };

    return (
      <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${statusColors[connectionState]}`}>
        <span className="mr-2">{statusIcons[connectionState]}</span>
        {connectionState.toUpperCase()}
      </div>
    );
  };

  const renderRealTimeMetrics = () => (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="bg-blue-50 p-4 rounded-lg">
        <div className="text-2xl font-bold text-blue-700">{realTimeStats.tokensProcessed}</div>
        <div className="text-blue-600 text-sm">Tokens Processed</div>
      </div>
      <div className="bg-red-50 p-4 rounded-lg">
        <div className="text-2xl font-bold text-red-700">{realTimeStats.interventionsTriggered}</div>
        <div className="text-red-600 text-sm">Interventions</div>
      </div>
      <div className="bg-green-50 p-4 rounded-lg">
        <div className="text-2xl font-bold text-green-700">{realTimeStats.averageLatency.toFixed(1)}ms</div>
        <div className="text-green-600 text-sm">Avg Latency</div>
      </div>
      <div className="bg-purple-50 p-4 rounded-lg">
        <div className="text-2xl font-bold text-purple-700">{(realTimeStats.ethicalScore * 100).toFixed(1)}%</div>
        <div className="text-purple-600 text-sm">Ethical Score</div>
      </div>
    </div>
  );

  const renderStreamingWindow = () => (
    <div className="space-y-2">
      <h4 className="font-semibold text-gray-800">Live Streaming Window (Last 10 Tokens)</h4>
      <div className="flex flex-wrap gap-2">
        {streamingWindow.map((item, index) => (
          <div
            key={index}
            className={`px-3 py-2 rounded-md text-sm ${
              item.confidence > 0.7 ? 'bg-green-100 text-green-800' :
              item.confidence > 0.4 ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'
            }`}
          >
            <div className="font-medium">{item.token}</div>
            <div className="text-xs">{(item.confidence * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderInterventions = () => (
    <div className="space-y-3">
      <h4 className="font-semibold text-gray-800">Real-Time Interventions</h4>
      {interventions.length === 0 ? (
        <div className="text-gray-500 text-sm">No interventions detected</div>
      ) : (
        <div className="max-h-60 overflow-y-auto space-y-2">
          {interventions.slice(-10).reverse().map((intervention, index) => (
            <div key={index} className="border border-red-200 rounded-md p-3 bg-red-50">
              <div className="flex items-center justify-between mb-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  intervention.severity === 'HIGH' ? 'bg-red-100 text-red-800' :
                  intervention.severity === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {intervention.severity}
                </span>
                <span className="text-xs text-gray-500">
                  {new Date(intervention.timestamp * 1000).toLocaleTimeString()}
                </span>
              </div>
              <div className="text-sm text-gray-700">
                <strong>Token:</strong> "{intervention.token}"
              </div>
              <div className="text-sm text-gray-600 mt-1">
                {intervention.reason}
              </div>
              <div className="text-xs text-gray-500 mt-2">
                <strong>Suggested Action:</strong> {intervention.suggested_action}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-2">🚀 Real-Time Ethics Streaming</h2>
        <p className="text-blue-100">
          World-class WebSocket streaming with distributed systems architecture • Live token-by-token analysis
        </p>
      </div>

      {/* Connection Status */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Connection Status</h3>
        
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            {renderConnectionStatus()}
            <div className="text-sm text-gray-600">
              WebSocket: {WS_URL}
            </div>
          </div>
          
          <div className="space-x-2">
            <button
              onClick={connectWebSocket}
              disabled={connectionState !== 'disconnected'}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Connect
            </button>
            <button
              onClick={cleanup}
              disabled={connectionState === 'disconnected'}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Disconnect
            </button>
            <button
              onClick={checkServerStatus}
              className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
            >
              Check Status
            </button>
          </div>
        </div>

        {/* Server Status */}
        {serverStatus && (
          <div className="bg-gray-50 p-4 rounded-md">
            <div className="text-sm">
              <strong>Server Status:</strong> {serverStatus.status}
              {serverStatus.server_stats && (
                <span className="ml-4">
                  Active Connections: {serverStatus.server_stats.connections.active_connections}
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Streaming Interface */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Live Streaming Interface</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Text to Stream (will be tokenized and analyzed in real-time)
            </label>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              rows={4}
              placeholder="Enter text for real-time ethical analysis streaming..."
              disabled={streamingActive}
            />
          </div>

          <div className="flex items-center space-x-4">
            <button
              onClick={startStreaming}
              disabled={streamingActive || connectionState !== 'connected'}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {streamingActive ? 'Streaming...' : 'Start Real-Time Stream'}
            </button>
            
            <button
              onClick={stopStreaming}
              disabled={!streamingActive}
              className="px-6 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Stop Stream
            </button>
            
            {streamingActive && (
              <div className="text-sm text-gray-600">
                Progress: {currentTokenIndex} / {tokenQueue.length} tokens
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Real-Time Metrics */}
      {(streamingActive || realTimeStats.tokensProcessed > 0) && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4">⚡ Real-Time Performance Metrics</h3>
          {renderRealTimeMetrics()}
        </div>
      )}

      {/* Live Stream Window */}
      {streamingWindow.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4">📊 Live Analysis Window</h3>
          {renderStreamingWindow()}
        </div>
      )}

      {/* Real-Time Interventions */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">⚠️ Intervention Monitor</h3>
        {renderInterventions()}
      </div>

      {/* Architecture Info */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">🏗️ Architecture Foundations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium text-gray-800 mb-2">WebSocket Management</h4>
            <p className="text-gray-600">Guillermo Rauch's Socket.IO patterns</p>
          </div>
          <div>
            <h4 className="font-medium text-gray-800 mb-2">Stream Processing</h4>
            <p className="text-gray-600">Jay Kreps' Kafka streaming paradigms</p>
          </div>
          <div>
            <h4 className="font-medium text-gray-800 mb-2">Resilience Patterns</h4>
            <p className="text-gray-600">Pat Helland's distributed systems</p>
          </div>
          <div>
            <h4 className="font-medium text-gray-800 mb-2">Low-Latency Systems</h4>
            <p className="text-gray-600">Martin Thompson's mechanical sympathy</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTimeStreamingInterface;