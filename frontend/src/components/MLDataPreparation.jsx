import React, { useState, useEffect } from 'react';
import axios from 'axios';

/**
 * ML Data Preparation Component
 * 
 * This component provides an interface for uploading text files, processing them
 * through the ethical evaluation engine, and downloading the results with ethical
 * and intent vectors attached for ML training.
 * 
 * Features:
 * - File upload with drag-and-drop support
 * - Progress tracking during processing
 * - Download of processed files
 * - Preview of output structure
 */
const MLDataPreparation = ({ backendUrl }) => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingResult, setProcessingResult] = useState(null);
  const [processedFiles, setProcessedFiles] = useState([]);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [sampleOutput, setSampleOutput] = useState(null);
  const [error, setError] = useState(null);
  
  const API = `${backendUrl}/api/ml-data`;
  
  // Load the list of processed files on component mount
  useEffect(() => {
    fetchProcessedFiles();
    fetchSampleOutput();
  }, []);
  
  // Fetch the list of already processed files
  const fetchProcessedFiles = async () => {
    try {
      const response = await axios.get(`${API}/list-outputs`);
      setProcessedFiles(response.data);
    } catch (err) {
      console.error('Error fetching processed files:', err);
      setError('Failed to load processed files. The ML data preparation service may not be available.');
    }
  };
  
  // Fetch a sample output for preview
  const fetchSampleOutput = async () => {
    try {
      const response = await axios.get(`${API}/sample-output`);
      setSampleOutput(response.data);
    } catch (err) {
      console.error('Error fetching sample output:', err);
    }
  };
  
  // Handle file selection via input
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };
  
  // Handle file drop
  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setError(null);
    }
  };
  
  // Prevent default drag behaviors
  const handleDragOver = (event) => {
    event.preventDefault();
  };
  
  // Upload and process the file
  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }
    
    // Check file extension
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const allowedExtensions = ['txt', 'json', 'jsonl'];
    
    if (!allowedExtensions.includes(fileExtension)) {
      setError(`Unsupported file format. Allowed formats: ${allowedExtensions.join(', ')}`);
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(0);
    setError(null);
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Upload with progress tracking
      const response = await axios.post(`${API}/upload-process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      });
      
      setProcessingResult(response.data);
      fetchProcessedFiles(); // Refresh the list
    } catch (err) {
      console.error('Upload failed:', err);
      setError(err.response?.data?.detail || 'File upload and processing failed');
    } finally {
      setIsUploading(false);
    }
  };
  
  // Download a processed file
  const handleDownload = async (filename) => {
    try {
      window.open(`${API}/download/${filename}`, '_blank');
    } catch (err) {
      console.error('Download failed:', err);
      setError('Failed to download the file');
    }
  };
  
  // Toggle preview visibility
  const togglePreview = () => {
    setPreviewVisible(!previewVisible);
  };
  
  // Reset the form
  const handleReset = () => {
    setFile(null);
    setProcessingResult(null);
    setError(null);
    setUploadProgress(0);
  };
  
  // Format file size for display
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
    else return (bytes / 1048576).toFixed(2) + ' MB';
  };
  
  // Format timestamp for display
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  return (
    <div className="ml-data-preparation p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-semibold mb-4">ML Data Preparation</h2>
      
      <div className="mb-6">
        <p className="text-gray-700 mb-4">
          Upload text files to process them through the ethical evaluation engine and generate ML-ready training data 
          with ethical vectors and intent vectors.
        </p>
        
        {/* File Upload Section */}
        <div 
          className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center cursor-pointer"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => document.getElementById('file-input').click()}
        >
          <input 
            type="file" 
            id="file-input"
            className="hidden" 
            onChange={handleFileChange}
            accept=".txt,.json,.jsonl"
          />
          
          {file ? (
            <div className="text-green-600">
              <p>Selected file: <strong>{file.name}</strong> ({formatFileSize(file.size)})</p>
              <p className="text-sm text-gray-500 mt-2">Click or drag another file to change selection</p>
            </div>
          ) : (
            <div>
              <p className="text-lg font-semibold text-gray-700">
                Drag & Drop a file here, or click to select
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Supports .txt, .json, and .jsonl files
              </p>
            </div>
          )}
        </div>
        
        {/* Error Message */}
        {error && (
          <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md">
            <p>{error}</p>
          </div>
        )}
        
        {/* Upload Controls */}
        <div className="mt-4 flex space-x-4">
          <button
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-400"
            onClick={handleUpload}
            disabled={!file || isUploading}
          >
            {isUploading ? 'Processing...' : 'Process File'}
          </button>
          
          <button
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
            onClick={handleReset}
            disabled={isUploading}
          >
            Reset
          </button>
          
          <button
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            onClick={togglePreview}
          >
            {previewVisible ? 'Hide Sample Output' : 'Show Sample Output'}
          </button>
        </div>
        
        {/* Progress Bar */}
        {isUploading && (
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-600 mt-1">Processing: {uploadProgress}%</p>
          </div>
        )}
        
        {/* Processing Result */}
        {processingResult && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <h3 className="text-lg font-semibold text-green-800">Processing Complete</h3>
            <div className="mt-2">
              <p><strong>Input File:</strong> {processingResult.input_file}</p>
              <p><strong>Output File:</strong> {processingResult.output_file}</p>
              <p><strong>Items Processed:</strong> {processingResult.items_processed}</p>
              <button
                className="mt-3 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
                onClick={() => handleDownload(processingResult.output_file)}
              >
                Download Processed File
              </button>
            </div>
          </div>
        )}
      </div>
      
      {/* Sample Output Preview */}
      {previewVisible && sampleOutput && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold mb-3">Sample Output Structure</h3>
          <div className="bg-gray-50 p-4 rounded-md overflow-auto max-h-96">
            <pre className="text-sm">
              {JSON.stringify(sampleOutput, null, 2)}
            </pre>
          </div>
          <p className="mt-2 text-sm text-gray-600">
            This is a sample of the output format. Your processed files will contain similar structure
            but with your specific content and ethical evaluation results.
          </p>
        </div>
      )}
      
      {/* Processed Files List */}
      <div className="mt-8">
        <h3 className="text-xl font-semibold mb-3">Previously Processed Files</h3>
        {processedFiles.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200">
              <thead>
                <tr>
                  <th className="py-2 px-4 border-b text-left">Filename</th>
                  <th className="py-2 px-4 border-b text-left">Size</th>
                  <th className="py-2 px-4 border-b text-left">Created</th>
                  <th className="py-2 px-4 border-b text-left">Action</th>
                </tr>
              </thead>
              <tbody>
                {processedFiles.map((file, index) => (
                  <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : ''}>
                    <td className="py-2 px-4 border-b">{file.filename}</td>
                    <td className="py-2 px-4 border-b">{formatFileSize(file.size_bytes)}</td>
                    <td className="py-2 px-4 border-b">{formatTimestamp(file.created_at)}</td>
                    <td className="py-2 px-4 border-b">
                      <button
                        className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                        onClick={() => handleDownload(file.filename)}
                      >
                        Download
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-600">No processed files available yet.</p>
        )}
      </div>
      
      {/* Information Section */}
      <div className="mt-8 p-4 bg-blue-50 rounded-lg">
        <h3 className="text-lg font-semibold text-blue-800 mb-2">About ML Data Preparation</h3>
        <div className="text-gray-700">
          <p className="mb-2">
            This tool helps you prepare training data for ML models by enriching your text with:
          </p>
          <ul className="list-disc list-inside mb-3 ml-2">
            <li>Comprehensive ethical vectors (virtue, deontological, consequentialist)</li>
            <li>Intent hierarchy classification vectors</li>
            <li>Aggregate ethical metrics and scores</li>
          </ul>
          <p className="mb-2">
            <strong>Supported File Formats:</strong>
          </p>
          <ul className="list-disc list-inside ml-2">
            <li><strong>Plain Text (.txt)</strong>: Each paragraph will be treated as a separate item</li>
            <li><strong>JSON (.json)</strong>: Single JSON object or array of objects</li>
            <li><strong>JSON Lines (.jsonl)</strong>: Each line contains a separate JSON object</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default MLDataPreparation;
