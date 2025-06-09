import React, { useState, useEffect, useCallback } from 'react';
import { MemoryRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

/*
 * AI Cameraman Desktop GUI - OpenCV Only Implementation
 * 
 * This GUI has been updated to REMOVE all FFmpeg implementations and use
 * only OpenCV for video processing. The following changes were made:
 * 
 * REMOVED (FFmpeg-based):
 * - Video Codec selection (h264_videotoolbox, libx264, etc.)
 * - Bitrate selection (10M, 15M, 20M, etc.)
 * - Video Stabilization flag (used FFmpeg vidstab filters)
 * - Color Correction flag (used FFmpeg eq filter)
 * 
 * KEPT (OpenCV-compatible):
 * - Quality presets (low, medium, high, ultra)
 * - disableStreaming flag (always true to ensure OpenCV usage)
 * 
 * The backend will now exclusively use OpenCV for all video processing operations.
 */

// Types
interface ProcessingStatus {
  isProcessing: boolean;
  progress: number;
  stage: string;
  details: string;
}

interface ProcessingOptions {
  quality: 'low' | 'medium' | 'high' | 'ultra';
  disableStreaming: boolean; // Always true to ensure OpenCV usage
}

function AICameramanMain() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string>('');
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus>({
    isProcessing: false,
    progress: 0,
    stage: '',
    details: ''
  });
  const [options, setOptions] = useState<ProcessingOptions>({
    quality: 'medium',
    disableStreaming: true // Always enabled to ensure OpenCV usage
  });
  const [logs, setLogs] = useState<string[]>([]);

  // Listen for processing progress updates
  useEffect(() => {
    const removeListener = window.electron.ipcRenderer.on('processing-progress', (status: ProcessingStatus) => {
      setProcessingStatus(status);
      setLogs(prev => [...prev, `[${status.stage}] ${status.details}`].slice(-10)); // Keep last 10 logs
    });

    return removeListener;
  }, []);

  const handleSelectFile = useCallback(async () => {
    try {
      const filePath = await window.electron.selectVideoFile();
      if (filePath) {
        setSelectedFile(filePath);
        // Auto-generate output path
        const fileName = filePath.split('/').pop()?.replace(/\.[^/.]+$/, '') || 'output';
        const dir = filePath.substring(0, filePath.lastIndexOf('/'));
        setOutputPath(`${dir}/${fileName}_ai_reframed.mp4`);
      }
    } catch (error) {
      console.error('Error selecting file:', error);
      await window.electron.showErrorDialog('File Selection Error', `Failed to select file: ${error}`);
    }
  }, []);

  const handleProcessVideo = useCallback(async () => {
    if (!selectedFile || !outputPath) {
      await window.electron.showErrorDialog('Missing Information', 'Please select an input file and specify an output path.');
      return;
    }

    try {
      setLogs(['Starting video processing...']);
      await window.electron.processVideo(selectedFile, outputPath, options);
      await window.electron.showInfoDialog('Success', 'Video processing completed successfully!');
    } catch (error) {
      console.error('Processing error:', error);
      await window.electron.showErrorDialog('Processing Error', `Failed to process video: ${error}`);
    }
  }, [selectedFile, outputPath, options]);

  const handleCancelProcessing = useCallback(async () => {
    try {
      await window.electron.cancelProcessing();
      setLogs(prev => [...prev, 'Processing cancelled by user']);
    } catch (error) {
      console.error('Cancel error:', error);
    }
  }, []);

  const handleOpenOutputFolder = useCallback(async () => {
    if (outputPath) {
      await window.electron.openOutputFolder(outputPath);
    }
  }, [outputPath]);

  const updateOption = useCallback((key: keyof ProcessingOptions, value: any) => {
    setOptions(prev => ({ ...prev, [key]: value }));
  }, []);

  return (
    <div className="ai-cameraman-app">
      <header className="app-header">
        <h1>🎬 AI Cameraman</h1>
        <p>Intelligent video reframing and enhancement</p>
      </header>

      <main className="app-main">
        {/* File Selection */}
        <section className="file-section">
          <h2>📁 Video Input</h2>
          <div className="file-input-group">
            <button 
              onClick={handleSelectFile}
              className="btn btn-primary"
              disabled={processingStatus.isProcessing}
            >
              Select Video File
            </button>
            {selectedFile && (
              <div className="selected-file">
                <span className="file-name">📹 {selectedFile.split('/').pop()}</span>
              </div>
            )}
          </div>
          
          {selectedFile && (
            <div className="output-path-group">
              <label htmlFor="output-path">Output Path:</label>
              <input
                id="output-path"
                type="text"
                value={outputPath}
                onChange={(e) => setOutputPath(e.target.value)}
                disabled={processingStatus.isProcessing}
                className="path-input"
                placeholder="Enter output file path..."
              />
            </div>
          )}
        </section>

        {/* Processing Options */}
        {selectedFile && (
          <section className="options-section">
            <h2>⚙️ Processing Options</h2>
            <div className="options-grid">
              <div className="option-group">
                <label>Quality Preset:</label>
                <select 
                  value={options.quality}
                  onChange={(e) => updateOption('quality', e.target.value)}
                  disabled={processingStatus.isProcessing}
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="ultra">Ultra</option>
                </select>
              </div>

              <div className="option-group info-group">
                <label>Processing Mode:</label>
                <div className="info-text">
                  <strong>✅ OpenCV Mode Enabled</strong>
                  <br />
                  <small>Uses pure OpenCV for video processing (no FFmpeg dependencies)</small>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Processing Controls */}
        {selectedFile && outputPath && (
          <section className="controls-section">
            <h2>🚀 Processing</h2>
            <div className="control-buttons">
              {!processingStatus.isProcessing ? (
                <button 
                  onClick={handleProcessVideo}
                  className="btn btn-success btn-large"
                >
                  Start AI Processing
                </button>
              ) : (
                <button 
                  onClick={handleCancelProcessing}
                  className="btn btn-danger btn-large"
                >
                  Cancel Processing
                </button>
              )}
              
              {outputPath && (
                <button 
                  onClick={handleOpenOutputFolder}
                  className="btn btn-secondary"
                  disabled={processingStatus.isProcessing}
                >
                  Open Output Folder
                </button>
              )}
            </div>
          </section>
        )}

        {/* Progress Display */}
        {processingStatus.isProcessing && (
          <section className="progress-section">
            <h2>📊 Processing Progress</h2>
            <div className="progress-info">
              <div className="progress-bar-container">
                <div 
                  className="progress-bar"
                  style={{ width: `${processingStatus.progress}%` }}
                />
                <span className="progress-text">
                  {processingStatus.progress.toFixed(1)}%
                </span>
              </div>
              <div className="progress-details">
                <div className="stage">Stage: {processingStatus.stage}</div>
                <div className="details">{processingStatus.details}</div>
              </div>
            </div>
          </section>
        )}

        {/* Logs */}
        {logs.length > 0 && (
          <section className="logs-section">
            <h2>📝 Processing Logs</h2>
            <div className="logs-container">
              {logs.map((log, index) => (
                <div key={index} className="log-entry">
                  {log}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Status Info */}
        <section className="status-section">
          <div className="status-info">
            <span>Status: {processingStatus.isProcessing ? '🔄 Processing' : '✅ Ready'}</span>
          </div>
        </section>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AICameramanMain />} />
      </Routes>
    </Router>
  );
}
