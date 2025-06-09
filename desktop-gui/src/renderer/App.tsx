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
    details: '',
  });
  const [options, setOptions] = useState<ProcessingOptions>({
    quality: 'medium',
    disableStreaming: true, // Always enabled to ensure OpenCV usage
  });
  const [logs, setLogs] = useState<string[]>([]);

  // API Key management state
  const [showApiKeyModal, setShowApiKeyModal] = useState<boolean>(false);
  const [apiKey, setApiKey] = useState<string>('');
  const [hasApiKey, setHasApiKey] = useState<boolean>(false);
  const [apiKeyInput, setApiKeyInput] = useState<string>('');
  const [apiKeyStatus, setApiKeyStatus] = useState<{
    type: 'success' | 'error' | null;
    message: string;
  }>({ type: null, message: '' });

  // Listen for processing progress updates
  useEffect(() => {
    const removeListener = window.electron.ipcRenderer.on(
      'processing-progress',
      (status: ProcessingStatus) => {
        setProcessingStatus(status);
        setLogs((prev) =>
          [...prev, `[${status.stage}] ${status.details}`].slice(-10),
        ); // Keep last 10 logs
      },
    );

    return removeListener;
  }, []);

  // Load API key on startup
  useEffect(() => {
    const loadApiKey = async () => {
      try {
        const savedApiKey = await window.electron.getApiKey();
        if (savedApiKey) {
          setApiKey(savedApiKey);
          setHasApiKey(true);
        }
      } catch (error) {
        console.error('Error loading API key:', error);
      }
    };

    loadApiKey();
  }, []);

  const handleSelectFile = useCallback(async () => {
    try {
      const filePath = await window.electron.selectVideoFile();
      if (filePath) {
        setSelectedFile(filePath);
        // Auto-generate output path
        const fileName =
          filePath
            .split('/')
            .pop()
            ?.replace(/\.[^/.]+$/, '') || 'output';
        const dir = filePath.substring(0, filePath.lastIndexOf('/'));
        setOutputPath(`${dir}/${fileName}_ai_reframed.mp4`);
      }
    } catch (error) {
      console.error('Error selecting file:', error);
      await window.electron.showErrorDialog(
        'File Selection Error',
        `Failed to select file: ${error}`,
      );
    }
  }, []);

  const handleProcessVideo = useCallback(async () => {
    if (!selectedFile || !outputPath) {
      await window.electron.showErrorDialog(
        'Missing Information',
        'Please select an input file and specify an output path.',
      );
      return;
    }

    if (!hasApiKey) {
      await window.electron.showErrorDialog(
        'API Key Required',
        'Please configure your Gemini API key before processing videos. Click the "Set API Key" button in the top-right corner.',
      );
      return;
    }

    try {
      setLogs(['Starting video processing...']);
      await window.electron.processVideo(selectedFile, outputPath, options);
      await window.electron.showInfoDialog(
        'Success',
        'Video processing completed successfully!',
      );
    } catch (error) {
      console.error('Processing error:', error);
      const errorMessage = String(error);

      if (
        errorMessage.includes('API key not configured') ||
        errorMessage.includes('GOOGLE_API_KEY')
      ) {
        await window.electron.showErrorDialog(
          'API Key Error',
          'Your Gemini API key is not configured or invalid. Please check your API key settings.',
        );
      } else {
        await window.electron.showErrorDialog(
          'Processing Error',
          `Failed to process video: ${error}`,
        );
      }
    }
  }, [selectedFile, outputPath, options, hasApiKey]);

  const handleCancelProcessing = useCallback(async () => {
    try {
      await window.electron.cancelProcessing();
      setLogs((prev) => [...prev, 'Processing cancelled by user']);
    } catch (error) {
      console.error('Cancel error:', error);
    }
  }, []);

  const handleOpenOutputFolder = useCallback(async () => {
    if (outputPath) {
      await window.electron.openOutputFolder(outputPath);
    }
  }, [outputPath]);

  const updateOption = useCallback(
    (key: keyof ProcessingOptions, value: any) => {
      setOptions((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  // API Key management functions
  const handleOpenApiKeyModal = useCallback(() => {
    setApiKeyInput(apiKey);
    setApiKeyStatus({ type: null, message: '' });
    setShowApiKeyModal(true);
  }, [apiKey]);

  const handleCloseApiKeyModal = useCallback(() => {
    setShowApiKeyModal(false);
    setApiKeyInput('');
    setApiKeyStatus({ type: null, message: '' });
  }, []);

  const handleSaveApiKey = useCallback(async () => {
    if (!apiKeyInput.trim()) {
      setApiKeyStatus({ type: 'error', message: 'Please enter an API key' });
      return;
    }

    // Basic validation for Gemini API key format
    const apiKeyPattern = /^AI[a-zA-Z0-9_-]{37}$/;
    if (!apiKeyPattern.test(apiKeyInput.trim())) {
      setApiKeyStatus({
        type: 'error',
        message:
          'Invalid API key format. Gemini API keys start with "AI" and are 39 characters long.',
      });
      return;
    }

    try {
      const success = await window.electron.saveApiKey(apiKeyInput.trim());
      if (success) {
        setApiKey(apiKeyInput.trim());
        setHasApiKey(true);
        setApiKeyStatus({
          type: 'success',
          message: 'API key saved successfully!',
        });

        // Close modal after a short delay
        setTimeout(() => {
          handleCloseApiKeyModal();
        }, 1500);
      } else {
        setApiKeyStatus({ type: 'error', message: 'Failed to save API key' });
      }
    } catch (error) {
      console.error('Error saving API key:', error);
      setApiKeyStatus({ type: 'error', message: 'Error saving API key' });
    }
  }, [apiKeyInput, handleCloseApiKeyModal]);

  const handleDeleteApiKey = useCallback(async () => {
    try {
      const success = await window.electron.deleteApiKey();
      if (success) {
        setApiKey('');
        setHasApiKey(false);
        setApiKeyStatus({
          type: 'success',
          message: 'API key deleted successfully!',
        });

        // Close modal after a short delay
        setTimeout(() => {
          handleCloseApiKeyModal();
        }, 1500);
      } else {
        setApiKeyStatus({ type: 'error', message: 'Failed to delete API key' });
      }
    } catch (error) {
      console.error('Error deleting API key:', error);
      setApiKeyStatus({ type: 'error', message: 'Error deleting API key' });
    }
  }, [handleCloseApiKeyModal]);

  return (
    <div className="ai-cameraman-app">
      {/* API Key Button */}
      <button
        type="button"
        onClick={handleOpenApiKeyModal}
        className={`api-key-button ${hasApiKey ? 'has-key' : ''}`}
        title={hasApiKey ? 'API key configured' : 'Configure Gemini API key'}
      >
        🔑 {hasApiKey ? 'API Key Set' : 'Set API Key'}
      </button>

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
              type="button"
              onClick={handleSelectFile}
              className="btn btn-primary"
              disabled={processingStatus.isProcessing}
            >
              Select Video File
            </button>
            {selectedFile && (
              <div className="selected-file">
                <span className="file-name">
                  📹 {selectedFile.split('/').pop()}
                </span>
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
                <label htmlFor="quality-preset">Quality Preset:</label>
                <select
                  id="quality-preset"
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
                <span>Processing Mode:</span>
                <div className="info-text">
                  <strong>✅ OpenCV Mode Enabled</strong>
                  <br />
                  <small>
                    Uses pure OpenCV for video processing (no FFmpeg
                    dependencies)
                  </small>
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
                  type="button"
                  onClick={handleProcessVideo}
                  className="btn btn-success btn-large"
                  disabled={!hasApiKey}
                  title={
                    !hasApiKey
                      ? 'Please configure your Gemini API key first'
                      : ''
                  }
                >
                  {!hasApiKey ? '🔑 API Key Required' : 'Start AI Processing'}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleCancelProcessing}
                  className="btn btn-danger btn-large"
                >
                  Cancel Processing
                </button>
              )}

              {outputPath && (
                <button
                  type="button"
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
                <div key={`log-${index}-${log.slice(0, 20)}`} className="log-entry">
                  {log}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Status Info */}
        <section className="status-section">
          <div className="status-info">
            <span>
              Status:{' '}
              {processingStatus.isProcessing ? '🔄 Processing' : '✅ Ready'}
            </span>
          </div>
        </section>
      </main>

      {/* API Key Modal */}
      {showApiKeyModal && (
        <div 
          className="modal-overlay" 
          onClick={handleCloseApiKeyModal}
          onKeyDown={(e) => e.key === 'Escape' && handleCloseApiKeyModal()}
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
        >
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>🔑 Gemini API Key</h2>
              <p>
                Enter your Gemini API key to enable AI video processing. Your
                key is stored locally and never shared.
              </p>
            </div>

            <div className="modal-body">
              <div className="form-group">
                <label htmlFor="api-key-input">API Key:</label>
                <input
                  id="api-key-input"
                  type="password"
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  placeholder="AIzaSyD..."
                />
                <div className="help-text">
                  Get your free API key from{' '}
                  <a
                    href="https://ai.google.dev/gemini-api/docs/api-key"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Google AI Studio
                  </a>
                </div>
              </div>

              {apiKeyStatus.type && (
                <div className={`api-key-status ${apiKeyStatus.type}`}>
                  {apiKeyStatus.type === 'success' ? '✅' : '❌'}{' '}
                  {apiKeyStatus.message}
                </div>
              )}
            </div>

            <div className="modal-footer">
              {hasApiKey && (
                <button
                  type="button"
                  onClick={handleDeleteApiKey}
                  className="btn btn-danger"
                  disabled={apiKeyStatus.type === 'success'}
                >
                  Delete Key
                </button>
              )}
              <button
                type="button"
                onClick={handleCloseApiKeyModal}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveApiKey}
                className="btn btn-primary"
                disabled={apiKeyStatus.type === 'success'}
              >
                Save Key
              </button>
            </div>
          </div>
        </div>
      )}
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
