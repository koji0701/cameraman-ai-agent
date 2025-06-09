import React, { useState, useEffect, useCallback } from 'react';
import { MemoryRouter as Router, Routes, Route } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { Header } from '@/components/layout/Header';
import { CenteredUploadSection } from '@/components/layout/CenteredUploadSection';
import { FileSelection } from '@/components/features/FileSelection';
import { ProcessingOptions } from '@/components/features/ProcessingOptions';
import { ProcessingStatus } from '@/components/features/ProcessingStatus';
import { ApiKeyModal } from '@/components/features/ApiKeyModal';

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
    setShowApiKeyModal(true);
  }, []);

  const handleCloseApiKeyModal = useCallback(() => {
    setShowApiKeyModal(false);
  }, []);

  const handleSaveApiKey = useCallback(async (newApiKey: string) => {
    try {
      const success = await window.electron.saveApiKey(newApiKey.trim());
      if (success) {
        setApiKey(newApiKey.trim());
        setHasApiKey(true);
      } else {
        throw new Error('Failed to save API key');
      }
    } catch (error) {
      console.error('Error saving API key:', error);
      throw error;
    }
  }, []);



  return (
    <AppShell>
      {/* Conditional Layout based on file selection */}
      {!selectedFile ? (
        // Hero Upload Section - shown when no video is selected
        <>
          <Header 
            hasApiKey={hasApiKey}
            onOpenApiKeyModal={handleOpenApiKeyModal}
          />
          <CenteredUploadSection 
            onSelectFile={handleSelectFile}
          />
        </>
      ) : (
        // Full Feature Layout - shown when video is selected
        <>
          <Header 
            hasApiKey={hasApiKey}
            onOpenApiKeyModal={handleOpenApiKeyModal}
          />
          
          <main className="container mx-auto px-6 py-8 space-y-8 max-w-6xl">
            {/* File Selection - Compact version when file is selected */}
            <div className="animate-fade-in-up">
              <FileSelection
                selectedFile={selectedFile}
                outputPath={outputPath}
                onSelectFile={handleSelectFile}
                onOutputPathChange={setOutputPath}
                onOpenOutputFolder={handleOpenOutputFolder}
              />
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Left Column */}
              <div className="space-y-8 animate-slide-in-up" style={{ animationDelay: '0.1s' }}>
                <ProcessingOptions
                  options={options}
                  onOptionsChange={updateOption}
                />
              </div>
              
              {/* Right Column */}
              <div className="space-y-8 animate-slide-in-up" style={{ animationDelay: '0.2s' }}>
                <ProcessingStatus
                  status={processingStatus}
                  logs={logs}
                  onProcess={handleProcessVideo}
                  onCancel={handleCancelProcessing}
                  canProcess={!!selectedFile && !!outputPath && hasApiKey}
                />
              </div>
            </div>
          </main>
        </>
      )}

      <ApiKeyModal
        isOpen={showApiKeyModal}
        onClose={handleCloseApiKeyModal}
        onSave={handleSaveApiKey}
        currentApiKey={apiKey}
      />
    </AppShell>
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
