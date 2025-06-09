import React, { useCallback, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Upload, Sparkles, FileVideo } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CenteredUploadSectionProps {
  onSelectFile: () => void;
  className?: string;
}

export function CenteredUploadSection({ onSelectFile, className }: CenteredUploadSectionProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    // TODO: Handle file drop
    console.log('File dropped');
  }, []);

  return (
    <div className={cn(
      "min-h-screen flex items-center justify-center p-8 relative",
      className
    )}>
      {/* Hero Container */}
      <div className="w-full max-w-2xl mx-auto text-center space-y-8 animate-fade-in-up">
        
        {/* Main Hero Content */}
        <div className="space-y-6">
          {/* Logo/Icon */}
          <div className="relative mx-auto w-24 h-24 mb-8">
            <div className="absolute inset-0 bg-gradient-to-br from-accent-purple to-accent-teal rounded-2xl glass-glow animate-breathe" />
            <div className="relative w-full h-full bg-gradient-to-br from-accent-purple to-accent-teal rounded-2xl flex items-center justify-center">
              <Sparkles className="w-12 h-12 text-white" />
            </div>
          </div>

          {/* Hero Text */}
          <div className="space-y-4">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-white via-gray-100 to-gray-300 bg-clip-text text-transparent leading-tight">
              AI Cameraman
            </h1>
            <p className="text-xl text-gray-400 max-w-lg mx-auto leading-relaxed">
              Transform your videos with intelligent AI-powered reframing and enhancement
            </p>
          </div>
        </div>

        {/* Upload Section */}
        <div 
          className={cn(
            "glass-card rounded-3xl p-12 transition-all duration-300 transform",
            isDragOver 
              ? "scale-105 border-accent-purple/50 bg-accent-purple/5 shadow-glow-purple" 
              : "hover:scale-[1.02] hover:shadow-glass-lg"
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="space-y-6">
            {/* Upload Icon */}
            <div className={cn(
              "w-20 h-20 mx-auto rounded-2xl flex items-center justify-center transition-all duration-300",
              isDragOver 
                ? "bg-accent-purple/20 text-accent-purple scale-110" 
                : "bg-white/5 text-gray-400 group-hover:text-white group-hover:bg-white/10"
            )}>
              <Upload className="w-10 h-10" />
            </div>

            {/* Upload Text */}
            <div className="space-y-3">
              <h3 className={cn(
                "text-2xl font-semibold transition-colors duration-300",
                isDragOver ? "text-accent-purple" : "text-white"
              )}>
                {isDragOver ? "Drop your video here" : "Upload your video"}
              </h3>
              <p className="text-gray-400 max-w-md mx-auto">
                Drag and drop your video file here, or click to browse. 
                Supports MP4, MOV, AVI, and MKV formats.
              </p>
            </div>

            {/* Upload Button */}
            <Button 
              onClick={onSelectFile}
              size="lg"
              className={cn(
                "bg-gradient-to-r from-accent-purple to-accent-teal hover:from-accent-purple/80 hover:to-accent-teal/80",
                "text-white border-0 px-8 py-4 text-lg font-medium rounded-xl",
                "transition-all duration-300 glow-hover",
                "shadow-glass-md hover:shadow-glow-purple"
              )}
            >
              <FileVideo className="w-5 h-5 mr-3" />
              Choose Video File
            </Button>

            {/* Format Support */}
            <div className="flex items-center justify-center space-x-6 text-sm text-gray-500 pt-4">
              <span className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-green-400"></div>
                <span>MP4</span>
              </span>
              <span className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-400"></div>
                <span>MOV</span>
              </span>
              <span className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                <span>AVI</span>
              </span>
              <span className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-orange-400"></div>
                <span>MKV</span>
              </span>
            </div>
          </div>
        </div>

        {/* Feature Highlights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-8">
          <div className="glass-panel rounded-2xl p-6 text-center hover:glass-glow transition-all duration-300">
            <div className="w-12 h-12 mx-auto mb-4 bg-gradient-to-br from-blue-500/20 to-purple-600/20 rounded-xl flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-blue-400" />
            </div>
            <h4 className="text-white font-medium mb-2">AI-Powered</h4>
            <p className="text-gray-400 text-sm">Intelligent scene analysis and automated framing</p>
          </div>
          
          <div className="glass-panel rounded-2xl p-6 text-center hover:glass-glow transition-all duration-300">
            <div className="w-12 h-12 mx-auto mb-4 bg-gradient-to-br from-green-500/20 to-teal-600/20 rounded-xl flex items-center justify-center">
              <FileVideo className="w-6 h-6 text-green-400" />
            </div>
            <h4 className="text-white font-medium mb-2">High Quality</h4>
            <p className="text-gray-400 text-sm">Preserve video quality with advanced processing</p>
          </div>
          
          <div className="glass-panel rounded-2xl p-6 text-center hover:glass-glow transition-all duration-300">
            <div className="w-12 h-12 mx-auto mb-4 bg-gradient-to-br from-orange-500/20 to-red-600/20 rounded-xl flex items-center justify-center">
              <Upload className="w-6 h-6 text-orange-400" />
            </div>
            <h4 className="text-white font-medium mb-2">Easy to Use</h4>
            <p className="text-gray-400 text-sm">Simple drag-and-drop interface for quick processing</p>
          </div>
        </div>
      </div>
    </div>
  );
} 