import React, { useCallback, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Upload, FileVideo, Folder, CheckCircle, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FileSelectionProps {
  selectedFile: string | null;
  outputPath: string;
  onSelectFile: () => void;
  onOutputPathChange: (path: string) => void;
  onOpenOutputFolder: () => void;
  className?: string;
}

export function FileSelection({
  selectedFile,
  outputPath,
  onSelectFile,
  onOutputPathChange,
  onOpenOutputFolder,
  className
}: FileSelectionProps) {
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

  const getFileName = (path: string) => {
    return path.split('/').pop() || path;
  };

  const getFileSize = (path: string) => {
    // TODO: Get actual file size
    return '~125 MB';
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* File Selection Card */}
      <Card className="glass-card border-glass-white/20 p-0 overflow-hidden shadow-glass-lg">
        <div className="p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <FileVideo className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Input Video</h3>
              <p className="text-sm text-muted-foreground">Select a video file to enhance</p>
            </div>
          </div>

          {selectedFile ? (
            <div className="space-y-4">
              <div className="glass-panel rounded-lg p-4 border-emerald-500/20 bg-emerald-500/5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="font-medium text-white">{getFileName(selectedFile)}</p>
                      <div className="flex items-center space-x-3 mt-1">
                        <Badge variant="secondary" className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                          {getFileSize(selectedFile)}
                        </Badge>
                        <span className="text-xs text-muted-foreground">{selectedFile}</span>
                      </div>
                    </div>
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={onSelectFile}
                    className="glass-panel border-glass-white/20 bg-glass-white/10 hover:bg-glass-white/20 text-white"
                  >
                    Change
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <div
              className={cn(
                "glass-panel rounded-xl border-2 border-dashed p-8 text-center transition-all duration-200",
                isDragOver 
                  ? "border-accent-purple bg-accent-purple/10 scale-105" 
                  : "border-glass-white/30 hover:border-accent-purple/50 hover:bg-glass-white/5"
              )}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload className={cn(
                "w-12 h-12 mx-auto mb-4 transition-colors",
                isDragOver ? "text-accent-purple" : "text-muted-foreground"
              )} />
              <h4 className="text-lg font-medium text-white mb-2">
                Drop your video here
              </h4>
              <p className="text-sm text-muted-foreground mb-4">
                Or click to browse files
              </p>
              <Button 
                onClick={onSelectFile}
                className="bg-gradient-to-r from-accent-purple to-accent-teal hover:from-accent-purple/80 hover:to-accent-teal/80 text-white border-0 glow-hover"
              >
                <Upload className="w-4 h-4 mr-2" />
                Select Video File
              </Button>
              <div className="mt-4 text-xs text-muted-foreground">
                Supports MP4, MOV, AVI, MKV files
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Output Path Card */}
      <Card className="glass-card border-glass-white/20 p-6 shadow-glass-lg">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
            <Folder className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Output Location</h3>
            <p className="text-sm text-muted-foreground">Where to save the enhanced video</p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex space-x-3">
            <Input
              value={outputPath}
              onChange={(e) => onOutputPathChange(e.target.value)}
              placeholder="Enter output file path..."
              className="flex-1 glass-panel border-glass-white/20 bg-glass-white/5 text-white placeholder:text-muted-foreground focus:border-accent-purple/50 focus:ring-accent-purple/30"
            />
            <Button
              variant="outline"
              onClick={onOpenOutputFolder}
              disabled={!outputPath}
              className="glass-panel border-glass-white/20 bg-glass-white/10 hover:bg-glass-white/20 text-white transition-all duration-200"
            >
              <Folder className="w-4 h-4" />
            </Button>
          </div>
          
          {outputPath && (
            <div className="glass-panel rounded-lg p-3 bg-blue-500/5 border-blue-500/20">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-blue-400">
                  Output: {getFileName(outputPath)}
                </span>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
} 