import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Play, 
  Square, 
  CheckCircle, 
  Loader2, 
  ChevronDown, 
  ChevronUp,
  Terminal,
  Activity
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProcessingStatus {
  isProcessing: boolean;
  progress: number;
  stage: string;
  details: string;
}

interface ProcessingStatusProps {
  status: ProcessingStatus;
  logs: string[];
  onProcess: () => void;
  onCancel: () => void;
  canProcess: boolean;
  className?: string;
}

export function ProcessingStatus({
  status,
  logs,
  onProcess,
  onCancel,
  canProcess,
  className
}: ProcessingStatusProps) {
  const [showLogs, setShowLogs] = useState(false);

  const getStatusIcon = () => {
    if (status.isProcessing) {
      return <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />;
    }
    if (status.progress === 100) {
      return <CheckCircle className="w-5 h-5 text-emerald-400" />;
    }
    return <Activity className="w-5 h-5 text-muted-foreground" />;
  };

  const getStatusColor = () => {
    if (status.isProcessing) return 'from-blue-500 to-cyan-600';
    if (status.progress === 100) return 'from-emerald-500 to-green-600';
    return 'from-gray-500 to-gray-600';
  };

  const getStatusBadge = () => {
    if (status.isProcessing) {
      return (
        <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30 animate-pulse">
          Processing...
        </Badge>
      );
    }
    if (status.progress === 100) {
      return (
        <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
          Complete
        </Badge>
      );
    }
    return (
      <Badge className="bg-gray-500/20 text-gray-400 border-gray-500/30">
        Ready
      </Badge>
    );
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Status Card */}
      <Card className="glass-card border-glass-white/20 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className={cn(
            "w-10 h-10 rounded-lg bg-gradient-to-br flex items-center justify-center",
            getStatusColor()
          )}>
            {getStatusIcon()}
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-3">
              <h3 className="text-lg font-semibold text-white">Processing Status</h3>
              {getStatusBadge()}
            </div>
            <p className="text-sm text-muted-foreground">Monitor video enhancement progress</p>
          </div>
        </div>

        {/* Progress Section */}
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-white">
                {status.stage || 'Waiting to start...'}
              </span>
              <span className="text-sm text-muted-foreground">
                {Math.round(status.progress)}%
              </span>
            </div>
            
            <Progress 
              value={status.progress} 
              className="h-2 glass-panel border-glass-white/20"
            />
            
            {status.details && (
              <p className="text-xs text-muted-foreground">
                {status.details}
              </p>
            )}
          </div>

          {/* Control Buttons */}
          <div className="flex space-x-3 pt-4">
            {!status.isProcessing ? (
              <Button
                onClick={onProcess}
                disabled={!canProcess}
                className="flex-1 bg-gradient-to-r from-accent-purple to-accent-teal hover:from-accent-purple/80 hover:to-accent-teal/80 text-white border-0 glow-hover"
              >
                <Play className="w-4 h-4 mr-2" />
                Start Processing
              </Button>
            ) : (
              <Button
                onClick={onCancel}
                variant="destructive"
                className="flex-1 bg-red-500/20 hover:bg-red-500/30 border-red-500/50 text-red-400"
              >
                <Square className="w-4 h-4 mr-2" />
                Cancel Processing
              </Button>
            )}
            
            <Button
              variant="outline"
              onClick={() => setShowLogs(!showLogs)}
              className="glass-panel border-glass-white/20 bg-glass-white/10 hover:bg-glass-white/20 text-white"
            >
              <Terminal className="w-4 h-4 mr-2" />
              Logs
              {showLogs ? <ChevronUp className="w-4 h-4 ml-2" /> : <ChevronDown className="w-4 h-4 ml-2" />}
            </Button>
          </div>
        </div>
      </Card>

      {/* Logs Panel */}
      {showLogs && (
        <Card className="glass-card border-glass-white/20 p-0 overflow-hidden">
          <div className="p-4 border-b border-glass-white/20">
            <div className="flex items-center space-x-3">
              <Terminal className="w-5 h-5 text-muted-foreground" />
              <h4 className="font-medium text-white">Processing Logs</h4>
              <Badge variant="secondary" className="bg-glass-white/10 text-muted-foreground">
                {logs.length} entries
              </Badge>
            </div>
          </div>
          
          <div className="p-4">
            <div className="glass-panel rounded-lg p-4 max-h-64 overflow-y-auto custom-scrollbar bg-black/20">
              {logs.length > 0 ? (
                <div className="space-y-1 font-mono text-sm">
                  {logs.map((log, index) => (
                    <div 
                      key={index} 
                      className={cn(
                        "text-gray-300 py-1 px-2 rounded transition-colors",
                        index === logs.length - 1 && "bg-blue-500/10 text-blue-300"
                      )}
                    >
                      <span className="text-gray-500 mr-2">
                        [{new Date().toLocaleTimeString()}]
                      </span>
                      {log}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-8">
                  <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No logs available</p>
                  <p className="text-xs">Logs will appear here during processing</p>
                </div>
              )}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
} 