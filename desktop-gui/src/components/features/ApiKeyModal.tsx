import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Key, Eye, EyeOff, CheckCircle, AlertCircle, ExternalLink, Shield } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (apiKey: string) => Promise<void>;
  currentApiKey: string;
  className?: string;
}

export function ApiKeyModal({ isOpen, onClose, onSave, currentApiKey, className }: ApiKeyModalProps) {
  const [apiKey, setApiKey] = useState(currentApiKey);
  const [showKey, setShowKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [status, setStatus] = useState<{ type: 'success' | 'error' | null; message: string }>({
    type: null,
    message: ''
  });

  useEffect(() => {
    if (isOpen) {
      setApiKey(currentApiKey);
      setStatus({ type: null, message: '' });
      setShowKey(false);
    }
  }, [isOpen, currentApiKey]);

  const validateApiKey = (key: string) => {
    if (!key.trim()) {
      return 'API key is required';
    }
    if (key.length < 20) {
      return 'API key appears to be too short';
    }
    if (!key.startsWith('AIza')) {
      return 'Invalid Google API key format';
    }
    return null;
  };

  const handleSave = async () => {
    const validation = validateApiKey(apiKey);
    if (validation) {
      setStatus({ type: 'error', message: validation });
      return;
    }

    setIsSaving(true);
    try {
      await onSave(apiKey);
      setStatus({ type: 'success', message: 'API key saved successfully!' });
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (error) {
      setStatus({ type: 'error', message: `Failed to save API key: ${error}` });
    } finally {
      setIsSaving(false);
    }
  };

  const handleTestKey = () => {
    // TODO: Implement API key testing
    setStatus({ type: 'success', message: 'API key test not implemented yet' });
  };

  const maskedKey = (key: string) => {
    if (key.length <= 8) return key;
    return key.substring(0, 4) + '•'.repeat(key.length - 8) + key.substring(key.length - 4);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className={cn(
        "glass-strong border-glass-white/30 bg-dark-300/90 backdrop-blur-xl text-white max-w-lg",
        className
      )}>
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-3 text-xl">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-purple to-accent-teal flex items-center justify-center">
              <Key className="w-5 h-5 text-white" />
            </div>
            <span>API Key Configuration</span>
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Configure your Google Gemini API key for AI video processing
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 pt-4">
          {/* Current Status */}
          <div className="glass-panel rounded-lg p-4 border-glass-white/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Shield className="w-5 h-5 text-muted-foreground" />
                <span className="text-sm font-medium">Current Status:</span>
              </div>
              <Badge className={cn(
                currentApiKey 
                  ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                  : "bg-red-500/20 text-red-400 border-red-500/30"
              )}>
                {currentApiKey ? 'Configured' : 'Not Set'}
              </Badge>
            </div>
            {currentApiKey && (
              <div className="mt-2 text-xs text-muted-foreground">
                Key: {maskedKey(currentApiKey)}
              </div>
            )}
          </div>

          {/* API Key Input */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-white">
              Google Gemini API Key
            </label>
            <div className="relative">
              <Input
                type={showKey ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your Google Gemini API key..."
                className="glass-panel border-glass-white/20 bg-glass-white/5 text-white placeholder:text-muted-foreground focus:border-accent-purple/50 focus:ring-accent-purple/30 pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-1 top-1/2 -translate-y-1/2 h-8 w-8 p-0 hover:bg-glass-white/10"
              >
                {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </Button>
            </div>
          </div>

          {/* Status Message */}
          {status.type && (
            <div className={cn(
              "glass-panel rounded-lg p-3 border",
              status.type === 'success' 
                ? "bg-emerald-500/10 border-emerald-500/30" 
                : "bg-red-500/10 border-red-500/30"
            )}>
              <div className="flex items-center space-x-2">
                {status.type === 'success' ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-red-400" />
                )}
                <span className={cn(
                  "text-sm",
                  status.type === 'success' ? "text-emerald-400" : "text-red-400"
                )}>
                  {status.message}
                </span>
              </div>
            </div>
          )}

          {/* Help Section */}
          <div className="glass-panel rounded-lg p-4 bg-blue-500/5 border-blue-500/20">
            <div className="text-sm text-blue-400">
              <p className="font-medium mb-2">How to get your API key:</p>
              <ol className="list-decimal list-inside space-y-1 text-xs">
                <li>Visit the Google AI Studio</li>
                <li>Create a new project or select existing one</li>
                <li>Generate a new API key</li>
                <li>Copy and paste it here</li>
              </ol>
              <Button
                variant="link"
                size="sm"
                className="p-0 h-auto text-blue-400 hover:text-blue-300 mt-2"
                onClick={() => window.open('https://aistudio.google.com/app/apikey', '_blank')}
              >
                <ExternalLink className="w-3 h-3 mr-1" />
                Open Google AI Studio
              </Button>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-3 pt-4">
            <Button
              variant="outline"
              onClick={onClose}
              className="glass-panel border-glass-white/20 bg-glass-white/10 hover:bg-glass-white/20 text-white flex-1"
            >
              Cancel
            </Button>
            
            {apiKey && (
              <Button
                variant="outline"
                onClick={handleTestKey}
                className="glass-panel border-blue-500/30 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400"
              >
                Test
              </Button>
            )}
            
            <Button
              onClick={handleSave}
              disabled={isSaving || !apiKey.trim()}
              className="bg-gradient-to-r from-accent-purple to-accent-teal hover:from-accent-purple/80 hover:to-accent-teal/80 text-white border-0 flex-1"
            >
              {isSaving ? 'Saving...' : 'Save Key'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
} 