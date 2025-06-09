import React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Settings, Key, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface HeaderProps {
  hasApiKey: boolean;
  onOpenApiKeyModal: () => void;
  className?: string;
}

export function Header({ hasApiKey, onOpenApiKeyModal, className }: HeaderProps) {
  return (
    <header className={cn(
      "glass-panel sticky top-0 z-50 border-b border-glass-white/10 backdrop-blur-xl",
      "bg-gradient-to-r from-black/20 via-black/10 to-black/20",
      className
    )}>
      <div className="flex items-center justify-between p-6">
        {/* Logo and branding */}
        <div className="flex items-center space-x-4">
          <div className="relative">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-purple to-accent-teal flex items-center justify-center shadow-lg glow">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div className="absolute -inset-0.5 bg-gradient-to-br from-accent-purple to-accent-teal rounded-xl blur opacity-30 animate-pulse" />
          </div>
          <div>
            <h1 className="text-2xl font-bold gradient-text">
              AI Cameraman
            </h1>
            <p className="text-sm text-muted-foreground">
              Intelligent Video Enhancement
            </p>
          </div>
        </div>
        
        {/* Action buttons */}
        <div className="flex items-center space-x-3">
          <Badge 
            variant={hasApiKey ? "default" : "destructive"} 
            className={cn(
              "px-3 py-1.5 text-sm font-medium",
              hasApiKey ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "bg-red-500/20 text-red-400 border-red-500/30"
            )}
          >
            {hasApiKey ? '● Connected' : '● Disconnected'}
          </Badge>
          
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenApiKeyModal}
            className={cn(
              "glass-panel border-glass-white/20 bg-glass-white/10 hover:bg-glass-white/20 text-white",
              "transition-all duration-200 hover:glow-hover",
              !hasApiKey && "border-red-500/30 bg-red-500/10 hover:bg-red-500/20"
            )}
          >
            <Key className="w-4 h-4 mr-2" />
            API Key
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            className="glass-panel border-glass-white/20 bg-glass-white/10 hover:bg-glass-white/20 text-white transition-all duration-200 hover:glow-hover"
          >
            <Settings className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </header>
  );
} 