import React from 'react';
import { Card } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Settings, Zap, Target, Sparkles, Info } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProcessingOptions {
  quality: 'low' | 'medium' | 'high' | 'ultra';
  disableStreaming: boolean;
}

interface ProcessingOptionsProps {
  options: ProcessingOptions;
  onOptionsChange: (key: keyof ProcessingOptions, value: any) => void;
  className?: string;
}

const qualityPresets = {
  low: {
    label: 'Fast',
    description: 'Quick processing, good quality',
    icon: Zap,
    color: 'from-green-500 to-emerald-600',
    badge: 'bg-green-500/20 text-green-400 border-green-500/30'
  },
  medium: {
    label: 'Balanced',
    description: 'Good balance of speed and quality',
    icon: Target,
    color: 'from-blue-500 to-cyan-600',
    badge: 'bg-blue-500/20 text-blue-400 border-blue-500/30'
  },
  high: {
    label: 'Quality',
    description: 'High quality, slower processing',
    icon: Sparkles,
    color: 'from-purple-500 to-pink-600',
    badge: 'bg-purple-500/20 text-purple-400 border-purple-500/30'
  },
  ultra: {
    label: 'Ultra',
    description: 'Maximum quality, longest processing',
    icon: Settings,
    color: 'from-orange-500 to-red-600',
    badge: 'bg-orange-500/20 text-orange-400 border-orange-500/30'
  }
};

export function ProcessingOptions({ options, onOptionsChange, className }: ProcessingOptionsProps) {
  const currentPreset = qualityPresets[options.quality];
  const Icon = currentPreset.icon;

  return (
    <Card className={cn("glass-card border-glass-white/20 p-6", className)}>
      <div className="flex items-center space-x-3 mb-6">
        <div className={cn(
          "w-10 h-10 rounded-lg bg-gradient-to-br flex items-center justify-center",
          currentPreset.color
        )}>
          <Icon className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">Processing Options</h3>
          <p className="text-sm text-muted-foreground">Configure video enhancement settings</p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Quality Preset */}
        <div className="space-y-3">
          <label className="text-sm font-medium text-white flex items-center space-x-2">
            <span>Quality Preset</span>
            <Badge className={currentPreset.badge}>
              {currentPreset.label}
            </Badge>
          </label>
          
          <Select 
            value={options.quality} 
            onValueChange={(value) => onOptionsChange('quality', value)}
          >
            <SelectTrigger className="glass-panel border-glass-white/20 bg-glass-white/5 text-white focus:border-accent-purple/50 focus:ring-accent-purple/30">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="glass-panel border-glass-white/20 bg-dark-300/95 backdrop-blur-xl">
              {Object.entries(qualityPresets).map(([key, preset]) => {
                const PresetIcon = preset.icon;
                return (
                  <SelectItem 
                    key={key} 
                    value={key}
                    className="text-white hover:bg-glass-white/10 focus:bg-glass-white/10"
                  >
                    <div className="flex items-center space-x-3">
                      <div className={cn(
                        "w-8 h-8 rounded-lg bg-gradient-to-br flex items-center justify-center",
                        preset.color
                      )}>
                        <PresetIcon className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <div className="font-medium">{preset.label}</div>
                        <div className="text-xs text-muted-foreground">{preset.description}</div>
                      </div>
                    </div>
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
          
          <div className="glass-panel rounded-lg p-3 bg-blue-500/5 border-blue-500/20">
            <div className="flex items-start space-x-2">
              <Info className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-blue-400">
                <strong>{currentPreset.label}:</strong> {currentPreset.description}
              </div>
            </div>
          </div>
        </div>

        {/* OpenCV Mode */}
        <div className="space-y-3">
          <label className="text-sm font-medium text-white">Processing Mode</label>
          
          <div className="glass-panel rounded-lg p-4 border-emerald-500/20 bg-emerald-500/5">
            <div className="flex items-center space-x-3">
              <Checkbox
                id="opencv-mode"
                checked={options.disableStreaming}
                onCheckedChange={(checked) => onOptionsChange('disableStreaming', checked)}
                className="border-emerald-500/50 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
              />
              <div className="flex-1">
                <label htmlFor="opencv-mode" className="text-sm font-medium text-emerald-400 cursor-pointer">
                  Use OpenCV Processing
                </label>
                <p className="text-xs text-emerald-300/80 mt-1">
                  Enables pure OpenCV processing for maximum compatibility and stability
                </p>
              </div>
              <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                Recommended
              </Badge>
            </div>
          </div>
        </div>

        {/* Processing Info */}
        <div className="glass-panel rounded-lg p-4 bg-gray-500/5 border-gray-500/20">
          <div className="text-sm text-gray-400">
            <div className="flex justify-between items-center mb-2">
              <span>Estimated processing time:</span>
              <span className="font-medium text-white">
                {options.quality === 'low' ? '2-5 min' : 
                 options.quality === 'medium' ? '5-10 min' :
                 options.quality === 'high' ? '10-20 min' : '20-40 min'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>Output quality:</span>
              <span className="font-medium text-white">
                {options.quality === 'low' ? 'Good' : 
                 options.quality === 'medium' ? 'Very Good' :
                 options.quality === 'high' ? 'Excellent' : 'Maximum'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
} 