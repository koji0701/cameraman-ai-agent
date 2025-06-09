import React from 'react';
import { cn } from '@/lib/utils';

interface AppShellProps {
  children: React.ReactNode;
  className?: string;
}

export function AppShell({ children, className }: AppShellProps) {
  return (
    <div className={cn(
      "min-h-screen relative overflow-hidden dark window-content",
      className
    )}>
      {/* Drag Region for window dragging */}
      <div 
        className="absolute top-0 left-0 right-0 h-8 z-50"
        style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
      />
      
      {/* Enhanced Glassmorphism Background */}
      <div className="absolute inset-0 glass-blur opacity-40" />
      
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Primary floating orbs */}
        <div className="absolute -top-40 -right-40 w-80 h-80 rounded-full bg-gradient-to-br from-accent-purple/15 to-accent-purple/5 blur-3xl animate-float" />
        <div className="absolute top-1/2 -left-40 w-96 h-96 rounded-full bg-gradient-to-br from-accent-teal/15 to-accent-teal/5 blur-3xl animate-float" style={{ animationDelay: '1s' }} />
        <div className="absolute -bottom-40 right-1/3 w-72 h-72 rounded-full bg-gradient-to-br from-accent-pink/15 to-accent-pink/5 blur-3xl animate-float" style={{ animationDelay: '2s' }} />
        
        {/* Secondary ambient orbs */}
        <div className="absolute top-1/4 right-1/4 w-60 h-60 rounded-full bg-gradient-to-br from-blue-500/8 to-purple-600/3 blur-2xl animate-breathe" style={{ animationDelay: '3s' }} />
        <div className="absolute bottom-1/4 left-1/4 w-48 h-48 rounded-full bg-gradient-to-br from-cyan-400/8 to-teal-500/3 blur-2xl animate-breathe" style={{ animationDelay: '4s' }} />
        
        {/* Subtle grain texture overlay */}
        <div className="absolute inset-0 opacity-[0.01] mix-blend-overlay" 
             style={{
               backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
             }} 
        />
      </div>
      
      {/* Main content with enhanced glass container */}
      <div className="relative z-10 min-h-screen pt-8">
        {children}
      </div>
    </div>
  );
} 