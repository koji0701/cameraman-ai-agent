import React from 'react';
import { cn } from '@/lib/utils';

interface AppShellProps {
  children: React.ReactNode;
  className?: string;
}

export function AppShell({ children, className }: AppShellProps) {
  return (
    <div className={cn(
      "min-h-screen bg-dark-gradient relative overflow-hidden",
      className
    )}>
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 rounded-full bg-accent-purple/10 blur-3xl animate-float" />
        <div className="absolute top-1/2 -left-40 w-96 h-96 rounded-full bg-accent-teal/10 blur-3xl animate-float" style={{ animationDelay: '1s' }} />
        <div className="absolute -bottom-40 right-1/3 w-72 h-72 rounded-full bg-accent-pink/10 blur-3xl animate-float" style={{ animationDelay: '2s' }} />
      </div>
      
      {/* Main content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
} 