import React from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  glow?: boolean;
  onClick?: () => void;
}

export const GlassCard: React.FC<GlassCardProps> = ({ children, className = '', glow = false, onClick }) => {
  return (
    <div
      onClick={onClick}
      className={`rounded-2xl p-6 transition-all duration-300 ${
        glow ? 'glass-panel-glow' : 'glass-panel hover:border-slate-700/80'
      } ${className}`}
    >
      {children}
    </div>
  );
};
