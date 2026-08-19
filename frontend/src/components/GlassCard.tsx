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
      className={`glass-panel p-6 ${className} ${onClick ? 'transition-transform hover:scale-[1.01] active:scale-[0.99] cursor-pointer' : ''} ${glow ? 'glass-panel-glow' : ''}`}
      onClick={onClick}
    >
      {children}
    </div>
  );
};
