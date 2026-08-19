import React from 'react';
import {
  LayoutDashboard, Globe, Users, Clock, Library, Brain, BookOpen,
  FileVideo, Video, ListOrdered, Calendar, BarChart3, Settings
} from 'lucide-react';

export type PageId =
  | 'dashboard'
  | 'universes'
  | 'characters'
  | 'timeline'
  | 'episodes'
  | 'memory'
  | 'knowledge'
  | 'assets'
  | 'videos'
  | 'render_queue'
  | 'scheduler'
  | 'analytics'
  | 'settings';

interface SidebarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activePage, onSelectPage }) => {
  const menuItems: Array<{ id: PageId; label: string; icon: React.FC<{ className?: string }> }> = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'universes', label: 'Universe Manager', icon: Globe },
    { id: 'characters', label: 'Character Manager', icon: Users },
    { id: 'timeline', label: 'Timeline', icon: Clock },
    { id: 'episodes', label: 'Episode Library', icon: Library },
    { id: 'memory', label: 'Memory Explorer', icon: Brain },
    { id: 'knowledge', label: 'Knowledge Library', icon: BookOpen },
    { id: 'assets', label: 'Asset Library', icon: FileVideo },
    { id: 'videos', label: 'Video Library', icon: Video },
    { id: 'render_queue', label: 'Render Queue', icon: ListOrdered },
    { id: 'scheduler', label: 'Scheduler', icon: Calendar },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-background flex flex-col justify-between py-6 sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto border-r border-neutral-border">
      <div className="space-y-1 px-4">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectPage(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-2xl font-medium text-sm transition-all duration-200 ${
                isActive
                  ? 'bg-accent-primary text-white shadow-sm'
                  : 'text-text-primary hover:bg-neutral-soft hover:text-accent-primary'
              }`}
            >
              <Icon className={`w-5 h-5 ${isActive ? 'text-white' : 'text-accent-primary'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      <div className="px-6 pt-4 text-xs text-text-secondary space-y-1">
        <div className="flex items-center justify-between font-mono text-[11px]">
          <span>SYSTEM LOOP:</span>
          <span className="text-accent-primary font-semibold flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-accent-primary" /> ACTIVE
          </span>
        </div>
        <div className="text-[10px] text-text-secondary">Open-Source AI Autonomous Platform<br/>v1.0.0</div>
      </div>
    </aside>
  );
};
