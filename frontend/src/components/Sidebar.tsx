import React from 'react';
import {
  LayoutDashboard, Globe, Users, Clock, Library, Brain,
  FileVideo, Video, ListOrdered, Calendar, BarChart3, Settings
} from 'lucide-react';

export type PageId =
  | 'dashboard'
  | 'universes'
  | 'characters'
  | 'timeline'
  | 'episodes'
  | 'memory'
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
    { id: 'assets', label: 'Asset Library', icon: FileVideo },
    { id: 'videos', label: 'Video Library', icon: Video },
    { id: 'render_queue', label: 'Render Queue', icon: ListOrdered },
    { id: 'scheduler', label: 'Scheduler', icon: Calendar },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 border-r border-slate-800 bg-[#080B13] flex flex-col justify-between py-6 sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto">
      <div className="space-y-1 px-3">
        <div className="px-3 pb-2 text-[10px] font-mono tracking-widest text-slate-500 uppercase">
          Autonomous Control Hub
        </div>
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectPage(item.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
                isActive
                  ? 'bg-gradient-to-r from-cyan-950/80 to-purple-950/60 text-cyan-300 border border-cyan-700/60 shadow-glow-cyan'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      <div className="px-6 pt-4 border-t border-slate-800/80 text-xs text-slate-500 space-y-1">
        <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
          <span>SYSTEM LOOP:</span>
          <span className="text-emerald-400 font-semibold flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" /> ACTIVE
          </span>
        </div>
        <div className="text-[10px] text-slate-600">Open-Source AI Autonomous Platform</div>
      </div>
    </aside>
  );
};
