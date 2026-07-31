import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar, PageId } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { UniverseManager } from './pages/UniverseManager';
import { CharacterManager } from './pages/CharacterManager';
import { Timeline } from './pages/Timeline';
import { EpisodeLibrary } from './pages/EpisodeLibrary';
import { MemoryExplorer } from './pages/MemoryExplorer';
import { AssetLibrary } from './pages/AssetLibrary';
import { VideoLibrary } from './pages/VideoLibrary';
import { RenderQueue } from './pages/RenderQueue';
import { SchedulerPage } from './pages/SchedulerPage';
import { Analytics } from './pages/Analytics';
import { SettingsPage } from './pages/SettingsPage';
import { api } from './services/api';
import { Universe } from './types';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('dashboard');
  const [isGenerating, setIsGenerating] = useState(false);
  const [universes, setUniverses] = useState<Universe[]>([]);
  const [activeUniverseId, setActiveUniverseId] = useState<string>(() => localStorage.getItem('activeUniverseId') || '');
  const [sceneDuration, setSceneDuration] = useState<number>(8);

  useEffect(() => {
    loadUniverses();
  }, []);

  const loadUniverses = async () => {
    const data = await api.getUniverses();
    setUniverses(data);
    if (data.length > 0) {
      const savedId = localStorage.getItem('activeUniverseId');
      const exists = data.some((u) => u.id === savedId);
      const chosenId = exists ? savedId! : data[0].id;
      setActiveUniverseId(chosenId);
      localStorage.setItem('activeUniverseId', chosenId);
    }
  };

  const handleSelectUniverse = (id: string) => {
    setActiveUniverseId(id);
    localStorage.setItem('activeUniverseId', id);
  };

  const handleGenerateEpisode = async (customSec?: number) => {
    setIsGenerating(true);
    try {
      const uId = activeUniverseId || (universes.length > 0 ? universes[0].id : 'u-cyber-99');
      await api.generateEpisode(uId, 'Uncover hidden core protocol secret', customSec || sceneDuration);
    } catch (err) {
      console.error(err);
    } finally {
      setIsGenerating(false);
    }
  };

  const renderCurrentPage = () => {
    switch (activePage) {
      case 'dashboard':
        return (
          <Dashboard
            onNavigatePage={setActivePage}
            onGenerateClick={handleGenerateEpisode}
            isGenerating={isGenerating}
            activeUniverseId={activeUniverseId}
          />
        );
      case 'universes':
        return (
          <UniverseManager
            activeUniverseId={activeUniverseId}
            onSelectUniverse={(id) => {
              handleSelectUniverse(id);
              loadUniverses();
            }}
          />
        );
      case 'characters':
        return <CharacterManager activeUniverseId={activeUniverseId} />;
      case 'timeline':
        return <Timeline activeUniverseId={activeUniverseId} />;
      case 'episodes':
        return <EpisodeLibrary activeUniverseId={activeUniverseId} />;
      case 'memory':
        return <MemoryExplorer activeUniverseId={activeUniverseId} />;
      case 'assets':
        return <AssetLibrary />;
      case 'videos':
        return <VideoLibrary activeUniverseId={activeUniverseId} />;
      case 'render_queue':
        return <RenderQueue />;
      case 'scheduler':
        return <SchedulerPage />;
      case 'analytics':
        return <Analytics />;
      case 'settings':
        return <SettingsPage />;
      default:
        return (
          <Dashboard
            onNavigatePage={setActivePage}
            onGenerateClick={handleGenerateEpisode}
            isGenerating={isGenerating}
            activeUniverseId={activeUniverseId}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-black">
      {/* Top Navbar with Story Universe Switcher */}
      <Navbar
        onGenerateClick={handleGenerateEpisode}
        isGenerating={isGenerating}
        universes={universes}
        activeUniverseId={activeUniverseId}
        onSelectUniverse={handleSelectUniverse}
        sceneDuration={sceneDuration}
        onSceneDurationChange={setSceneDuration}
      />

      {/* Main Layout Body */}
      <div className="flex flex-1">
        {/* Sidebar Navigation */}
        <Sidebar activePage={activePage} onSelectPage={setActivePage} />

        {/* Page Content Viewport */}
        <main className="flex-1 p-8 overflow-y-auto max-w-7xl mx-auto">
          {renderCurrentPage()}
        </main>
      </div>
    </div>
  );
};

export default App;
