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
import { KnowledgeLibrary } from './pages/KnowledgeLibrary';
import { EpisodeModal } from './components/EpisodeModal';
import { api } from './services/api';
import { Universe, Episode } from './types';
import { Sparkles, CheckCircle2, AlertCircle, X } from 'lucide-react';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('dashboard');
  const [isGenerating, setIsGenerating] = useState(false);
  const [universes, setUniverses] = useState<Universe[]>([]);
  const [activeUniverseId, setActiveUniverseId] = useState<string>(() => localStorage.getItem('activeUniverseId') || '');
  const [sceneDuration, setSceneDuration] = useState<number>(8);
  const [refreshTrigger, setRefreshTrigger] = useState<number>(0);
  const [selectedEpForModal, setSelectedEpForModal] = useState<Episode | null>(null);

  // Toast Notifications
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'info' | 'error' } | null>(null);

  const showToast = (message: string, type: 'success' | 'info' | 'error' = 'success') => {
    setToast({ message, type });
    setTimeout(() => {
      setToast(null);
    }, 4000);
  };

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
    } else {
      setActiveUniverseId('');
      localStorage.removeItem('activeUniverseId');
    }
  };

  const handleSelectUniverse = (id: string) => {
    setActiveUniverseId(id);
    localStorage.setItem('activeUniverseId', id);
    setRefreshTrigger(prev => prev + 1);
  };

  const handleGenerateEpisode = async (customSec?: number, customPrompt?: string) => {
    if (!activeUniverseId) {
      showToast('No active universe. Please create a story universe first.', 'error');
      return;
    }
    setIsGenerating(true);
    showToast('Autonomous Engine: Writing Screenplay & Rendering Video Clips...', 'info');
    
    // Read RAG settings if they were set in Dashboard
    const ragEnabled = localStorage.getItem('rag_reference_enabled') === 'true';
    const refId = ragEnabled ? (localStorage.getItem('rag_reference_id') || undefined) : undefined;
    const refInfluence = ragEnabled ? (localStorage.getItem('rag_reference_influence') || undefined) : undefined;
    
    try {
      const newEp = await api.generateEpisode(
        activeUniverseId, 
        customPrompt || 'Uncover hidden core protocol secret', 
        customSec || sceneDuration,
        refId,
        refInfluence
      );
      
      await loadUniverses();
      setRefreshTrigger(prev => prev + 1);
      
      showToast(`✨ Generated ${newEp.title}! Click feed card to watch.`, 'success');
      setSelectedEpForModal(newEp);
    } catch (err: any) {
      console.error(err);
      showToast(`Generation Error: ${err?.message || 'Failed to generate episode'}`, 'error');
    } finally {
      setIsGenerating(false);
      // Removed the cleanup of local storage so RAG settings persist
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
            refreshTrigger={refreshTrigger}
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
            onUniversesChanged={() => {
              loadUniverses();
              setRefreshTrigger(prev => prev + 1);
            }}
          />
        );
      case 'characters':
        return <CharacterManager activeUniverseId={activeUniverseId} />;
      case 'timeline':
        return <Timeline activeUniverseId={activeUniverseId} />;
      case 'episodes':
        return <EpisodeLibrary activeUniverseId={activeUniverseId} refreshTrigger={refreshTrigger} />;
      case 'memory':
        return <MemoryExplorer activeUniverseId={activeUniverseId} />;
      case 'knowledge':
        return <KnowledgeLibrary />;
      case 'assets':
        return <AssetLibrary />;
      case 'videos':
        return <VideoLibrary activeUniverseId={activeUniverseId} refreshTrigger={refreshTrigger} />;
      case 'render_queue':
        return <RenderQueue isGenerating={isGenerating} refreshTrigger={refreshTrigger} />;
      case 'scheduler':
        return <SchedulerPage onTriggerGenerate={handleGenerateEpisode} refreshTrigger={refreshTrigger} />;
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
            refreshTrigger={refreshTrigger}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-black">
      {/* Toast Feedback Notification Banner */}
      {toast && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center space-x-3 px-5 py-3.5 rounded-2xl bg-slate-900/95 border border-cyan-500/80 shadow-2xl backdrop-blur-xl animate-fadeIn">
          {toast.type === 'success' && <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />}
          {toast.type === 'info' && <Sparkles className="w-5 h-5 text-cyan-400 animate-spin shrink-0" />}
          {toast.type === 'error' && <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />}
          <span className="text-xs font-mono font-bold text-white tracking-wide">{toast.message}</span>
          <button onClick={() => setToast(null)} className="p-1 rounded-lg text-slate-400 hover:text-white">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Episode Detail Inspector Modal */}
      {selectedEpForModal && (
        <EpisodeModal episode={selectedEpForModal} onClose={() => setSelectedEpForModal(null)} />
      )}

      {/* Top Navbar with Story Universe Switcher */}
      <Navbar
        onGenerateClick={handleGenerateEpisode}
        isGenerating={isGenerating}
        universes={universes}
        activeUniverseId={activeUniverseId}
        onSelectUniverse={handleSelectUniverse}
        sceneDuration={sceneDuration}
        onSceneDurationChange={setSceneDuration}
        onLogoClick={() => setActivePage('dashboard')}
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
