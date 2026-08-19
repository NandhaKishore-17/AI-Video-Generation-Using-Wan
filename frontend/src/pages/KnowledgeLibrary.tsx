import React, { useState, useEffect } from 'react';
import { BookOpen, Upload, Trash2, Loader2, BookType, BookAudio, FileText } from 'lucide-react';
import { api } from '../services/api';
import { KnowledgeDocument } from '../types';

export const KnowledgeLibrary: React.FC = () => {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    const data = await api.getKnowledgeList();
    setDocuments(data);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    
    // Only accept basic formats for now
    if (!file.name.endsWith('.pdf') && !file.name.endsWith('.txt') && !file.name.endsWith('.docx')) {
      alert("Unsupported file type. Please upload .pdf, .txt, or .docx");
      return;
    }

    setIsUploading(true);
    setUploadStatus(`Uploading and parsing ${file.name}...`);
    try {
      await api.uploadKnowledge(file);
      await loadDocuments();
    } catch (err) {
      console.error(err);
      alert("Failed to upload document.");
    } finally {
      setIsUploading(false);
      setUploadStatus('');
      // Reset input
      e.target.value = '';
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this reference document?")) return;
    try {
      await api.deleteKnowledge(id);
      setDocuments(documents.filter(d => d.id !== id));
    } catch (err) {
      console.error(err);
    }
  };

  const getFileIcon = (fileType: string, fileName: string) => {
    if (fileName.endsWith('.pdf')) return <BookType className="w-8 h-8 text-text-secondary" />;
    if (fileName.endsWith('.docx')) return <BookAudio className="w-8 h-8 text-blue-400" />;
    return <FileText className="w-8 h-8 text-text-secondary" />;
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold bg-accent-primary bg-clip-text text-transparent flex items-center gap-3">
            <BookOpen className="w-7 h-7 text-accent-primary" />
            Reference Knowledge Library
          </h1>
          <p className="text-text-secondary text-sm mt-1">Upload books and documents to inspire original stories.</p>
        </div>
        
        <div className="relative">
          <input
            type="file"
            id="knowledge-upload"
            className="hidden"
            accept=".pdf,.txt,.docx"
            onChange={handleFileUpload}
            disabled={isUploading}
          />
          <label
            htmlFor="knowledge-upload"
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
              isUploading 
                ? 'bg-neutral-soft text-text-secondary cursor-not-allowed'
                : 'bg-accent-light0/20 text-accent-primary hover:bg-accent-light0/30 border border-accent-primary cursor-pointer shadow-glow-emerald'
            }`}
          >
            {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span>{isUploading ? 'Processing...' : 'Upload Reference'}</span>
          </label>
        </div>
      </div>

      {isUploading && (
        <div className="bg-neutral-soft/50 border border-neutral-border p-4 rounded-xl flex items-center space-x-4">
          <Loader2 className="w-6 h-6 text-accent-primary animate-spin" />
          <div className="text-sm font-medium text-text-secondary">
            {uploadStatus}
            <div className="text-xs text-text-secondary mt-1">Extracting text, creating chunks, and generating embeddings...</div>
          </div>
        </div>
      )}

      {documents.length === 0 && !isUploading ? (
        <div className="text-center py-20 bg-neutral-soft/30 border border-neutral-border rounded-2xl border-dashed">
          <BookOpen className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-text-secondary">No Reference Documents</h3>
          <p className="text-text-secondary text-sm mt-1">Upload a book or reference material to use as RAG inspiration.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {documents.map((doc) => (
            <div key={doc.id} className="bg-neutral-soft/50 border border-neutral-border/50 rounded-xl p-5 hover:border-neutral-border transition-colors flex flex-col justify-between">
              <div className="flex items-start justify-between">
                <div className="flex space-x-3">
                  <div className="mt-1">
                    {getFileIcon(doc.file_type, doc.name)}
                  </div>
                  <div>
                    <h3 className="font-semibold text-text-secondary">{doc.name}</h3>
                    <div className="text-xs text-text-secondary font-mono mt-1">
                      {doc.chunk_count} concept chunks extracted
                    </div>
                  </div>
                </div>
                <button 
                  onClick={() => handleDelete(doc.id)}
                  className="text-text-secondary hover:text-text-secondary p-1 rounded transition-colors bg-neutral-soft hover:bg-neutral-soft"
                  title="Delete Document"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              <div className="mt-4 flex items-center justify-between text-xs border-t border-neutral-border pt-3">
                <span className={`px-2 py-1 rounded font-medium ${
                  doc.status === 'COMPLETED' ? 'bg-accent-light0/10 text-accent-primary' :
                  doc.status === 'FAILED' ? 'bg-neutral-soft text-text-secondary' :
                  'bg-yellow-500/10 text-yellow-400'
                }`}>
                  {doc.status}
                </span>
                <span className="text-text-secondary">
                  {new Date(doc.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
