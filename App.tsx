import React, { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import AnalysisResults from './components/AnalysisResults';
import PerformanceCharts from './components/PerformanceCharts';
import TrainingDashboard from './components/TrainingDashboard';
import { AnalysisResult, DiseaseType } from './types';
import { learnFromCorrection, analyzeImage, initLocalModels } from './services/cnnService';

const App: React.FC = () => {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'analysis' | 'training'>('analysis');

  // Load saved models on mount
  useEffect(() => {
     const loadModels = async () => {
         const success = await initLocalModels();
         if (success) {
             console.log("Pre-trained models loaded successfully.");
         }
     };
     loadModels();
  }, []);

  const handleAnalysisStart = () => {
    setIsAnalyzing(true);
    setError(null);
  };

  const handleAnalysisComplete = (result: AnalysisResult, src: string) => {
    setResult(result);
    setImageSrc(src);
    setIsAnalyzing(false);
  };

  const handleAnalysisError = (msg: string) => {
    setError(msg);
    setIsAnalyzing(false);
  };

  const handleReset = () => {
    setResult(null);
    setImageSrc(null);
    setError(null);
  };
  
  const handleCorrection = async (correctLabel: DiseaseType) => {
      if (!imageSrc) return;
      try {
          // 1. Update weights
          await learnFromCorrection(imageSrc, correctLabel);
          
          // 2. Re-analyze to show updated confidence
          const base64 = imageSrc.split(',')[1];
          const mime = imageSrc.split(':')[1].split(';')[0];
          const newResult = await analyzeImage(base64, mime);
          
          setResult(newResult);
      } catch (err) {
          console.error("Correction failed:", err);
          setError("Failed to update model.");
      }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans selection:bg-blue-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('analysis')}>
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-xl shadow-blue-200 shadow-lg">R</div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-blue-500">
              RetinaAI
            </h1>
          </div>
          <nav className="flex space-x-2 md:space-x-4">
            <button 
                onClick={() => setActiveTab('analysis')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === 'analysis' ? 'bg-blue-50 text-blue-700' : 'text-gray-500 hover:text-gray-900'}`}
            >
                Analysis
            </button>
            <button 
                onClick={() => setActiveTab('training')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === 'training' ? 'bg-blue-50 text-blue-700' : 'text-gray-500 hover:text-gray-900'}`}
            >
                Training Studio
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        
        {activeTab === 'training' ? (
            <TrainingDashboard />
        ) : (
            <>
                {/* Hero / Intro */}
                {!result && (
                  <div className="text-center mb-16 max-w-3xl mx-auto animate-fade-in">
                    <h2 className="text-4xl sm:text-5xl font-extrabold text-gray-900 tracking-tight mb-6">
                      Smart Detection of <br/>
                      <span className="text-blue-600">Macular Disorders</span>
                    </h2>
                    <p className="text-lg text-gray-600 mb-8 leading-relaxed">
                      A robust hybrid system utilizing <strong>Transfer Learning</strong> for deep feature extraction, followed by a dual-head classification pipeline (CNN + SVM).
                    </p>
                    <div className="flex flex-wrap justify-center gap-2 mb-8">
                      <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-xs font-semibold uppercase tracking-wide">VGG16 / ResNet50</span>
                      <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-semibold uppercase tracking-wide">SVM Classifier</span>
                      <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-semibold uppercase tracking-wide">Ensemble Logic</span>
                    </div>
                  </div>
                )}

                {/* Interaction Area */}
                <div className="transition-all duration-500 ease-in-out">
                  {!result ? (
                    <>
                      <ImageUpload 
                        onAnalysisStart={handleAnalysisStart}
                        onAnalysisComplete={handleAnalysisComplete}
                        onAnalysisError={handleAnalysisError}
                        isAnalyzing={isAnalyzing}
                      />
                      
                      {isAnalyzing && (
                        <div className="flex flex-col items-center justify-center mt-8 space-y-4">
                          <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
                          <p className="text-gray-500 font-medium animate-pulse">Running Hybrid Model Inference...</p>
                          <p className="text-xs text-gray-400">Loading Backbone → Extracting Features → Running Classifiers</p>
                        </div>
                      )}

                      {error && (
                        <div className="max-w-2xl mx-auto mt-4 p-4 bg-red-50 text-red-700 rounded-lg border border-red-200 flex items-center">
                           <svg className="w-5 h-5 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                             <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                           </svg>
                           {error}
                        </div>
                      )}
                    </>
                  ) : (
                    <AnalysisResults 
                      result={result} 
                      imageSrc={imageSrc!} 
                      reset={handleReset} 
                      onCorrectDiagnosis={handleCorrection}
                    />
                  )}
                </div>

                {/* Metrics Section - Always visible unless analyzing or has result */}
                {!isAnalyzing && !result && (
                  <PerformanceCharts />
                )}
            </>
        )}

      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12 py-12">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-gray-400 text-sm">
            © 2026 RetinaAI Research Project. Non-clinical use only. <br/>
            Powered by TensorFlow.js, VGG16 Backbone & SVM.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;