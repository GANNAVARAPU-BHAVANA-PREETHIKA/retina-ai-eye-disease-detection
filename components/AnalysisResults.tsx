import React, { useState } from 'react';
import { AnalysisResult, DiseaseType, RiskCategory, EstimatedStage } from '../types';

interface AnalysisResultsProps {
  result: AnalysisResult;
  imageSrc: string;
  reset: () => void;
  onCorrectDiagnosis?: (correctLabel: DiseaseType) => void;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ result, imageSrc, reset, onCorrectDiagnosis }) => {
  const [isCorrecting, setIsCorrecting] = useState(false);
  const [correctionLoading, setCorrectionLoading] = useState(false);
  
  const getRiskColor = (risk: RiskCategory) => {
    switch (risk) {
      case RiskCategory.HIGH: return 'bg-red-100 text-red-800 border-red-200';
      case RiskCategory.MEDIUM: return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case RiskCategory.LOW: return 'bg-green-100 text-green-800 border-green-200';
      default: return 'hidden';
    }
  };

  const getStageColor = (stage: EstimatedStage) => {
    switch (stage) {
      case EstimatedStage.ADVANCED: return 'text-red-600';
      case EstimatedStage.INTERMEDIATE: return 'text-yellow-600';
      case EstimatedStage.EARLY: return 'text-green-600';
      default: return 'hidden';
    }
  };
  
  const handleCorrection = async (label: DiseaseType) => {
      if (!onCorrectDiagnosis) return;
      setCorrectionLoading(true);
      await onCorrectDiagnosis(label);
      setCorrectionLoading(false);
      setIsCorrecting(false);
  };

  // Extract Gemini confidence specifically - robustly finding 'Gemini' anywhere in model name
  let geminiConfidence = result.ensembleBreakdown?.find(m => m.modelName.includes('Gemini'))?.confidence ?? result.confidence;
  
  // UI Safety: Normalize 0-1 range to 0-100 just in case backend didn't catch it
  if (geminiConfidence > 0 && geminiConfidence <= 1) {
      geminiConfidence *= 100;
  }

  return (
    <div className="w-full max-w-6xl mx-auto animate-fade-in-up">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Left Col: Image & Pipeline Info */}
        <div className="flex flex-col space-y-6">
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Input OCT Image</h3>
            <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-gray-900 flex items-center justify-center">
              <img src={imageSrc} alt="Analyzed OCT" className="max-w-full max-h-full object-contain" />
              <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md text-white px-3 py-1 rounded-full text-xs font-mono">
                224 x 224 px
              </div>
            </div>
          </div>

          {/* Architecture Info */}
          <div className="bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-700 text-white">
             <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Ensemble Architecture</h3>
             <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-xs font-bold">1</div>
                  <div className="flex-1">
                    <p className="font-bold text-sm">Multimodal Foundation</p>
                    <p className="text-xs text-slate-400">Gemini 2.0 Flash (Global Context)</p>
                  </div>
                </div>
                <div className="h-4 border-l-2 border-slate-600 ml-4"></div>
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center text-xs font-bold">2</div>
                  <div className="flex-1">
                    <p className="font-bold text-sm">Deep Feature Extraction</p>
                    <p className="text-xs text-slate-400">ResNet50 / VGG16 (Local Backbone)</p>
                  </div>
                </div>
                <div className="h-4 border-l-2 border-slate-600 ml-4"></div>
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-xs font-bold">3</div>
                  <div className="flex-1">
                    <p className="font-bold text-sm">Dual Classifiers</p>
                    <p className="text-xs text-slate-400">Dense CNN Head + Linear SVM</p>
                  </div>
                </div>
             </div>
          </div>

          {/* Breakdown Table Removed as per request */}
        </div>

        {/* Right Col: Diagnosis */}
        <div className="flex flex-col space-y-6">
          
          {/* Main Card */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100 relative overflow-hidden">
            <div className={`absolute top-0 left-0 w-2 h-full ${result.detectedDisease === DiseaseType.NORMAL ? 'bg-green-500' : 'bg-red-500'}`}></div>
            
            <div className="flex justify-between items-start mb-6">
              <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-1">{result.detectedDisease}</h2>
                <p className="text-gray-500 text-sm flex items-center gap-1">
                    <svg className="w-4 h-4 text-blue-500" fill="currentColor" viewBox="0 0 20 20"><path d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z"/></svg>
                    {result.primaryModel}
                </p>
              </div>
              
              {result.riskCategory !== RiskCategory.NONE && (
                <div className={`px-4 py-2 rounded-lg border text-sm font-bold ${getRiskColor(result.riskCategory)}`}>
                  {result.riskCategory} Risk
                </div>
              )}
            </div>
            
            {/* Active Learning / Correction UI */}
            <div className="mb-6">
                {!isCorrecting ? (
                    <button 
                        onClick={() => setIsCorrecting(true)}
                        className="text-xs text-blue-600 hover:text-blue-800 underline font-medium flex items-center"
                    >
                        <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                        Incorrect Diagnosis? Train Ensemble.
                    </button>
                ) : (
                    <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 animate-fade-in">
                        {correctionLoading ? (
                            <div className="flex items-center space-x-2 text-blue-800">
                                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                                <span className="text-sm font-bold">Updating Local Weights...</span>
                            </div>
                        ) : (
                            <>
                                <p className="text-xs font-bold text-blue-800 uppercase mb-2">Select Correct Disease Type:</p>
                                <div className="grid grid-cols-2 gap-2">
                                    {Object.values(DiseaseType).map((type) => (
                                        <button
                                            key={type}
                                            onClick={() => handleCorrection(type)}
                                            className={`px-3 py-2 text-xs font-semibold rounded-md border transition-colors ${
                                                type === result.detectedDisease 
                                                ? 'bg-blue-200 border-blue-300 text-blue-800 cursor-default opacity-50' 
                                                : 'bg-white border-blue-200 text-blue-600 hover:bg-blue-100'
                                            }`}
                                            disabled={type === result.detectedDisease}
                                        >
                                            {type}
                                        </button>
                                    ))}
                                </div>
                                <button onClick={() => setIsCorrecting(false)} className="mt-2 text-xs text-gray-500 hover:text-gray-700 underline">Cancel</button>
                            </>
                        )}
                    </div>
                )}
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
              {result.estimatedStage !== EstimatedStage.NONE && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Estimated Stage</p>
                    <p className={`text-xl font-bold ${getStageColor(result.estimatedStage)}`}>
                      {result.estimatedStage}
                    </p>
                  </div>
              )}
              <div className={`bg-gray-50 rounded-lg p-4 ${result.estimatedStage === EstimatedStage.NONE ? 'col-span-2' : ''}`}>
                <p className="text-xs text-gray-500 uppercase font-semibold mb-1">Gemini Confidence</p>
                <div className="flex items-end">
                  <p className={`text-xl font-bold ${geminiConfidence === 0 ? 'text-red-500' : 'text-blue-600'}`}>
                    {geminiConfidence === 0 ? "Error / Unavailable" : `${geminiConfidence.toFixed(1)}%`}
                  </p>
                </div>
              </div>
            </div>

            <div className="mb-6">
              <p className="text-xs text-gray-500 uppercase font-semibold mb-2">Detailed Reasoning</p>
              <p className={`text-sm leading-relaxed p-4 rounded-lg border ${result.reasoning.includes('Error') ? 'bg-red-50 border-red-200 text-red-800' : 'bg-blue-50 border-blue-100 text-gray-700'}`}>
                "{result.reasoning}"
              </p>
            </div>
          </div>

          <button 
            onClick={reset}
            className="w-full py-4 bg-gray-900 hover:bg-black text-white rounded-xl font-semibold transition-all shadow-md flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span>Analyze Another Image</span>
          </button>

        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;