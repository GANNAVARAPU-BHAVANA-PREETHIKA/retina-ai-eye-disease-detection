import React, { useCallback, useState } from 'react';
import { AnalysisResult } from '../types';
import { analyzeImage } from '../services/cnnService';

interface ImageUploadProps {
  onAnalysisStart: () => void;
  onAnalysisComplete: (result: AnalysisResult, imageSrc: string) => void;
  onAnalysisError: (error: string) => void;
  isAnalyzing: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ 
  onAnalysisStart, 
  onAnalysisComplete, 
  onAnalysisError,
  isAnalyzing 
}) => {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const processFile = async (file: File) => {
    if (!file || !file.type.startsWith('image/')) {
      onAnalysisError("Please upload a valid image file (JPG, PNG).");
      return;
    }

    onAnalysisStart();

    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = async () => {
      const base64String = reader.result as string;
      const base64Content = base64String.split(',')[1];
      
      try {
        const result = await analyzeImage(base64Content, file.type);
        onAnalysisComplete(result, base64String);
      } catch (error: any) {
        console.error(error);
        // Show the actual error message (e.g., Network Error, Model failed to load)
        onAnalysisError(error.message || "Failed to analyze image. Check console for details.");
      }
    };
    reader.onerror = () => {
      onAnalysisError("Error reading file.");
    };
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto mb-8">
      <div 
        className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-300 ease-in-out text-center cursor-pointer ${
          isAnalyzing ? 'opacity-50 pointer-events-none border-gray-300' :
          dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-upload')?.click()}
      >
        <input 
          id="file-upload" 
          type="file" 
          className="hidden" 
          accept="image/png, image/jpeg" 
          onChange={handleChange} 
        />
        
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="p-4 bg-white rounded-full shadow-sm">
            <svg className="w-8 h-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <div>
            <p className="text-lg font-medium text-gray-700">
              {isAnalyzing ? "Analyzing Retina Scan..." : "Upload OCT Image"}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Drag & drop or click to browse (JPG, PNG)
            </p>
            <p className="text-xs text-blue-400 mt-2">
              Processed locally using Transfer Learning (TF.js)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;