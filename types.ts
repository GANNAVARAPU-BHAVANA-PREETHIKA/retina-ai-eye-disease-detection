export enum DiseaseType {
  NORMAL = 'Normal',
  CNV = 'CNV',
  DME = 'DME',
  DRUSEN = 'Drusen'
}

export enum RiskCategory {
  LOW = 'Low',
  MEDIUM = 'Medium',
  HIGH = 'High',
  NONE = 'None'
}

export enum EstimatedStage {
  EARLY = 'Early',
  INTERMEDIATE = 'Intermediate',
  ADVANCED = 'Advanced',
  NONE = 'None'
}

export interface EnsembleMember {
  modelName: string;
  prediction: DiseaseType;
  confidence: number;
  weight: number; // The influence this model had on the final result (0-1)
}

export interface AnalysisResult {
  detectedDisease: DiseaseType;
  riskCategory: RiskCategory;
  estimatedStage: EstimatedStage;
  confidence: number;
  reasoning: string;
  
  // Unified Ensemble Fields
  isTrained: boolean; // Indicates if the local models contributed
  primaryModel: string; // Will likely say "Hybrid Ensemble"
  
  // Detailed breakdown for transparency
  ensembleBreakdown?: EnsembleMember[];
}

export interface PerformanceMetric {
  metric: string;
  cnn: number;
  svm: number;
}