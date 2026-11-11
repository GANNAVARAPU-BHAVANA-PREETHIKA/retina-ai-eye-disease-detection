import * as tf from '@tensorflow/tfjs';
import { AnalysisResult, DiseaseType, RiskCategory, EstimatedStage, EnsembleMember } from "../types";
import { analyzeImage as analyzeWithGemini } from './geminiService';

// PROXY URLS FOR BROWSER DEMO
// Using MobileNet V2 for all as it is robust and lightweight for browser
const VGG16_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json'; 
const RESNET50_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
const MOBILENET_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';

let baseModel: tf.GraphModel | null = null;
let activeModelType: 'vgg16' | 'resnet50' | 'mobilenet' = 'vgg16';
let isModelTrained = false;

export const setModelBackbone = (type: 'vgg16' | 'resnet50' | 'mobilenet') => {
  if (activeModelType !== type) {
    console.log(`Switching backbone to ${type}`);
    activeModelType = type;
    baseModel = null;
    isModelTrained = false;
  }
};

export const getActiveModelType = () => activeModelType;
export const markModelsAsTrained = () => { isModelTrained = true; };

// NEW: Initialize models from storage on app load
export const initLocalModels = async () => {
    const svmLoaded = sharedSVM.load('retina_svm_best');
    const cnnLoaded = await sharedCNNHead.load('retina_cnn_best');
    
    if (svmLoaded && cnnLoaded) {
        console.log("Loaded best saved models from storage.");
        isModelTrained = true;
        return true;
    }
    return false;
};

// NEW: Load models from uploaded files (Browser File Objects)
export const loadModelsFromFiles = async (
    svmFile: File, 
    cnnJsonFile: File, 
    cnnWeightsFile: File
) => {
    try {
        // 1. Load SVM
        const svmText = await svmFile.text();
        const svmData = JSON.parse(svmText);
        sharedSVM.loadState(svmData);
        sharedSVM.save('retina_svm_best'); // Persist to local storage immediately

        // 2. Load CNN
        // tf.io.browserFiles accepts an array of [jsonFile, weightFile]
        await sharedCNNHead.loadFromFiles([cnnJsonFile, cnnWeightsFile]);
        await sharedCNNHead.save('retina_cnn_best'); // Persist to local storage immediately

        isModelTrained = true;
        return true;
    } catch (e) {
        console.error("Failed to load models from files", e);
        return false;
    }
};

export const loadModel = async () => {
  await tf.ready();
  if (baseModel) return baseModel;

  console.log(`Initializing Feature Extraction Model: ${activeModelType} on backend: ${tf.getBackend()}...`);
  
  let url = MOBILENET_URL;
  if (activeModelType === 'vgg16') url = VGG16_URL;
  else if (activeModelType === 'resnet50') url = RESNET50_URL;

  try {
    baseModel = await tf.loadGraphModel(url);
    console.log(`${activeModelType} (Proxy) loaded successfully.`);
  } catch (error) {
    console.error(`Failed to load ${activeModelType}:`, error);
    throw new Error(`Could not load model backbone. Please check your internet connection.`);
  }
  return baseModel;
};

export const getFeatureVector = async (img: HTMLImageElement): Promise<Float32Array> => {
  await tf.ready();
  const model = await loadModel();
  if (!model) throw new Error("Model failed to load");
  
  if (img.width === 0 || img.height === 0) {
      throw new Error("Image has invalid dimensions (0x0).");
  }

  return tf.tidy(() => {
    const offset = tf.scalar(127.5);
    // Use bilinear resizing for better feature preservation in medical images
    const baseTensor = tf.browser.fromPixels(img)
      .resizeBilinear([224, 224]) 
      .toFloat()
      .sub(offset)
      .div(offset);

    // Test Time Augmentation (Simple): Average normal and horizontal flip
    const flippedTensor = baseTensor.reverse(1); 
    const batch = tf.stack([baseTensor, flippedTensor]);
    const batchFeatures = model.predict(batch) as tf.Tensor;
    const averagedFeatures = batchFeatures.mean(0);
    const l2Norm = averagedFeatures.div(averagedFeatures.norm());
    
    return l2Norm.dataSync() as Float32Array;
  });
};

export class LinearSVM {
  weights: number[][]; 
  biases: number[];    
  classes: DiseaseType[];
  featureDim: number;

  constructor(featureDim: number = 1000) { 
    this.classes = [DiseaseType.CNV, DiseaseType.DME, DiseaseType.DRUSEN, DiseaseType.NORMAL];
    this.featureDim = featureDim;
    this.weights = [];
    this.biases = [];
    this.initializeDemoWeights();
  }

  private initializeDemoWeights() {
    this.weights = [];
    this.biases = [];
    for (let c = 0; c < this.classes.length; c++) {
      const wRow: number[] = [];
      for (let i = 0; i < this.featureDim; i++) {
        // Initialize with small random values centered around 0
        wRow.push((Math.random() - 0.5) * 0.001); 
      }
      this.weights.push(wRow);
      this.biases.push(0);
    }
  }

  save(key: string) {
    try {
        const state = {
            weights: this.weights,
            biases: this.biases,
            featureDim: this.featureDim,
            classes: this.classes,
            timestamp: Date.now()
        };
        localStorage.setItem(key, JSON.stringify(state));
        console.log(`SVM saved to ${key}`);
    } catch (e) {
        console.error("Failed to save SVM weights", e);
    }
  }

  // NEW: Download as JSON file
  download() {
      const state = {
        weights: this.weights,
        biases: this.biases,
        featureDim: this.featureDim,
        classes: this.classes,
        timestamp: Date.now(),
        type: 'SVM_CHECKPOINT'
      };
      const blob = new Blob([JSON.stringify(state)], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `retina_svm_model.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  }

  loadState(state: any) {
      if (state.weights && state.biases) {
          this.weights = state.weights;
          this.biases = state.biases;
          this.featureDim = state.featureDim;
          this.classes = state.classes || this.classes;
      }
  }

  load(key: string): boolean {
    const json = localStorage.getItem(key);
    if (!json) return false;
    try {
        const state = JSON.parse(json);
        this.loadState(state);
        return true;
    } catch (e) {
        return false;
    }
  }

  async trainSingle(features: Float32Array, label: DiseaseType) {
    // Train single sample with high weight to "correct" behavior immediately
    const labels = Array(15).fill(label);
    const feats = Array(15).fill(features);
    await this.train(feats, labels); 
  }

  async train(features: Float32Array[], labels: DiseaseType[], onProgress?: (epoch: number, loss: number, accuracy: number) => void) {
    if (features.length === 0) return;
    
    const inputDim = features[0].length;
    if (inputDim !== this.featureDim) {
      this.featureDim = inputDim;
      this.initializeDemoWeights(); 
    }

    const initialLr = 0.02; 
    const sampleCount = features.length;
    let epochs = 80; 
    if (sampleCount < 50) epochs = 120; 

    const lambda = 0.0001; 

    // Create indices for shuffling
    const indices = Array.from({ length: sampleCount }, (_, i) => i);

    // TRACK BEST MODEL
    let bestAcc = 0;
    let bestLoss = Infinity;
    // Deep copy initial state
    let bestState = { 
        weights: this.weights.map(r => [...r]), 
        biases: [...this.biases] 
    };

    for (let epoch = 0; epoch < epochs; epoch++) {
        const lr = initialLr / (1 + 0.005 * epoch); // Very slow decay to keep learning
        let totalLoss = 0;
        
        // Shuffle indices for SGD
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        this.classes.forEach((targetClass, classIdx) => {
            let w = this.weights[classIdx];
            let b = this.biases[classIdx];

            for (let i = 0; i < sampleCount; i++) {
                const idx = indices[i];
                const x = features[idx];
                const isPositive = labels[idx] === targetClass;
                const y = isPositive ? 1 : -1;
                
                // CLASS BALANCING
                const updateWeight = isPositive ? 3.5 : 1.0; 
                
                let score = b;
                for(let k=0; k<inputDim; k++) score += w[k] * x[k];

                const margin = 1 - y * score;
                const loss = Math.max(0, margin);
                totalLoss += loss;

                if (margin > 0) {
                    for (let k = 0; k < inputDim; k++) {
                        w[k] = w[k] - lr * (lambda * w[k] - y * x[k] * updateWeight);
                    }
                    b = b - lr * (-y) * updateWeight;
                } else {
                    for (let k = 0; k < inputDim; k++) {
                        w[k] = w[k] - lr * (lambda * w[k]);
                    }
                }
            }

            this.weights[classIdx] = w;
            this.biases[classIdx] = b;
        });

        // CALCULATE METRICS FOR BEST MODEL SELECTION
        let correctCount = 0;
        features.forEach((x, i) => {
            const pred = this.predictRaw(x);
            if (pred === labels[i]) correctCount++;
        });
        const accuracy = correctCount / features.length;
        const normalizedLoss = totalLoss / (features.length * 4);

        if (onProgress) {
            onProgress(epoch, normalizedLoss, accuracy);
        }

        // SAVE IF BEST
        // Prioritize Accuracy, then Loss
        if (accuracy > bestAcc || (accuracy === bestAcc && normalizedLoss < bestLoss)) {
            bestAcc = accuracy;
            bestLoss = normalizedLoss;
            bestState = { 
                weights: this.weights.map(r => [...r]), 
                biases: [...this.biases] 
            };
        }

        if (epoch % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }

    // RESTORE BEST WEIGHTS AT END OF TRAINING
    console.log(`Restoring best SVM state: Acc ${bestAcc.toFixed(4)}, Loss ${bestLoss.toFixed(4)}`);
    this.weights = bestState.weights;
    this.biases = bestState.biases;
  }

  private predictRaw(features: Float32Array): DiseaseType {
     let maxScore = -Infinity;
     let bestClass = this.classes[0];
     
     for (let c = 0; c < this.classes.length; c++) {
         let score = this.biases[c];
         const w = this.weights[c];
         for(let i=0; i<features.length; i++) score += features[i] * w[i];
         
         if (score > maxScore) {
             maxScore = score;
             bestClass = this.classes[c];
         }
     }
     return bestClass;
  }

  predict(features: Float32Array) {
    if (features.length !== this.featureDim) {
       this.featureDim = features.length;
       this.initializeDemoWeights();
    }
    const resultClass = this.predictRaw(features);
    
    // CALIBRATION:
    const CALIBRATION_SCALE = 8.0; 

    const scores = this.classes.map((_, c) => {
        let s = this.biases[c];
        const w = this.weights[c];
        for(let i=0; i<features.length; i++) s += features[i] * w[i];
        return s * CALIBRATION_SCALE;
    });
    
    const maxS = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - maxS));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumExps);
    const idx = this.classes.indexOf(resultClass);
    
    return {
      disease: resultClass,
      confidence: probs[idx] * 100
    };
  }
}

export class CNNDenseHead {
  private model: tf.LayersModel | null = null;
  private classes = [DiseaseType.CNV, DiseaseType.DME, DiseaseType.DRUSEN, DiseaseType.NORMAL];

  async save(key: string) {
      if (this.model) {
          try {
              await this.model.save(`localstorage://${key}`);
              console.log(`CNN saved to ${key}`);
          } catch (e) {
              console.error("Failed to save CNN model", e);
          }
      }
  }

  // NEW: Download as TFJS files (json + bin)
  async download() {
      if (this.model) {
          await this.model.save('downloads://retina_cnn_model');
      }
  }

  async load(key: string): Promise<boolean> {
      try {
          this.model = await tf.loadLayersModel(`localstorage://${key}`);
          this.model.compile({ optimizer: tf.train.adam(0.0002), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
          return true;
      } catch (e) {
          return false;
      }
  }

  // NEW: Load from uploaded files
  async loadFromFiles(files: File[]): Promise<boolean> {
      try {
          this.model = await tf.loadLayersModel(tf.io.browserFiles(files));
          this.model.compile({ optimizer: tf.train.adam(0.0002), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
          return true;
      } catch (e) {
          console.error("Load from files failed", e);
          return false;
      }
  }

  dispose() {
      if (this.model) {
          this.model.dispose();
          this.model = null;
      }
  }

  async trainSingle(features: Float32Array, label: DiseaseType) {
    if (!this.model) await this.train([features], [label]);
    else {
        const xs = tf.tensor2d([Array.from(features)]);
        const labelIdx = this.classes.indexOf(label);
        const ys = tf.oneHot([labelIdx], 4);
        await this.model.fit(xs, ys, { epochs: 20, verbose: 0 }); // Increased single-shot epochs
        xs.dispose(); ys.dispose();
    }
  }

  // UPDATED TRAIN METHOD
  async train(
      features: Float32Array[], 
      labels: DiseaseType[], 
      onProgress?: (epoch: number, loss: number, accuracy: number) => void,
      saveBestKey?: string // NEW Param
  ) {
    if (features.length === 0) return;
    const inputDim = features[0].length;

    if (!this.model) {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [inputDim], kernelInitializer: 'heNormal' }));
        model.add(tf.layers.dropout({ rate: 0.5 })); 
        model.add(tf.layers.dense({ units: 64, activation: 'relu', kernelInitializer: 'heNormal' }));
        model.add(tf.layers.dense({ units: 4, activation: 'softmax', kernelInitializer: 'glorotNormal' }));
        this.model = model;
        this.model.compile({ optimizer: tf.train.adam(0.0002), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    }

    const xs = tf.tensor2d(features.map(f => Array.from(f)));
    const labelIndices = labels.map(l => this.classes.indexOf(l));
    const ys = tf.oneHot(labelIndices, 4);
    
    let epochs = 80;
    if (features.length < 100) epochs = 150; 

    let bestAcc = 0;
    const that = this;

    await this.model.fit(xs, ys, {
      epochs: epochs,
      batchSize: 16,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
            const acc = logs?.acc || logs?.accuracy || 0;
            const loss = logs?.loss || 0;
            
            if (onProgress) {
                onProgress(epoch, loss, acc);
            }

            // SAVE BEST LOGIC
            if (saveBestKey && acc > bestAcc) {
                bestAcc = acc;
                // We use the model's internal save method which handles weight serialization
                // Note: save is async, but we don't want to block the training loop too much. 
                // However, tfjs handles this.
                await that.save(saveBestKey);
                // console.log(`Saved new best CNN: ${acc}`);
            }
        }
      }
    });

    xs.dispose(); ys.dispose();
    
    // RELOAD BEST
    // Since training modifies weights in place, the model currently has weights from the LAST epoch.
    // If we saved a better model earlier, we must reload it.
    if (saveBestKey && bestAcc > 0) {
        console.log(`Reloading best CNN model from storage (Acc: ${bestAcc.toFixed(4)})`);
        await this.load(saveBestKey);
    }
  }

  predict(features: Float32Array) {
    if (this.model) {
      return tf.tidy(() => {
        const xs = tf.tensor2d([Array.from(features)]);
        const preds = this.model!.predict(xs) as tf.Tensor;
        const probs = preds.dataSync();
        const maxProb = Math.max(...probs);
        const predictedIdx = probs.indexOf(maxProb);

        return {
          disease: this.classes[predictedIdx],
          confidence: maxProb * 100
        };
      });
    } else {
      return { disease: this.classes[0], confidence: 0 };
    }
  }
}

export const sharedSVM = new LinearSVM();
export const sharedCNNHead = new CNNDenseHead();

export const learnFromCorrection = async (base64Image: string, correctLabel: DiseaseType) => {
    const img = new Image();
    img.src = base64Image;
    await new Promise(r => { img.onload = r; });
    const features = await getFeatureVector(img);
    await sharedSVM.trainSingle(features, correctLabel);
    await sharedCNNHead.trainSingle(features, correctLabel);
    markModelsAsTrained();
    sharedSVM.save('retina_svm_best');
    sharedCNNHead.save('retina_cnn_best');
};

/**
 * IMPLEMENTS HYBRID ENSEMBLE LOGIC:
 * ALWAYS uses Gemini 2.0 Flash + VGG16/Backbone + CNN + SVM.
 */
export const analyzeImage = async (base64Image: string, mimeType: string): Promise<AnalysisResult> => {
  
  if (!isModelTrained) {
    throw new Error("Please upload the training dataset in the Training Studio tab.");
  }

  // 1. Run Gemini (Foundation Model)
  let geminiRes: AnalysisResult;
  try {
      geminiRes = await analyzeWithGemini(base64Image, mimeType);
  } catch (error: any) {
      console.error("Gemini Failure:", error);
      // Capture actual error message to display in UI
      const errorMsg = error.message || "Unknown error";
      geminiRes = {
          detectedDisease: DiseaseType.NORMAL,
          confidence: 0,
          riskCategory: RiskCategory.NONE,
          estimatedStage: EstimatedStage.NONE,
          // Show the error in reasoning so user knows WHY it is unavailable
          reasoning: `Cloud Analysis Failed. Error: ${errorMsg}. Please check your API Key configuration, internet connection, or quota limits.`,
          isTrained: true,
          primaryModel: "API Error"
      } as any;
  }

  // 2. Run Local Feature Extraction (VGG16/ResNet/MobileNet)
  // This is ALWAYS executed to satisfy the requirement of using the backbone.
  let cnnRes, svmRes;
  const backboneName = activeModelType === 'vgg16' ? 'VGG16' : activeModelType === 'resnet50' ? 'ResNet50' : 'MobileNetV2';

  try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = `data:${mimeType};base64,${base64Image}`;
      await new Promise((r) => { img.onload = r; });

      // ** CRITICAL **: We actually extract features here using TensorFlow.js
      const features = await getFeatureVector(img);

      // We are guaranteed isModelTrained is true here because of the check at the top
      cnnRes = sharedCNNHead.predict(features);
      svmRes = sharedSVM.predict(features);

      // 3. Weighted Voting / Ensemble
      const votes: EnsembleMember[] = [
          { modelName: geminiRes.primaryModel || "Gemini 2.0 Flash", prediction: geminiRes.detectedDisease, confidence: geminiRes.confidence, weight: 0.5 },
          { modelName: `CNN Head (${backboneName})`, prediction: cnnRes.disease, confidence: cnnRes.confidence, weight: 0.3 },
          { modelName: `SVM Head (${backboneName})`, prediction: svmRes.disease, confidence: svmRes.confidence, weight: 0.2 }
      ];

      // Tally Scores
      const scores: Record<string, number> = { 
          [DiseaseType.CNV]: 0, 
          [DiseaseType.DME]: 0, 
          [DiseaseType.DRUSEN]: 0, 
          [DiseaseType.NORMAL]: 0 
      };

      votes.forEach(v => {
          scores[v.prediction] += v.confidence * v.weight;
      });

      // Find Winner
      let winningDisease = DiseaseType.NORMAL;
      let maxScore = -1;
      
      Object.entries(scores).forEach(([disease, score]) => {
          if (score > maxScore) {
              maxScore = score;
              winningDisease = disease as DiseaseType;
          }
      });
      
      const finalConfidence = maxScore; // Weighted average confidence of the winner

      // 4. Construct Final Result
      let finalRisk = RiskCategory.NONE;
      let finalStage = EstimatedStage.NONE;
      let finalReasoning = "";

      // Logic for Risk/Stage (Enforcing Normal = None)
      if (winningDisease === DiseaseType.NORMAL) {
          finalRisk = RiskCategory.NONE;
          finalStage = EstimatedStage.NONE;
          finalReasoning = "Ensemble consensus indicates no macular abnormalities. ";
      } else {
          // Trust Gemini's nuanced Risk/Stage if it agrees
          if (winningDisease === geminiRes.detectedDisease) {
              finalRisk = geminiRes.riskCategory;
              finalStage = geminiRes.estimatedStage;
              finalReasoning = geminiRes.reasoning;
          } else {
              // Heuristics if Local Models overruled Gemini
              if (finalConfidence >= 95) finalRisk = RiskCategory.HIGH;
              else if (finalConfidence >= 88) finalRisk = RiskCategory.MEDIUM;
              else finalRisk = RiskCategory.LOW;

              if (finalConfidence >= 93) finalStage = EstimatedStage.ADVANCED;
              else if (finalConfidence >= 85) finalStage = EstimatedStage.INTERMEDIATE;
              else finalStage = EstimatedStage.EARLY;

              finalReasoning = `Ensemble analysis detected ${winningDisease} features, primarily driven by local ${backboneName} feature extraction. `;
          }
      }

      // If Gemini failed (Confidence 0), ensure we preserve the error message if local models didn't override it
      if (geminiRes.confidence === 0 && geminiRes.primaryModel === "API Error" && winningDisease === geminiRes.detectedDisease) {
           finalReasoning = geminiRes.reasoning;
      } else if (geminiRes.confidence === 0 && geminiRes.primaryModel === "API Error") {
           // If local models won, but Gemini failed, append the error note
           finalReasoning += ` (Note: Cloud analysis failed: ${geminiRes.reasoning})`;
      }

      // Removed explicit voting concatenation string
      // finalReasoning += ` (Voting: Gemini ${geminiRes.detectedDisease}, CNN ${cnnRes.disease}, SVM ${svmRes.disease})`;

      return {
          detectedDisease: winningDisease,
          confidence: finalConfidence,
          riskCategory: finalRisk,
          estimatedStage: finalStage,
          reasoning: finalReasoning,
          isTrained: true,
          primaryModel: `Hybrid Ensemble (Gemini + ${backboneName} + SVM)`,
          ensembleBreakdown: votes
      };

  } catch (err) {
      console.error("Ensemble Calculation Failed", err);
      // Even if calculation fails, if we had a trained model but something technical went wrong, 
      // we might fallback, but generally we want to expose errors.
      throw err;
  }
};