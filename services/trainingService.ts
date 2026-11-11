import { DiseaseType } from '../types';
import { sharedSVM, sharedCNNHead, getFeatureVector, markModelsAsTrained, LinearSVM, CNNDenseHead } from './cnnService';

export interface TrainingSample {
  imageBase64: string;
  label: DiseaseType;
}

// Fisher-Yates shuffle
function shuffleData(features: Float32Array[], labels: DiseaseType[]) {
  let currentIndex = features.length;
  while (currentIndex !== 0) {
    const randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [features[currentIndex], features[randomIndex]] = [features[randomIndex], features[currentIndex]];
    [labels[currentIndex], labels[randomIndex]] = [labels[randomIndex], labels[currentIndex]];
  }
}

async function evaluateModel(model: LinearSVM | CNNDenseHead, features: Float32Array[], labels: DiseaseType[]): Promise<number> {
    let correct = 0;
    for (let i = 0; i < features.length; i++) {
        const pred = model.predict(features[i]);
        if (pred.disease === labels[i]) correct++;
    }
    return correct / features.length;
}

export const trainModels = async (
    dataset: TrainingSample[], 
    onLog: (msg: string) => void,
    onProgress: (phase: 'SVM' | 'CNN', epoch: number, loss: number, accuracy: number) => void
): Promise<{ samplesProcessed: number }> => {
  if (!dataset || dataset.length === 0) {
    throw new Error("Dataset is empty.");
  }

  onLog("Starting Training Pipeline...");
  
  const features: Float32Array[] = [];
  const labels: DiseaseType[] = [];

  // 1. Batch Feature Extraction
  for (let i = 0; i < dataset.length; i++) {
    const sample = dataset[i];
    try {
      const img = new Image();
      img.src = sample.imageBase64;
      await new Promise((resolve, reject) => { 
        img.onload = resolve;
        img.onerror = reject;
      });

      const vector = await getFeatureVector(img);
      features.push(vector);
      labels.push(sample.label);
      
      if (i % 5 === 0) {
          onLog(`Feature Extraction: Processed ${i + 1}/${dataset.length} images...`);
          await new Promise(r => setTimeout(r, 0));
      }
    } catch (err) {
      console.warn(`Failed to process training sample ${i}`, err);
    }
  }

  if (features.length === 0) {
    throw new Error("No valid features could be extracted from the dataset.");
  }

  onLog(`Extraction Complete. Shuffling ${features.length} samples for optimal convergence...`);
  
  // 2. Shuffle Data
  shuffleData(features, labels);

  // 3. Perform K-Fold Cross Validation (if we have enough data)
  const K = 5;
  if (features.length >= K * 2) {
      onLog(`Running ${K}-Fold Cross-Validation evaluation...`);
      const foldSize = Math.floor(features.length / K);
      let totalSvmAcc = 0;
      let totalCnnAcc = 0;

      for (let i = 0; i < K; i++) {
          const start = i * foldSize;
          const end = start + foldSize;
          
          const valFeats = features.slice(start, end);
          const valLabels = labels.slice(start, end);
          
          const trainFeats = [...features.slice(0, start), ...features.slice(end)];
          const trainLabels = [...labels.slice(0, start), ...labels.slice(end)];

          // Instantiate temporary models for this fold
          const tempSVM = new LinearSVM(features[0].length);
          const tempCNN = new CNNDenseHead();

          // Train temp models (silently, no UI progress needed for CV loop usually, or it gets spammy)
          // We don't pass a save key here because we don't want to overwrite the main model with fold models
          await tempSVM.train(trainFeats, trainLabels);
          await tempCNN.train(trainFeats, trainLabels);

          // Evaluate on validation fold
          const svmAcc = await evaluateModel(tempSVM, valFeats, valLabels);
          const cnnAcc = await evaluateModel(tempCNN, valFeats, valLabels);

          totalSvmAcc += svmAcc;
          totalCnnAcc += cnnAcc;

          onLog(`Fold ${i + 1}/${K}: SVM ${(svmAcc * 100).toFixed(1)}%, CNN ${(cnnAcc * 100).toFixed(1)}%`);
          
          // Cleanup TF memory
          tempCNN.dispose();
      }

      const avgSvm = (totalSvmAcc / K) * 100;
      const avgCnn = (totalCnnAcc / K) * 100;
      onLog(`CV Results (Avg): SVM ${avgSvm.toFixed(1)}%, CNN ${avgCnn.toFixed(1)}%`);
      onLog("Proceeding to final full-dataset training...");
  } else {
      onLog("Skipping Cross-Validation (dataset too small).");
  }

  // 4. Train Final Models (On Full Dataset)
  onLog("Starting SVM Training (Full Dataset)...");
  await new Promise(r => setTimeout(r, 100)); 
  
  // The SVM tracks best internally and restores it at the end
  await sharedSVM.train(features, labels, (epoch, loss, acc) => {
     onProgress('SVM', epoch, loss, acc);
  });
  
  // SVM is now at 'best' state. Save it.
  sharedSVM.save('retina_svm_best');
  onLog("Best SVM model saved.");

  onLog("Starting CNN Head Training (Full Dataset)...");
  // Pass key 'retina_cnn_best' so it saves checkpoints during training
  await sharedCNNHead.train(features, labels, (epoch, loss, acc) => {
     onProgress('CNN', epoch, loss, acc);
  }, 'retina_cnn_best');
  
  // The CNN train method reloads the best model at the end if a key was provided.

  markModelsAsTrained(); 
  
  onLog("Training Pipeline Finished Successfully. Best models are now active.");

  // AUTO-DOWNLOAD FOR PERSISTENCE
  onLog("Initiating download of trained model weights...");
  try {
      sharedSVM.download(); // Downloads retina_svm_model.json
      await sharedCNNHead.download(); // Downloads retina_cnn_model.json and .bin
      onLog("Downloads started. Keep these files to skip training next time!");
  } catch (err) {
      console.error("Auto-download failed", err);
      onLog("Warning: Auto-download of model files failed. Please check popup blocker.");
  }

  return {
    samplesProcessed: features.length,
  };
};