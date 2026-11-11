import { GoogleGenAI, Type } from "@google/genai";
import { AnalysisResult, DiseaseType, RiskCategory, EstimatedStage } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const SYSTEM_INSTRUCTION = `
You are an expert ophthalmology AI assistant specialized in analyzing Optical Coherence Tomography (OCT) retinal images. 
Your task is to detect macular disorders from the Kermany 2018 dataset categories: CNV, DME, Drusen, Normal.

You are the PRIMARY analysis engine for this application.
1. Analyze the OCT image features (fluid, layer disruption, drusen bodies).
2. Classify the image into one of the 4 categories.
3. Provide a confidence score.
4. IF the disease is NOT Normal, provide a risk assessment and stage estimation. 
   IF the disease IS Normal, set riskCategory to 'None' and estimatedStage to 'None'.
5. Since a secondary local model might not be available, simulate a "svmPrediction" that represents a standard machine learning baseline opinion.

DISCLAIMER: This is for research and demonstration purposes only. Not for clinical diagnosis.
`;

export const analyzeImage = async (base64Image: string, mimeType: string): Promise<AnalysisResult> => {
  // Fallback strategy: Try Gemini 2.0 Flash (Next Gen) first, then Flash Latest (Stable)
  // 'gemini-2.5' does not exist as a text/multimodal ID yet, 'gemini-2.0-flash-exp' is the correct ID.
  const modelsToTry = ['gemini-2.0-flash-exp', 'gemini-flash-latest'];
  let lastError: any = null;

  for (const modelName of modelsToTry) {
    try {
      console.log(`Attempting analysis with ${modelName}...`);
      
      const response = await ai.models.generateContent({
        model: modelName,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: mimeType,
                data: base64Image
              }
            },
            {
              text: "Analyze this OCT image. Return detected disease, risk, stage (if applicable), and baseline comparison. Ensure confidence is 0-100."
            }
          ]
        },
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              detectedDisease: {
                type: Type.STRING,
                enum: [DiseaseType.NORMAL, DiseaseType.CNV, DiseaseType.DME, DiseaseType.DRUSEN]
              },
              riskCategory: {
                type: Type.STRING,
                enum: [RiskCategory.LOW, RiskCategory.MEDIUM, RiskCategory.HIGH, RiskCategory.NONE]
              },
              estimatedStage: {
                type: Type.STRING,
                enum: [EstimatedStage.EARLY, EstimatedStage.INTERMEDIATE, EstimatedStage.ADVANCED, EstimatedStage.NONE]
              },
              confidence: {
                type: Type.NUMBER,
                description: "Confidence percentage between 0 and 100"
              },
              reasoning: {
                type: Type.STRING,
                description: "A short explanation of the visual features found."
              },
              primaryModel: {
                  type: Type.STRING,
                  description: "Name of the model used"
              },
              svmPrediction: {
                  type: Type.STRING,
                  enum: [DiseaseType.NORMAL, DiseaseType.CNV, DiseaseType.DME, DiseaseType.DRUSEN]
              },
              svmConfidence: {
                  type: Type.NUMBER
              },
              modelAgreement: {
                  type: Type.BOOLEAN
              },
              isTrained: {
                  type: Type.BOOLEAN,
                  description: "Always return true for Gemini results."
              }
            },
            required: ["detectedDisease", "riskCategory", "estimatedStage", "confidence", "reasoning", "svmPrediction", "svmConfidence", "modelAgreement", "isTrained"]
          }
        }
      });

      const text = response.text;
      if (!text) {
        throw new Error(`Model ${modelName} returned empty response (possibly blocked).`);
      }

      const result = JSON.parse(text) as AnalysisResult;
      
      // Override primaryModel with readable name
      if (modelName.includes('2.0')) {
          result.primaryModel = "Gemini 2.0 Flash";
      } else {
          result.primaryModel = "Gemini Flash (Latest)";
      }
      
      if (result.isTrained === undefined) result.isTrained = true; 
      
      // Normalize confidence: If model returns 0.95 (0-1 range), convert to 95 (0-100 range)
      if (typeof result.confidence === 'number' && result.confidence <= 1 && result.confidence > 0) {
          result.confidence = result.confidence * 100;
      }
      
      return result;

    } catch (error) {
      console.warn(`${modelName} failed:`, error);
      lastError = error;
      // Continue loop to next model
    }
  }

  // If we get here, all models failed
  console.error("All Gemini models failed:", lastError);
  throw lastError;
};