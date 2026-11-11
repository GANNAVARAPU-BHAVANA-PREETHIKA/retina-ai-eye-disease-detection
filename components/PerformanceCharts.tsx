import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { PerformanceMetric } from '../types';

const data: PerformanceMetric[] = [
  { metric: 'Accuracy', cnn: 0.95, svm: 0.89 },
  { metric: 'Precision', cnn: 0.94, svm: 0.87 },
  { metric: 'Recall', cnn: 0.96, svm: 0.88 },
  { metric: 'F1-Score', cnn: 0.95, svm: 0.87 },
];

const PerformanceCharts: React.FC = () => {
  return (
    <div className="w-full mt-16 bg-white rounded-xl shadow-md border border-gray-200 p-8">
      <div className="mb-8 border-b border-gray-100 pb-4">
        <h2 className="text-2xl font-bold text-gray-800">Model Evaluation Metrics</h2>
        <p className="text-gray-500">Comparative performance analysis: CNN (Deep Learning) vs. SVM (Machine Learning)</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Bar Chart */}
        <div className="h-80 w-full flex flex-col">
          <h4 className="text-center text-sm font-semibold text-gray-500 mb-4">Performance Comparison</h4>
          <div className="flex-grow min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="metric" tick={{fontSize: 12}} />
                <YAxis domain={[0, 1]} tick={{fontSize: 12}} />
                <Tooltip 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                  cursor={{fill: '#f3f4f6'}}
                />
                <Legend wrapperStyle={{paddingTop: '20px'}} />
                <Bar dataKey="cnn" name="Proposed CNN (Transfer Learning)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="svm" name="SVM (Feature Extraction)" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Radar Chart */}
        <div className="h-80 w-full flex flex-col">
          <h4 className="text-center text-sm font-semibold text-gray-500 mb-4">Holistic Model Score</h4>
          <div className="flex-grow min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" tick={{fontSize: 11, fill: '#64748b'}} />
                <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} />
                <Radar name="Proposed CNN" dataKey="cnn" stroke="#3b82f6" strokeWidth={2} fill="#3b82f6" fillOpacity={0.3} />
                <Radar name="SVM" dataKey="svm" stroke="#94a3b8" strokeWidth={2} fill="#94a3b8" fillOpacity={0.3} />
                <Legend wrapperStyle={{paddingTop: '20px'}} />
                <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      <div className="mt-8 bg-blue-50 p-4 rounded-lg border border-blue-100 text-sm text-blue-900">
        <span className="font-bold">Observation:</span> The Transfer Learning based CNN model (VGG16 backbone) consistently outperforms the SVM classifier across all key metrics (Accuracy, Precision, Recall, F1), demonstrating superior capability in extracting complex hierarchical features from retinal OCT images.
      </div>
    </div>
  );
};

export default PerformanceCharts;