import React, { useState, useEffect } from 'react';
import { Settings, Save, RefreshCw, AlertCircle, CheckCircle, Info } from 'lucide-react';
import axios from 'axios';

const SettingsPage = () => {
  const [threshold, setThreshold] = useState(0.5);
  const [originalThreshold, setOriginalThreshold] = useState(0.5);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Fetch current threshold on component mount
  useEffect(() => {
    fetchThreshold();
  }, []);

  const fetchThreshold = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get('/api/threshold');
      const currentThreshold = response.data.threshold;
      setThreshold(currentThreshold);
      setOriginalThreshold(currentThreshold);
      setError('');
    } catch (err) {
      setError('Failed to fetch current threshold');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('/api/threshold', {
        threshold: parseFloat(threshold)
      });

      setResult(response.data);
      setOriginalThreshold(threshold);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update threshold');
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    setThreshold(originalThreshold);
    setResult(null);
    setError('');
  };

  const handleThresholdChange = (e) => {
    setThreshold(parseFloat(e.target.value));
    setResult(null);
    setError('');
  };

  const hasChanges = threshold !== originalThreshold;

  const getThresholdDescription = (value) => {
    if (value >= 0.8) return "Very strict - Only very close matches will be accepted";
    if (value >= 0.6) return "Strict - Good balance between security and usability";
    if (value >= 0.4) return "Moderate - More lenient matching";
    if (value >= 0.2) return "Lenient - Accepts broader range of matches";
    return "Very lenient - May accept false positives";
  };

  const getThresholdColor = (value) => {
    if (value >= 0.7) return "text-green-600";
    if (value >= 0.4) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
            <Settings className="h-8 w-8 text-primary-600" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Verification Settings
          </h1>
          <p className="text-gray-600">
            Configure face verification sensitivity
          </p>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center space-x-2 text-gray-500">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span>Loading settings...</span>
            </div>
          </div>
        )}

        {/* Settings Form */}
        {!isLoading && (
          <div className="space-y-6">
            {/* Threshold Setting */}
            <div>
              <label htmlFor="threshold" className="block text-sm font-medium text-gray-700 mb-2">
                Verification Threshold
              </label>
              
              <div className="space-y-4">
                {/* Slider */}
                <div className="relative">
                  <input
                    type="range"
                    id="threshold"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={threshold}
                    onChange={handleThresholdChange}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    style={{
                      background: `linear-gradient(to right, #ef4444 0%, #f59e0b 50%, #10b981 100%)`
                    }}
                  />
                  
                  {/* Tick marks */}
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0.1</span>
                    <span>0.3</span>
                    <span>0.5</span>
                    <span>0.7</span>
                    <span>0.9</span>
                  </div>
                </div>

                {/* Current Value Display */}
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <div className="font-medium text-gray-900">Current Value</div>
                    <div className={`text-sm ${getThresholdColor(threshold)}`}>
                      {getThresholdDescription(threshold)}
                    </div>
                  </div>
                  <div className={`text-2xl font-bold ${getThresholdColor(threshold)}`}>
                    {threshold.toFixed(2)}
                  </div>
                </div>

                {/* Info Box */}
                <div className="flex items-start space-x-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <Info className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
                  <div className="text-blue-700 text-sm">
                    <p className="font-medium mb-1">How threshold works:</p>
                    <ul className="list-disc list-inside space-y-1">
                      <li><strong>Higher values (0.7-0.9):</strong> More secure but may reject valid faces</li>
                      <li><strong>Medium values (0.4-0.7):</strong> Balanced approach (recommended)</li>
                      <li><strong>Lower values (0.1-0.4):</strong> More permissive but may accept wrong faces</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="flex items-center space-x-2 p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                <p className="text-red-700">{error}</p>
              </div>
            )}

            {/* Success Display */}
            {result && (
              <div className="flex items-center space-x-2 p-4 bg-green-50 border border-green-200 rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
                <div className="text-green-700">
                  <p className="font-medium">Settings Updated Successfully</p>
                  <p className="text-sm">{result.message}</p>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-4">
              <button
                onClick={handleSave}
                disabled={!hasChanges || isSaving}
                className="flex-1 btn-primary flex items-center justify-center space-x-2"
              >
                {isSaving ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Saving...</span>
                  </>
                ) : (
                  <>
                    <Save className="h-5 w-5" />
                    <span>Save Changes</span>
                  </>
                )}
              </button>

              {hasChanges && (
                <button
                  onClick={handleReset}
                  className="btn-secondary"
                  disabled={isSaving}
                >
                  Reset
                </button>
              )}
              
              <button
                onClick={fetchThreshold}
                className="btn-secondary flex items-center space-x-2"
                disabled={isSaving || isLoading}
              >
                <RefreshCw className="h-5 w-5" />
                <span>Refresh</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SettingsPage;