import React, { useState } from 'react';
import { Search, Eye, Download, RefreshCw, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import FileUpload from './FileUpload';
import axios from 'axios';

const VerifyPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [imagePreview, setImagePreview] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResult(null);
    setError('');
    
    // Create image preview
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  };

  const handleVerify = async () => {
    if (!selectedFile) {
      setError('Please select an image file');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post('/api/verify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to verify faces');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadResult = () => {
    if (!result?.annotated_image_path) return;

    // Create download link
    const link = document.createElement('a');
    link.href = `${axios.defaults.baseURL || ''}${result.annotated_image_path}`;
    link.download = `verification_result_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const resetForm = () => {
    setSelectedFile(null);
    setResult(null);
    setError('');
    setImagePreview(null);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getStatusColor = (isVerified) => {
    return isVerified ? 'text-green-600' : 'text-red-600';
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column - Upload and Controls */}
        <div className="space-y-6">
          <div className="card p-6">
            {/* Header */}
            <div className="text-center mb-6">
              <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
                <Search className="h-8 w-8 text-primary-600" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900 mb-2">
                Face Verification
              </h1>
              <p className="text-gray-600">
                Upload an image to verify faces
              </p>
            </div>

            {/* File Upload */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Image
              </label>
              <FileUpload
                onFileSelect={handleFileSelect}
                multiple={false}
                disabled={isLoading}
              />
            </div>

            {/* Original Image Preview */}
            {imagePreview && (
              <div className="mb-6">
                <h3 className="text-lg font-medium text-gray-900 mb-3">Original Image</h3>
                <div className="relative">
                  <img
                    src={imagePreview}
                    alt="Original"
                    className="w-full h-64 object-contain bg-gray-100 rounded-lg border"
                  />
                </div>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="flex items-center space-x-2 p-4 bg-red-50 border border-red-200 rounded-lg mb-6">
                <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                <p className="text-red-700">{error}</p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-4">
              <button
                onClick={handleVerify}
                disabled={isLoading || !selectedFile}
                className="flex-1 btn-primary flex items-center justify-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <Loader className="h-5 w-5 animate-spin" />
                    <span>Verifying...</span>
                  </>
                ) : (
                  <>
                    <Eye className="h-5 w-5" />
                    <span>Verify Faces</span>
                  </>
                )}
              </button>

              {(result || error) && (
                <button
                  onClick={resetForm}
                  className="btn-secondary flex items-center justify-center space-x-2"
                  disabled={isLoading}
                >
                  <RefreshCw className="h-5 w-5" />
                  <span>Reset</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Column - Results */}
        {result && (
          <div className="space-y-6">
            {/* Summary Card */}
            <div className="card p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Verification Results</h2>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-gray-900">{result.total_faces || 0}</div>
                  <div className="text-sm text-gray-600">Total Faces</div>
                </div>
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {result.verified_faces?.length || 0}
                  </div>
                  <div className="text-sm text-gray-600">Verified Faces</div>
                </div>
              </div>

              {result.verified_faces && result.verified_faces.length > 0 && (
                <div className="space-y-3">
                  <h3 className="font-medium text-gray-900">Detected People</h3>
                  {result.verified_faces.map((face, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-white border rounded-lg">
                      <div>
                        <div className="font-medium text-gray-900">{face.person_name}</div>
                        <div className="text-sm text-gray-500">
                          Position: ({Math.round(face.bbox[0])}, {Math.round(face.bbox[1])})
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-lg font-bold ${getStatusColor(face.is_verified)}`}>
                          {face.is_verified ? '✓' : '✗'}
                        </div>
                        <div className={`text-xs px-2 py-1 rounded-full ${getConfidenceColor(face.confidence)}`}>
                          {(face.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Annotated Image */}
            {result.annotated_image_path && (
              <div className="card p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-gray-900">Annotated Result</h3>
                  <button
                    onClick={downloadResult}
                    className="btn-secondary flex items-center space-x-2"
                  >
                    <Download className="h-4 w-4" />
                    <span>Download</span>
                  </button>
                </div>
                
                <div className="relative">
                  <img
                    src={`${axios.defaults.baseURL || ''}${result.annotated_image_path}`}
                    alt="Annotated verification result"
                    className="w-full h-auto object-contain bg-gray-100 rounded-lg border"
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                </div>
              </div>
            )}

            {/* Processing Time */}
            {result.processing_time && (
              <div className="card p-4">
                <div className="flex items-center justify-center space-x-2 text-gray-600">
                  <span className="text-sm">Processing time:</span>
                  <span className="text-sm font-medium">{result.processing_time.toFixed(2)}s</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VerifyPage;