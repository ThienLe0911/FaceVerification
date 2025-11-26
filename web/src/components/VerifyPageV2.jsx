import React, { useState } from 'react';
import { Search, Eye, Download, RefreshCw, AlertCircle, CheckCircle, XCircle, Users, Target, Camera } from 'lucide-react';
import axios from 'axios';

const VerifyPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState(null);
  const [error, setError] = useState('');
  const [imagePreview, setImagePreview] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setVerifyResult(null);
      setError('');
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setVerifyResult(null);
      setError('');
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const verifyImage = async () => {
    if (!selectedFile) {
      setError('Vui lòng chọn ảnh để verify');
      return;
    }

    setIsVerifying(true);
    setError('');
    setVerifyResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post('/api/verify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setVerifyResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Lỗi verify ảnh');
    } finally {
      setIsVerifying(false);
    }
  };

  const downloadAnnotatedImage = () => {
    if (!verifyResult?.annotated_image_path) return;
    
    const link = document.createElement('a');
    link.href = `http://localhost:8000${verifyResult.annotated_image_path}`;
    link.download = `verification_result_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const resetVerification = () => {
    setSelectedFile(null);
    setVerifyResult(null);
    setError('');
    setImagePreview(null);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getVerdictIcon = (verdict) => {
    if (verdict.includes('Có PersonA')) {
      return <CheckCircle className="h-6 w-6 text-green-500" />;
    } else {
      return <XCircle className="h-6 w-6 text-red-500" />;
    }
  };

  const getVerdictColor = (verdict) => {
    if (verdict.includes('Có PersonA')) {
      return 'text-green-800 bg-green-100 border-green-200';
    } else {
      return 'text-red-800 bg-red-100 border-red-200';
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
          <Search className="h-8 w-8 text-primary-600" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Verify PersonA
        </h1>
        <p className="text-gray-600">
          Upload ảnh để kiểm tra có PersonA hay không. Hỗ trợ ảnh đơn và ảnh nhiều người.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Upload and Original Image */}
        <div className="space-y-6">
          {/* Upload Section */}
          <div className="card p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Upload ảnh để Verify
            </h3>
            
            <div 
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors cursor-pointer"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => document.getElementById('verify-file-input').click()}
            >
              <Camera className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <div className="mb-4">
                <span className="font-medium text-primary-600 hover:text-primary-500">
                  Click để chọn ảnh
                </span>
                <span className="text-gray-500"> hoặc kéo thả vào đây</span>
              </div>
              <p className="text-sm text-gray-500">
                PNG, JPG, GIF tối đa 10MB
              </p>
              <input
                id="verify-file-input"
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>

            {selectedFile && (
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{selectedFile.name}</p>
                    <p className="text-sm text-gray-500">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="text-red-500 hover:text-red-700"
                  >
                    <XCircle className="h-5 w-5" />
                  </button>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="mt-6 flex space-x-3">
              <button
                onClick={verifyImage}
                disabled={!selectedFile || isVerifying}
                className="flex-1 btn-primary flex items-center justify-center space-x-2"
              >
                {isVerifying ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Đang verify...</span>
                  </>
                ) : (
                  <>
                    <Eye className="h-5 w-5" />
                    <span>Verify PersonA</span>
                  </>
                )}
              </button>

              {(verifyResult || error) && (
                <button
                  onClick={resetVerification}
                  className="btn-secondary flex items-center space-x-2"
                  disabled={isVerifying}
                >
                  <RefreshCw className="h-5 w-5" />
                  <span>Reset</span>
                </button>
              )}
            </div>
          </div>

          {/* Original Image Preview */}
          {imagePreview && (
            <div className="card p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Ảnh gốc</h3>
              <div className="relative">
                <img
                  src={imagePreview}
                  alt="Original"
                  className="w-full h-auto max-h-96 object-contain bg-gray-100 rounded-lg"
                />
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Results */}
        <div className="space-y-6">
          {verifyResult && (
            <>
              {/* Verdict Card */}
              <div className={`card p-6 border-2 ${getVerdictColor(verifyResult.verdict)}`}>
                <div className="flex items-center space-x-3 mb-4">
                  {getVerdictIcon(verifyResult.verdict)}
                  <h3 className="text-xl font-bold">
                    {verifyResult.verdict}
                  </h3>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-900">
                      {verifyResult.total_faces}
                    </div>
                    <div className="text-sm text-gray-600">Tổng số khuôn mặt</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary-600">
                      {verifyResult.faces.filter(f => f.is_personA).length}
                    </div>
                    <div className="text-sm text-gray-600">PersonA detected</div>
                  </div>
                </div>

                <div className="text-center">
                  <span className={`inline-block px-4 py-2 rounded-full text-lg font-medium ${getConfidenceColor(verifyResult.confidence)}`}>
                    Confidence: {(verifyResult.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Face Details Table */}
              <div className="card p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    Chi tiết các khuôn mặt
                  </h3>
                  <Users className="h-5 w-5 text-gray-400" />
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                          Face ID
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                          Predicted
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                          Similarity
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                          Confidence
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                          Status
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {verifyResult.faces.map((face, index) => (
                        <tr key={index} className="hover:bg-gray-50">
                          <td className="px-4 py-3">
                            <div className="flex items-center space-x-2">
                              <Target className="h-4 w-4 text-gray-400" />
                              <span className="font-medium">#{face.face_id}</span>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              face.predicted === 'PersonA' 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-gray-100 text-gray-800'
                            }`}>
                              {face.predicted}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center space-x-2">
                              <div className="w-16 bg-gray-200 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full ${
                                    face.similarity >= 0.7 ? 'bg-green-500' :
                                    face.similarity >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                  style={{ width: `${face.similarity * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-sm font-medium">
                                {(face.similarity * 100).toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(face.confidence)}`}>
                              {(face.confidence * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {face.is_personA ? (
                              <CheckCircle className="h-5 w-5 text-green-500" />
                            ) : (
                              <XCircle className="h-5 w-5 text-red-500" />
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Annotated Image */}
              {verifyResult.annotated_image_path && (
                <div className="card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-medium text-gray-900">
                      Ảnh có annotation
                    </h3>
                    <button
                      onClick={downloadAnnotatedImage}
                      className="btn-secondary flex items-center space-x-2"
                    >
                      <Download className="h-4 w-4" />
                      <span>Download</span>
                    </button>
                  </div>
                  
                  <div className="relative">
                    <img
                      src={`http://localhost:8000${verifyResult.annotated_image_path}`}
                      alt="Annotated verification result"
                      className="w-full h-auto max-h-96 object-contain bg-gray-100 rounded-lg"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Suggestions */}
              {verifyResult.suggestions.length > 0 && (
                <div className="card p-6 bg-yellow-50 border border-yellow-200">
                  <div className="flex items-center space-x-2 mb-3">
                    <AlertCircle className="h-5 w-5 text-yellow-600" />
                    <h4 className="font-medium text-yellow-800">Gợi ý cải thiện</h4>
                  </div>
                  <ul className="space-y-2">
                    {verifyResult.suggestions.map((suggestion, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2 flex-shrink-0"></div>
                        <span className="text-sm text-yellow-700">{suggestion}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}

          {/* Error Display */}
          {error && (
            <div className="card p-6 bg-red-50 border border-red-200">
              <div className="flex items-center space-x-2">
                <AlertCircle className="h-5 w-5 text-red-500" />
                <div>
                  <h4 className="font-medium text-red-800">Lỗi verify</h4>
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Instructions Card (when no result) */}
          {!verifyResult && !error && (
            <div className="card p-6 bg-blue-50 border border-blue-200">
              <div className="flex items-center space-x-2 mb-3">
                <Eye className="h-5 w-5 text-blue-600" />
                <h4 className="font-medium text-blue-800">Hướng dẫn sử dụng</h4>
              </div>
              <ul className="space-y-2 text-sm text-blue-700">
                <li className="flex items-start space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Upload ảnh cần kiểm tra (có thể là ảnh đơn hoặc ảnh nhóm)</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Hệ thống sẽ phát hiện tất cả khuôn mặt trong ảnh</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>So sánh với gallery PersonA và đưa ra verdict</span>
                </li>
                <li className="flex items-start space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Xem chi tiết từng khuôn mặt với confidence score</span>
                </li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VerifyPage;