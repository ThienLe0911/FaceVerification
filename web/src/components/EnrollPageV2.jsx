import React, { useState, useEffect } from 'react';
import { UserPlus, Upload, CheckCircle, AlertTriangle, BarChart3, Target, Trash2, RefreshCw, Users } from 'lucide-react';
import axios from 'axios';

const EnrollPage = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [galleryStats, setGalleryStats] = useState({
    count: 0,
    diversity_score: 0,
    quality_score: 0,
    status: 'empty',
    recommendations: []
  });
  const [isUploading, setIsUploading] = useState(false);
  const [isCreatingGallery, setIsCreatingGallery] = useState(false);
  const [uploadResults, setUploadResults] = useState(null);
  const [error, setError] = useState('');

  // Fetch gallery stats on mount
  useEffect(() => {
    fetchGalleryStats();
  }, []);

  const fetchGalleryStats = async () => {
    try {
      const response = await axios.get('/api/enroll/stats');
      setGalleryStats(response.data);
    } catch (err) {
      console.error('Error fetching gallery stats:', err);
    }
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    if (files.length + selectedFiles.length > 40) {
      setError('Tối đa 40 ảnh. Vui lòng chọn ít ảnh hơn.');
      return;
    }
    setSelectedFiles([...selectedFiles, ...files]);
    setError('');
  };

  const removeFile = (index) => {
    setSelectedFiles(selectedFiles.filter((_, i) => i !== index));
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    if (files.length + selectedFiles.length > 40) {
      setError('Tối đa 40 ảnh. Vui lòng chọn ít ảnh hơn.');
      return;
    }
    setSelectedFiles([...selectedFiles, ...files]);
    setError('');
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const uploadBatch = async () => {
    if (selectedFiles.length === 0) {
      setError('Vui lòng chọn ít nhất 1 ảnh');
      return;
    }

    setIsUploading(true);
    setError('');

    try {
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append('files', file);
      });

      const response = await axios.post('/api/enroll/batch', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadResults(response.data);
      setSelectedFiles([]);
      await fetchGalleryStats(); // Refresh stats

    } catch (err) {
      setError(err.response?.data?.detail || 'Lỗi upload ảnh');
    } finally {
      setIsUploading(false);
    }
  };

  const createGallery = async () => {
    setIsCreatingGallery(true);
    setError('');

    try {
      const response = await axios.post('/api/enroll/create-gallery');
      setUploadResults({
        ...uploadResults,
        gallery_created: true,
        gallery_message: response.data.message
      });
      await fetchGalleryStats();
    } catch (err) {
      setError(err.response?.data?.detail || 'Lỗi tạo gallery');
    } finally {
      setIsCreatingGallery(false);
    }
  };

  const resetGallery = async () => {
    if (!window.confirm('Bạn có chắc muốn xóa toàn bộ gallery?')) return;
    
    try {
      await axios.delete('/api/enroll/reset');
      setGalleryStats({
        count: 0,
        diversity_score: 0,
        quality_score: 0,
        status: 'empty',
        recommendations: []
      });
      setUploadResults(null);
    } catch (err) {
      setError('Lỗi reset gallery');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'empty': return 'text-gray-500';
      case 'insufficient': return 'text-yellow-600';
      case 'ready': return 'text-green-600';
      case 'excellent': return 'text-blue-600';
      default: return 'text-gray-500';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'empty': return 'Chưa có ảnh';
      case 'insufficient': return 'Chưa đủ';
      case 'ready': return 'Sẵn sàng';
      case 'excellent': return 'Xuất sắc';
      default: return 'Không xác định';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
          <UserPlus className="h-8 w-8 text-primary-600" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Enroll PersonA
        </h1>
        <p className="text-gray-600">
          Upload 20-40 ảnh PersonA để tạo gallery. Khuyến nghị góc độ và ánh sáng đa dạng.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Section */}
        <div className="lg:col-span-2 space-y-6">
          {/* File Upload Area */}
          <div className="card p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Upload Ảnh PersonA
            </h3>
            
            <div 
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <div className="mb-4">
                <label className="cursor-pointer">
                  <span className="font-medium text-primary-600 hover:text-primary-500">
                    Click để chọn ảnh
                  </span>
                  <span className="text-gray-500"> hoặc kéo thả vào đây</span>
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
              </div>
              <p className="text-sm text-gray-500">
                PNG, JPG, GIF tối đa 10MB. Khuyến nghị 20-40 ảnh.
              </p>
            </div>

            {/* Selected Files */}
            {selectedFiles.length > 0 && (
              <div className="mt-4">
                <h4 className="font-medium text-gray-900 mb-2">
                  Đã chọn {selectedFiles.length} ảnh
                </h4>
                <div className="max-h-40 overflow-y-auto space-y-2">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span className="text-sm text-gray-700 truncate">{file.name}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500">
                          {(file.size / 1024 / 1024).toFixed(1)}MB
                        </span>
                        <button
                          onClick={() => removeFile(index)}
                          className="text-red-500 hover:text-red-700"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Upload Button */}
            <div className="mt-6">
              <button
                onClick={uploadBatch}
                disabled={selectedFiles.length === 0 || isUploading}
                className="w-full btn-primary flex items-center justify-center space-x-2"
              >
                {isUploading ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Đang xử lý...</span>
                  </>
                ) : (
                  <>
                    <Upload className="h-5 w-5" />
                    <span>Upload {selectedFiles.length} ảnh</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Upload Results */}
          {uploadResults && (
            <div className="card p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Kết quả xử lý
              </h3>
              
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {uploadResults.processed_images.map((result, index) => (
                  <div key={index} className={`flex items-center justify-between p-3 rounded-lg ${
                    result.status === 'success' ? 'bg-green-50' : 'bg-red-50'
                  }`}>
                    <div className="flex items-center space-x-2">
                      {result.status === 'success' ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-red-500" />
                      )}
                      <span className="text-sm font-medium">{result.filename}</span>
                    </div>
                    <div className="text-sm">
                      {result.quality && (
                        <span className={`px-2 py-1 rounded-full text-xs ${getScoreColor(result.quality)}`}>
                          Quality: {result.quality}%
                        </span>
                      )}
                      <span className={`ml-2 ${
                        result.status === 'success' ? 'text-green-700' : 'text-red-700'
                      }`}>
                        {result.message}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {uploadResults.suggestions.length > 0 && (
                <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h4 className="font-medium text-yellow-800 mb-2">Gợi ý:</h4>
                  <ul className="list-disc list-inside text-sm text-yellow-700 space-y-1">
                    {uploadResults.suggestions.map((suggestion, index) => (
                      <li key={index}>{suggestion}</li>
                    ))}
                  </ul>
                </div>
              )}

              {uploadResults.gallery_created && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="font-medium text-green-800">
                      {uploadResults.gallery_message}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Gallery Stats */}
        <div className="space-y-6">
          {/* Stats Card */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">
                Gallery Statistics
              </h3>
              <button
                onClick={fetchGalleryStats}
                className="text-gray-400 hover:text-gray-600"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>

            {/* Image Count */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Số ảnh</span>
                <span className="text-2xl font-bold text-gray-900">{galleryStats.count}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min((galleryStats.count / 40) * 100, 100)}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0</span>
                <span>Khuyến nghị: 20-40</span>
              </div>
            </div>

            {/* Quality Scores */}
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <BarChart3 className="h-4 w-4 text-gray-500" />
                    <span className="text-sm font-medium text-gray-700">Diversity</span>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getScoreColor(galleryStats.diversity_score)}`}>
                    {galleryStats.diversity_score}/100
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${galleryStats.diversity_score}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <Target className="h-4 w-4 text-gray-500" />
                    <span className="text-sm font-medium text-gray-700">Quality</span>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getScoreColor(galleryStats.quality_score)}`}>
                    {galleryStats.quality_score}/100
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${galleryStats.quality_score}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Status */}
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Trạng thái:</span>
                <span className={`font-medium ${getStatusColor(galleryStats.status)}`}>
                  {getStatusText(galleryStats.status)}
                </span>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          {galleryStats.recommendations.length > 0 && (
            <div className="card p-6">
              <h4 className="font-medium text-gray-900 mb-3">Khuyến nghị</h4>
              <ul className="space-y-2">
                {galleryStats.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span className="text-sm text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Actions */}
          <div className="space-y-3">
            {galleryStats.status === 'ready' || galleryStats.status === 'excellent' ? (
              <button
                onClick={createGallery}
                disabled={isCreatingGallery}
                className="w-full btn-success flex items-center justify-center space-x-2"
              >
                {isCreatingGallery ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Đang tạo Gallery...</span>
                  </>
                ) : (
                  <>
                    <Users className="h-5 w-5" />
                    <span>Tạo Gallery PersonA</span>
                  </>
                )}
              </button>
            ) : null}

            {galleryStats.count > 0 && (
              <button
                onClick={resetGallery}
                className="w-full btn-danger flex items-center justify-center space-x-2"
              >
                <Trash2 className="h-5 w-5" />
                <span>Reset Gallery</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="card p-4 bg-red-50 border border-red-200">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnrollPage;