import React, { useState } from 'react';
import { UserPlus, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import FileUpload from './FileUpload';
import axios from 'axios';

const EnrollPage = () => {
  const [personName, setPersonName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResult(null);
    setError('');
  };

  const handleEnroll = async () => {
    if (!personName.trim()) {
      setError('Please enter a person name');
      return;
    }

    if (!selectedFile) {
      setError('Please select an image file');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('person_name', personName.trim());
      formData.append('file', selectedFile);

      const response = await axios.post('/api/enroll', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
      
      // Reset form on success
      if (response.data.status === 'success') {
        setPersonName('');
        setSelectedFile(null);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to enroll person');
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setPersonName('');
    setSelectedFile(null);
    setResult(null);
    setError('');
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mx-auto mb-4">
            <UserPlus className="h-8 w-8 text-primary-600" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Enroll New Person
          </h1>
          <p className="text-gray-600">
            Add a new person to the face verification database
          </p>
        </div>

        {/* Form */}
        <div className="space-y-6">
          {/* Person Name Input */}
          <div>
            <label htmlFor="personName" className="block text-sm font-medium text-gray-700 mb-2">
              Person Name
            </label>
            <input
              type="text"
              id="personName"
              value={personName}
              onChange={(e) => setPersonName(e.target.value)}
              placeholder="Enter person's name"
              className="input"
              disabled={isLoading}
            />
          </div>

          {/* File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload Photo
            </label>
            <FileUpload
              onFileSelect={handleFileSelect}
              multiple={false}
              disabled={isLoading}
            />
          </div>

          {/* Error Display */}
          {error && (
            <div className="flex items-center space-x-2 p-4 bg-red-50 border border-red-200 rounded-lg">
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className={`p-4 rounded-lg border ${
              result.status === 'success' 
                ? 'bg-green-50 border-green-200' 
                : 'bg-red-50 border-red-200'
            }`}>
              <div className="flex items-start space-x-2">
                {result.status === 'success' ? (
                  <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1">
                  <h3 className={`font-medium ${
                    result.status === 'success' ? 'text-green-800' : 'text-red-800'
                  }`}>
                    {result.status === 'success' ? 'Enrollment Successful!' : 'Enrollment Failed'}
                  </h3>
                  <p className={`mt-1 text-sm ${
                    result.status === 'success' ? 'text-green-700' : 'text-red-700'
                  }`}>
                    {result.message}
                  </p>
                  {result.faces_count !== undefined && (
                    <p className={`mt-1 text-sm ${
                      result.status === 'success' ? 'text-green-700' : 'text-red-700'
                    }`}>
                      Faces detected: {result.faces_count}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-4">
            <button
              onClick={handleEnroll}
              disabled={isLoading || !personName.trim() || !selectedFile}
              className="flex-1 btn-primary flex items-center justify-center space-x-2"
            >
              {isLoading ? (
                <>
                  <Loader className="h-5 w-5 animate-spin" />
                  <span>Enrolling...</span>
                </>
              ) : (
                <>
                  <UserPlus className="h-5 w-5" />
                  <span>Enroll Person</span>
                </>
              )}
            </button>

            {(result || error) && (
              <button
                onClick={resetForm}
                className="btn-secondary"
                disabled={isLoading}
              >
                Reset
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnrollPage;