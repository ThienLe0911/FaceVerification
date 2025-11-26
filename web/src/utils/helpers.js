import { UPLOAD_CONFIG, UI_CONFIG } from './constants';

/**
 * Validates if a file is a valid image file
 */
export const isValidImageFile = (file) => {
  if (!file) return false;
  
  // Check file type
  if (!UPLOAD_CONFIG.ACCEPTED_TYPES.includes(file.type)) {
    return false;
  }
  
  // Check file size
  if (file.size > UPLOAD_CONFIG.MAX_FILE_SIZE) {
    return false;
  }
  
  return true;
};

/**
 * Formats file size to human readable format
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Gets confidence level based on threshold
 */
export const getConfidenceLevel = (confidence) => {
  if (confidence >= UI_CONFIG.CONFIDENCE_THRESHOLDS.HIGH) return 'high';
  if (confidence >= UI_CONFIG.CONFIDENCE_THRESHOLDS.MEDIUM) return 'medium';
  if (confidence >= UI_CONFIG.CONFIDENCE_THRESHOLDS.LOW) return 'low';
  return 'very-low';
};

/**
 * Gets color for confidence level
 */
export const getConfidenceColor = (confidence) => {
  const level = getConfidenceLevel(confidence);
  
  switch (level) {
    case 'high': return 'text-green-600 bg-green-100';
    case 'medium': return 'text-yellow-600 bg-yellow-100';
    case 'low': return 'text-orange-600 bg-orange-100';
    default: return 'text-red-600 bg-red-100';
  }
};

/**
 * Formats confidence percentage
 */
export const formatConfidence = (confidence) => {
  return (confidence * 100).toFixed(1) + '%';
};

/**
 * Checks if verification is successful based on threshold
 */
export const isVerificationSuccessful = (confidence, threshold) => {
  return confidence >= threshold;
};

/**
 * Generates a random ID for components
 */
export const generateId = () => {
  return Math.random().toString(36).substring(2, 15) + 
         Math.random().toString(36).substring(2, 15);
};

/**
 * Debounce function for performance optimization
 */
export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

/**
 * Creates a download link for files
 */
export const downloadFile = (url, filename) => {
  const link = document.createElement('a');
  link.href = url;
  link.download = filename || 'download';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

/**
 * Validates person name input
 */
export const isValidPersonName = (name) => {
  if (!name || typeof name !== 'string') return false;
  
  const trimmedName = name.trim();
  
  // Check minimum length
  if (trimmedName.length < 2) return false;
  
  // Check maximum length
  if (trimmedName.length > 50) return false;
  
  // Check for valid characters (letters, spaces, hyphens, apostrophes)
  const validNameRegex = /^[a-zA-Z\s\-']+$/;
  return validNameRegex.test(trimmedName);
};

/**
 * Sanitizes person name
 */
export const sanitizePersonName = (name) => {
  if (!name) return '';
  
  return name
    .trim()
    .replace(/\s+/g, ' ') // Replace multiple spaces with single space
    .replace(/[^a-zA-Z\s\-']/g, '') // Remove invalid characters
    .substring(0, 50); // Limit length
};