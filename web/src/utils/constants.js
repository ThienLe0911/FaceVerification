// API Configuration
export const API_BASE_URL = 'http://localhost:8000';
export const API_ENDPOINTS = {
  ENROLL: '/api/enroll',
  VERIFY: '/api/verify',
  THRESHOLD: '/api/threshold',
};

// File Upload Configuration
export const UPLOAD_CONFIG = {
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  ACCEPTED_TYPES: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
  ACCEPTED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.gif', '.webp'],
};

// UI Configuration
export const UI_CONFIG = {
  CONFIDENCE_THRESHOLDS: {
    HIGH: 0.8,
    MEDIUM: 0.6,
    LOW: 0.4,
  },
  COLORS: {
    PRIMARY: '#2563eb',
    SUCCESS: '#10b981',
    WARNING: '#f59e0b',
    DANGER: '#ef4444',
    INFO: '#3b82f6',
  },
};

// Default Values
export const DEFAULTS = {
  VERIFICATION_THRESHOLD: 0.5,
  PROCESSING_TIMEOUT: 30000, // 30 seconds
};