// Mock for static file imports in tests
// This file is used by Jest to mock static assets like images, fonts, etc.

module.exports = {
  // For ES6 modules
  __esModule: true,
  default: 'test-file-stub',
  
  // For CommonJS modules
  src: 'test-file-stub',
  
  // Mock file properties
  name: 'test-file.png',
  size: 1024,
  type: 'image/png',
  lastModified: Date.now(),
  
  // Mock methods that might be called on file objects
  toString: () => 'test-file-stub',
  valueOf: () => 'test-file-stub',
};

// Export as string for simple cases
module.exports.toString = () => 'test-file-stub';