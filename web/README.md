# Face Verification Web Interface

A modern React-based web interface for the Face Verification System, providing an intuitive way to enroll people and verify faces.

## Features

### 🔐 Face Enrollment
- Upload photos to register new people
- Automatic face detection and validation
- Real-time feedback on enrollment status
- Support for multiple image formats

### 🔍 Face Verification  
- Multi-face detection in uploaded images
- Real-time confidence scoring
- Visual annotations with bounding boxes
- Downloadable annotated results
- Detailed verification reports

### ⚙️ Settings Management
- Adjustable verification threshold
- Interactive threshold configuration
- Real-time sensitivity preview
- Persistent settings storage

### 🎨 User Interface
- Modern, responsive design
- Drag-and-drop file upload
- Real-time progress indicators  
- Comprehensive error handling
- Mobile-friendly interface

## Technology Stack

- **Frontend Framework**: React 18 with Vite
- **Styling**: Tailwind CSS with custom components
- **Routing**: React Router DOM
- **HTTP Client**: Axios with interceptors
- **Icons**: Lucide React
- **Development**: Hot reload, proxy configuration

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python backend server running on port 8000

### Installation

1. **Install dependencies**
   ```bash
   cd web
   npm install
   ```

2. **Start development server**
   ```bash
   npm run dev
   ```

3. **Open browser**
   ```
   http://localhost:5173
   ```

## Project Structure

```
web/
├── public/                 # Static assets
├── src/
│   ├── components/        # React components
│   │   ├── Header.jsx     # Navigation header
│   │   ├── EnrollPage.jsx # Person enrollment
│   │   ├── VerifyPage.jsx # Face verification
│   │   ├── SettingsPage.jsx # Configuration
│   │   └── FileUpload.jsx # File upload component
│   ├── services/         # API services
│   │   └── api.js        # Axios configuration
│   ├── utils/            # Utilities
│   │   ├── constants.js   # App constants
│   │   └── helpers.js     # Helper functions
│   ├── App.jsx           # Main app component
│   ├── main.jsx          # Entry point
│   └── index.css         # Global styles
├── package.json          # Dependencies
├── vite.config.js        # Vite configuration
└── tailwind.config.js    # Tailwind configuration
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Integration

The frontend communicates with the FastAPI backend through these endpoints:

- **POST /api/enroll** - Enroll new person
- **POST /api/verify** - Verify faces in image
- **GET/POST /api/threshold** - Manage verification threshold

## Configuration

### Environment Variables
- `VITE_API_BASE_URL` - Backend API URL (default: http://localhost:8000)

### Customization
- Modify colors in `tailwind.config.js`
- Adjust constants in `src/utils/constants.js`
- Update API configuration in `src/services/api.js`

## Deployment

### Production Build
```bash
npm run build
```

### Serve Static Files
The built files in `dist/` can be served by any static file server or integrated with the FastAPI backend.

## Browser Support

- Chrome 80+
- Firefox 75+  
- Safari 13+
- Edge 80+

## Features in Detail

### File Upload Component
- Drag and drop support
- File type validation
- Size limit enforcement
- Preview functionality
- Multiple file support

### Responsive Design
- Mobile-first approach
- Tablet and desktop optimized
- Touch-friendly interactions
- Accessible components

### Error Handling
- Network error recovery
- File validation feedback
- API error display
- Graceful degradation

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure backend server is running on port 8000
   - Check CORS configuration
   - Verify API endpoints

2. **File Upload Errors**
   - Check file size (max 10MB)
   - Verify image format (JPG, PNG, GIF, WebP)
   - Ensure proper file permissions

3. **Build Issues**
   - Clear node_modules and reinstall
   - Check Node.js version compatibility
   - Verify package.json dependencies

### Development Tips
- Use browser dev tools for debugging
- Check network tab for API calls
- Monitor console for errors
- Test with different image formats

## Contributing

1. Follow React best practices
2. Use TypeScript for new components
3. Maintain responsive design
4. Test across browsers
5. Update documentation