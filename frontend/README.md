# Ethical AI Developer Testbed - Frontend v1.0.1
## v3.0 Semantic Embedding Framework Interface

This React frontend implements a sophisticated interface for the Ethical AI Developer Testbed Version 1.0.1, featuring the revolutionary v3.0 semantic embedding framework with autonomy-maximization principles.

## **Framework Integration**

### **v3.0 Semantic Framework Features**
- **Autonomy-Based Evaluation**: Interface for evaluating text against human autonomy principles
- **Dimension-Specific Controls**: Controls for D1-D5 autonomy dimensions
- **Mathematical Transparency**: Visualization of orthogonal vector projections
- **Real-time Assessment**: Immediate autonomy violation detection
- **Professional Design**: Production-ready interface with autonomy focus

### **Core Axiom Integration**
The interface implements controls and displays for the Core Axiom:
**Maximize human autonomy (Σ D_i) within objective empirical truth (t ≥ 0.95)**

## **Available Scripts**

### **Development Scripts**
```bash
# Start development server with v3.0 framework
yarn start

# Run tests for autonomy interface
yarn test

# Build production version with v3.0 framework
yarn build

# Eject configuration (one-way operation)
yarn eject
```

### **v3.0 Framework Specific Scripts**
```bash
# Start with autonomy-focused development
REACT_APP_AUTONOMY_MODE=true yarn start

# Build for production with mathematical framework
yarn build:production

# Test autonomy interface components
yarn test:autonomy
```

## **Component Architecture**

### **Main Components**

#### **App.js - Main Application**
- **Dual-Tab Interface**: Text evaluation and parameter calibration
- **Autonomy Integration**: v3.0 semantic framework integration
- **State Management**: Autonomy-focused state handling
- **API Integration**: Backend communication for autonomy evaluation

#### **Text Evaluation Panel**
- **Autonomy Input**: Text input for autonomy violation detection
- **Dimension Indicators**: Real-time autonomy dimension status
- **Results Display**: Comprehensive autonomy violation breakdown
- **Clean Text Output**: Autonomy-preserving text version

#### **Parameter Calibration Panel**
- **Dimension Controls**: D1-D5 autonomy dimension thresholds
- **Mathematical Controls**: Orthogonal vector adjustment
- **Truth Prerequisites**: T1-T4 truth prerequisite settings
- **Principle Configuration**: P1-P8 ethical principle controls

#### **Results Visualization**
- **Violations Tab**: Autonomy violations with dimension mapping
- **All Spans Tab**: Complete analysis with autonomy indicators
- **Learning & Feedback Tab**: Dimension-specific feedback options
- **Dynamic Scaling Tab**: Autonomy-aware scaling information

## **v3.0 Framework Integration**

### **Autonomy-Focused Features**
- **Cognitive Autonomy**: Reasoning independence violation display
- **Behavioral Autonomy**: Coercion and manipulation indicators
- **Social Autonomy**: Bias and suppression visualization
- **Bodily Autonomy**: Harm and surveillance detection display
- **Existential Autonomy**: Future sovereignty threat indicators

### **Mathematical Framework Display**
- **Vector Projections**: Real-time s_P(i,j) = x_{i:j} · p_P visualization
- **Orthogonal Vectors**: Independence verification display
- **Minimal Spans**: Dynamic programming result visualization
- **Veto Logic**: E_v(S) ∨ E_d(S) ∨ E_c(S) = 1 assessment display

### **Performance Features**
- **Real-time Processing**: Immediate autonomy assessment
- **Caching Integration**: 2500x speedup utilization
- **Progressive Loading**: Enhanced user experience
- **Error Handling**: Graceful autonomy evaluation error handling

## **Technical Specifications**

### **Framework Integration**
- **React Version**: 19.0.0 with enhanced autonomy support
- **Tailwind CSS**: 3.4.17 with autonomy-focused styling
- **API Integration**: Autonomy-based backend communication
- **State Management**: Autonomy-focused React hooks
- **Performance**: Optimized for v3.0 framework

### **Build Configuration**
- **Craco Configuration**: Enhanced for v3.0 framework
- **PostCSS**: Autonomy-focused CSS processing
- **Environment Variables**: v3.0 framework configuration
- **Production Build**: Optimized for autonomy evaluation

### **Dependencies**
```json
{
  "react": "^19.0.0",
  "react-dom": "^19.0.0",
  "axios": "^1.x.x",
  "tailwindcss": "^3.4.17",
  "craco": "^7.x.x"
}
```

## **Development Setup**

### **Prerequisites**
- Node.js 18+ for enhanced autonomy features
- Yarn package manager
- Backend v3.0 framework running on port 8001

### **Installation**
```bash
# Clone repository
git clone <repository-url>
cd ethical-ai-testbed/frontend

# Install dependencies
yarn install

# Configure environment for v3.0 framework
cp .env.example .env
# Edit .env with appropriate REACT_APP_BACKEND_URL
```

### **Development Environment**
```bash
# Start development server
yarn start

# The app will be available at http://localhost:3000
# Backend should be running at http://localhost:8001
```

## **Environment Configuration**

### **Required Environment Variables**
```env
# Backend API URL for autonomy evaluation
REACT_APP_BACKEND_URL=http://localhost:8001

# v3.0 Framework Configuration
REACT_APP_AUTONOMY_MODE=true
REACT_APP_MATHEMATICAL_DISPLAY=true
REACT_APP_DIMENSION_INDICATORS=true
```

### **Optional Configuration**
```env
# Development features
REACT_APP_DEBUG_MODE=false
REACT_APP_PERFORMANCE_MONITORING=true

# v3.0 Framework options
REACT_APP_VECTOR_VISUALIZATION=true
REACT_APP_MINIMAL_SPAN_DISPLAY=true
```

## **API Integration**

### **Backend Communication**
The frontend communicates with the v3.0 semantic framework backend through:

#### **Autonomy Evaluation**
```javascript
// Evaluate text for autonomy violations
const response = await fetch(`${BACKEND_URL}/api/evaluate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: inputText,
    parameters: autonomyParameters
  })
});
```

#### **Parameter Management**
```javascript
// Update autonomy-based parameters
const response = await fetch(`${BACKEND_URL}/api/update-parameters`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    parameters: {
      virtue_threshold: 0.15,
      deontological_threshold: 0.15,
      consequentialist_threshold: 0.15,
      autonomy_mode: true
    }
  })
});
```

## **Component Details**

### **Autonomy Evaluation Interface**
```jsx
// Enhanced autonomy evaluation component
const AutonomyEvaluationPanel = () => {
  const [inputText, setInputText] = useState('');
  const [autonomyResult, setAutonomyResult] = useState(null);
  const [dimensionViolations, setDimensionViolations] = useState([]);
  
  const evaluateAutonomy = async () => {
    // Autonomy-based evaluation logic
  };
  
  return (
    <div className="autonomy-evaluation-panel">
      <AutonomyTextInput value={inputText} onChange={setInputText} />
      <DimensionIndicators violations={dimensionViolations} />
      <AutonomyResults result={autonomyResult} />
    </div>
  );
};
```

### **Parameter Calibration Interface**
```jsx
// Autonomy-focused parameter controls
const AutonomyParameterPanel = () => {
  const [dimensionThresholds, setDimensionThresholds] = useState({
    bodily: 0.15,
    cognitive: 0.15,
    behavioral: 0.15,
    social: 0.15,
    existential: 0.15
  });
  
  return (
    <div className="autonomy-parameter-panel">
      <DimensionControls 
        thresholds={dimensionThresholds}
        onChange={setDimensionThresholds}
      />
      <TruthPrerequisites />
      <EthicalPrinciples />
    </div>
  );
};
```

## **Styling & Design**

### **Autonomy-Focused Styling**
The interface uses enhanced Tailwind CSS with autonomy-focused design:

#### **Color Scheme**
- **Primary**: Autonomy-focused blue (#3B82F6)
- **Secondary**: Truth-based green (#10B981)
- **Violations**: Autonomy-violation red (#EF4444)
- **Neutral**: Professional gray (#6B7280)

#### **Component Styling**
```css
/* Autonomy dimension indicators */
.autonomy-dimension {
  @apply flex items-center space-x-2 p-3 rounded-lg border;
}

.autonomy-violation {
  @apply bg-red-50 border-red-200 text-red-800;
}

.autonomy-compliant {
  @apply bg-green-50 border-green-200 text-green-800;
}

/* Mathematical framework visualization */
.vector-projection {
  @apply bg-blue-50 border-blue-200 rounded-lg p-4;
}

.minimal-span {
  @apply bg-yellow-100 border-yellow-300 rounded px-2 py-1;
}
```

## **Testing**

### **Test Structure**
```bash
src/
├── components/
│   ├── AutonomyEvaluation.test.js
│   ├── ParameterCalibration.test.js
│   └── ResultsVisualization.test.js
├── hooks/
│   ├── useAutonomyEvaluation.test.js
│   └── useParameterManagement.test.js
└── utils/
    ├── autonomyHelpers.test.js
    └── mathematicalFramework.test.js
```

### **Running Tests**
```bash
# Run all tests
yarn test

# Run autonomy-specific tests
yarn test:autonomy

# Run with coverage
yarn test:coverage
```

## **Build & Deployment**

### **Production Build**
```bash
# Build for production
yarn build

# The build folder will contain optimized files for deployment
```

### **Deployment Considerations**
- **Static Hosting**: Can be deployed to any static hosting service
- **Environment Variables**: Ensure production REACT_APP_BACKEND_URL is set
- **v3.0 Framework**: Requires compatible backend deployment
- **Performance**: Built with autonomy evaluation optimization

## **Performance Optimization**

### **v3.0 Framework Performance**
- **Component Memoization**: Autonomy evaluation results cached
- **Lazy Loading**: Mathematical framework components loaded on demand
- **Debounced Input**: Autonomy evaluation triggered with debouncing
- **Efficient Re-renders**: Minimized re-rendering with React.memo

### **Bundle Optimization**
- **Code Splitting**: Autonomy components split for faster loading
- **Tree Shaking**: Unused mathematical framework code removed
- **Compression**: Gzip compression for autonomy evaluation assets
- **Caching**: Effective caching strategies for production

## **Contributing**

### **Development Guidelines**
1. **Component Structure**: Follow autonomy-focused component patterns
2. **State Management**: Use React hooks for autonomy state
3. **API Integration**: Maintain v3.0 framework compatibility
4. **Testing**: Include autonomy evaluation tests
5. **Documentation**: Update autonomy-focused documentation

### **Code Quality**
- **ESLint**: Configured for autonomy-focused code quality
- **Prettier**: Consistent code formatting
- **TypeScript**: Optional for enhanced autonomy type safety
- **Testing**: Comprehensive autonomy evaluation testing

## **Future Enhancements**

### **Planned v3.0 Framework Features**
- **Enhanced Visualization**: Mathematical framework visualization
- **Dimension Mapping**: Interactive autonomy dimension mapping
- **Real-time Feedback**: Enhanced autonomy assessment feedback
- **Performance Metrics**: v3.0 framework performance monitoring

### **UI/UX Improvements**
- **Accessibility**: Enhanced autonomy interface accessibility
- **Mobile Optimization**: Improved mobile autonomy evaluation
- **Internationalization**: Multi-language autonomy support
- **Dark Mode**: Autonomy-focused dark theme

## **Learn More**

### **Documentation**
- [React Documentation](https://reactjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [v3.0 Semantic Framework Guide](../PROJECT_DOCUMENTATION.md)
- [Autonomy-Maximization Principles](../README.md)

### **v3.0 Framework Resources**
- **Mathematical Framework**: Orthogonal vector documentation
- **Autonomy Principles**: Core Axiom implementation guide
- **Performance Optimization**: v3.0 framework performance guide
- **API Integration**: Backend communication documentation

---

*Frontend Version: 1.0.1 - v3.0 Semantic Embedding Framework*
*Status: Production Ready with Autonomy-Maximization Interface*
*Framework: Enhanced React with v3.0 Mathematical Framework Integration*