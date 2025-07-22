# Ethical AI Developer Testbed - Version 1.2

A unified ethical AI evaluation platform implementing philosophical framework analysis through computational methods.

## Version 1.2 - Unified Architecture

This version implements Clean Architecture principles with a centralized orchestrator pattern for ethical text analysis.

### Architecture Components
- **Unified Orchestrator**: Coordinates ethical analysis workflows
- **Clean Architecture**: Separation of concerns with dependency injection
- **Configuration Manager**: Environment-based configuration management
- **FastAPI Backend**: Async request handling with health monitoring
- **React Frontend**: Multi-tab interface for evaluation tasks

### Philosophical Framework Integration
- **Multi-Framework Analysis**: Virtue ethics, deontological ethics, consequentialism
- **Autonomy Assessment**: Framework for autonomy dimension evaluation
- **Mathematical Approach**: Vector-based analysis for ethical perspectives
- **Citation Structure**: Framework for academic reference integration

### Performance Characteristics
- **Response Time**: 0.025s measured average (based on testing)
- **Concurrent Processing**: Supports multiple simultaneous requests
- **Error Handling**: HTTP status codes with graceful failure handling
- **Health Monitoring**: System status reporting via `/api/health`

## Complete Repository Structure

**For AI Analysis Tools**: See `FILE_MANIFEST.md` for complete file listing with descriptions.

### Backend Components
```
/backend/
├── server.py                            # FastAPI main application server
├── unified_ethical_orchestrator.py      # Central analysis coordination
├── unified_configuration_manager.py     # Configuration management  
├── enhanced_ethics_pipeline.py          # Multi-layer philosophical analysis
├── knowledge_integration_layer.py       # External knowledge integration
├── realtime_streaming_engine.py         # WebSocket streaming capabilities
├── production_features.py               # Authentication and security features
├── core/                                # Core processing components
│   ├── __init__.py                      # Module initialization
│   ├── embedding_service.py             # Text-to-vector embedding service
│   └── evaluation_engine.py             # Async ethical evaluation engine
├── utils/                               # Utility components
│   └── caching_manager.py               # Multi-level caching system
├── ethical_engine.py                    # Original ethical evaluation engine
├── ml_ethics_engine.py                  # ML ethics specialized engine
├── multi_modal_evaluation.py            # Multi-modal evaluation capabilities
├── smart_buffer.py                      # Intelligent buffering system
├── server_legacy_backup.py              # Legacy server implementation
├── .env                                 # Environment variables
└── requirements.txt                     # Python dependencies
```

### Frontend Components
```
/frontend/
├── src/
│   ├── App.js                           # Main React application
│   ├── index.js                         # React application entry point
│   └── components/                      # React components
│       ├── EthicalChart.jsx             # Heat-map visualization component
│       ├── MLTrainingAssistant.jsx      # ML ethics interface component
│       └── RealTimeStreamingInterface.jsx # Real-time streaming UI
├── package.json                         # Node.js dependencies and scripts
├── .env                                 # Frontend environment variables
├── tailwind.config.js                   # Tailwind CSS configuration
├── postcss.config.js                    # PostCSS configuration
└── craco.config.js                      # Create React App configuration
```

### Documentation and Testing
```
/app/
├── README.md                            # Main project documentation (this file)
├── FILE_MANIFEST.md                     # Complete repository file listing
├── COMPREHENSIVE_IMPLEMENTATION_STATUS.md # Implementation status analysis
├── VERSION_1.2_CERTIFICATION.md         # Production certification report
├── COMPREHENSIVE_ASSESSMENT_V1.2.md     # Performance assessment
├── VERSION_EVOLUTION_HISTORY.md         # Development history
├── README_ACCURACY_VALIDATION.md        # Documentation validation
├── PRODUCTION_DEPLOYMENT_GUIDE.md       # Production deployment guide
├── TESTING_STATUS.md                    # Testing status and results
├── test_result.md                       # Testing results and logs
├── backend_test.py                      # Backend testing script
├── user_issue_verification_test.py      # User issue verification script
├── [JSON result files]                  # Testing results in JSON format
└── .emergent/                           # Platform configuration
    ├── emergent.yml                     # Emergent platform config
    └── summary.txt                      # Platform summary
```

## Features

### Text Evaluation
- Multi-framework ethical analysis (virtue, deontological, consequentialist)
- Confidence scoring system
- Request tracking with unique identifiers
- JSON response format with structured data

### Visualization  
- Heat-map generation for ethical analysis
- Multi-dimensional perspective visualization
- SVG-based rendering

### System Management
- Health monitoring with component status
- Parameter configuration interface
- Performance metrics collection
- Error tracking and logging

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB (local or remote)
- Git

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
yarn install
```

### Environment Configuration
Create `.env` files in both backend and frontend directories:

**Backend `.env`:**
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=ethical_ai_testbed
ETHICAL_AI_MODE=production
ETHICAL_AI_JWT_SECRET=your_secret_key_here
```

**Frontend `.env`:**
```
REACT_APP_BACKEND_URL=http://localhost:8001
```

## Usage

### Starting the Application
```bash
# Start all services
sudo supervisorctl restart all

# Or start individually  
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
```

### API Endpoints
- `GET /api/health` - System health and status
- `POST /api/evaluate` - Text ethical evaluation
- `GET /api/parameters` - Current configuration parameters
- `POST /api/update-parameters` - Update system parameters
- `POST /api/heat-map-mock` - Generate visualization data
- `GET /api/learning-stats` - System learning statistics

### Frontend Interface
1. **Evaluate Text**: Input text for ethical analysis
2. **Heat-Map**: Visual representation of ethical assessment
3. **ML Ethics Assistant**: Analysis mode selection interface
4. **Real-Time Streaming**: WebSocket-based evaluation
5. **Parameter Tuning**: Configuration adjustment interface

## Technical Specifications

### Performance
- Response time: 0.025s average (measured)
- Database: MongoDB with async driver
- Concurrent requests: Tested with 5 simultaneous users
- Error handling: Structured HTTP responses

### Architecture Principles
- Clean Architecture with dependency inversion
- SOLID principles implementation
- Async processing patterns
- Type safety with Pydantic models

### Configuration
- Environment-based configuration
- Runtime parameter updates
- Health check integration
- Structured logging

## Development

### Testing
- Backend testing via `deep_testing_backend_v2`
- Frontend testing via `auto_frontend_testing_agent`
- API validation and performance testing
- Integration testing across components

### Code Organization
- Modular design with clear separation
- Type hints and validation throughout
- Configuration management system
- Error boundaries and recovery

## Version History

### Key Versions
- **v1.0.0**: Initial ethical evaluation implementation
- **v1.1.0**: Performance optimization with caching
- **v1.2.0**: Unified architecture with Clean Architecture principles

## Production Deployment

### System Requirements
- CPU: 4+ cores recommended
- RAM: 8GB+ for optimal performance
- Storage: 20GB+ for database and application
- Network: Stable connection for external integrations

### Configuration
- SSL/TLS encryption for production
- Rate limiting configuration
- Authentication setup
- Database clustering options
- Monitoring and alerting integration

## Contributing

### Development Guidelines
1. Follow Clean Architecture principles
2. Maintain type safety with comprehensive validation
3. Include unit and integration tests
4. Document implementation decisions
5. Measure performance impact of changes

### Code Quality
- Python: PEP 8 compliance with docstrings
- JavaScript: ESLint configuration
- Architecture: Dependency inversion patterns
- Testing: Comprehensive test coverage
- Documentation: Technical specification focus

---

**Version 1.2.0 - Unified Architecture Implementation**  
*Backend Status: Operational (24/24 API tests successful)*  
*Frontend Status: Interface implemented, functionality testing required*
*Performance: 0.025s measured response times*  
*Implementation: Backend operational, frontend interface complete*

---

*The Ethical AI Developer Testbed provides a computational approach to ethical analysis using established philosophical frameworks, implemented through modern software architecture patterns.*