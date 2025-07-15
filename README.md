# Ethical AI Developer Testbed

A sophisticated, research-grade web application that implements a multi-perspective mathematical framework for evaluating text content for ethical violations.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![React](https://img.shields.io/badge/react-19.0.0-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.110.1-green.svg)
![MongoDB](https://img.shields.io/badge/mongodb-4.4+-green.svg)

## 🌟 Features

- **Multi-Perspective Ethical Analysis**: Evaluates text through Virtue, Deontological, and Consequentialist ethical frameworks
- **Real-Time Evaluation**: Instant ethical analysis with detailed violation detection
- **Interactive Parameter Calibration**: Fine-tune evaluation thresholds and weights
- **Clean Text Generation**: Automatic removal of ethically problematic content
- **Comprehensive API**: RESTful API with 8 endpoints for all operations
- **Performance Monitoring**: Built-in metrics and processing time analysis
- **Calibration System**: Test case creation and validation framework

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB 4.4+
- Yarn package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ethical-ai-testbed.git
   cd ethical-ai-testbed
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   cp .env.example .env
   # Configure MONGO_URL and DB_NAME in .env
   ```

3. **Frontend Setup**
   ```bash
   cd ../frontend
   yarn install
   cp .env.example .env
   # Configure REACT_APP_BACKEND_URL in .env
   ```

4. **Start MongoDB**
   ```bash
   # Make sure MongoDB is running on localhost:27017
   mongod
   ```

5. **Run the Application**
   ```bash
   # Start backend
   cd backend
   uvicorn server:app --host 0.0.0.0 --port 8001 --reload
   
   # Start frontend (in another terminal)
   cd frontend
   yarn start
   ```

6. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8001/api
   - API Documentation: http://localhost:8001/docs

## 📖 Usage

### Text Evaluation
1. Navigate to the **Evaluate Text** tab
2. Enter your text in the textarea
3. Click **Evaluate Text** to analyze for ethical violations
4. Review the results:
   - Overall ethical status
   - Detected violations with explanations
   - Clean text with problematic content removed
   - Processing time and performance metrics

### Parameter Calibration
1. Navigate to the **Parameter Tuning** tab
2. Adjust thresholds for each ethical perspective:
   - **Virtue Threshold**: Character-based evaluation sensitivity
   - **Deontological Threshold**: Rule-based evaluation sensitivity
   - **Consequentialist Threshold**: Outcome-based evaluation sensitivity
3. Adjust weights to emphasize different perspectives
4. Changes are applied immediately to the evaluation engine

## 🏗️ Architecture

### Backend (FastAPI + Python)
- **Ethical Engine**: Core evaluation logic with ML models
- **RESTful API**: 8 comprehensive endpoints
- **Database Integration**: MongoDB with async operations
- **Performance Optimization**: Async processing and caching

### Frontend (React + Tailwind CSS)
- **Modern UI**: Clean, responsive design
- **Real-Time Updates**: Instant parameter synchronization
- **Debug Tools**: Built-in testing and diagnostics
- **Mobile Friendly**: Responsive design for all devices

### AI/ML Stack
- **Sentence Transformers**: all-MiniLM-L6-v2 model
- **scikit-learn**: Machine learning utilities
- **PyTorch**: Deep learning framework
- **Ethical Vectors**: Custom philosophical perspective embeddings

## 🔧 API Documentation

### Core Endpoints

#### Evaluate Text
```http
POST /api/evaluate
Content-Type: application/json

{
  "text": "Your text to evaluate",
  "parameters": {
    "virtue_threshold": 0.25,
    "deontological_threshold": 0.25,
    "consequentialist_threshold": 0.25
  }
}
```

#### Get/Update Parameters
```http
GET /api/parameters
POST /api/update-parameters
```

#### Calibration System
```http
POST /api/calibration-test
POST /api/run-calibration-test/{test_id}
GET /api/calibration-tests
```

For complete API documentation, visit `/docs` when running the backend.

## 🧪 Testing

### Run Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Run Frontend Tests
```bash
cd frontend
yarn test
```

### Comprehensive Testing
The application includes comprehensive test coverage for:
- API endpoints and error handling
- Ethical evaluation engine accuracy
- Database operations and serialization
- Parameter management and calibration
- Performance and scalability

## 📊 Performance

- **Processing Time**: 0.2-0.5 seconds for typical text
- **Throughput**: ~10-20 evaluations per second
- **Memory Usage**: ~500MB for loaded ML models
- **Token Limit**: 50 tokens per evaluation (optimized for real-time use)

## 🔬 Mathematical Framework

The evaluation engine implements a sophisticated mathematical framework:

1. **Text Tokenization**: Breaks text into analyzable components
2. **Span Generation**: Creates all possible text spans up to max length
3. **Perspective Scoring**: Computes ethical scores for each perspective
4. **Violation Detection**: Identifies spans exceeding thresholds
5. **Minimal Span Selection**: Finds minimal problematic spans
6. **Clean Text Generation**: Removes violations while preserving meaning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -m 'Add new feature'`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Use React best practices for frontend
- Add comprehensive tests for new features
- Update documentation for significant changes
- Consider performance impact of changes

## 📋 Requirements

### Backend Dependencies
- FastAPI 0.110.1
- sentence-transformers 5.0.0
- scikit-learn
- PyTorch 2.7.1
- Motor (MongoDB async driver)
- Pydantic for data validation

### Frontend Dependencies
- React 19.0.0
- Tailwind CSS 3.4.17
- Axios for API calls
- React Router for navigation

## 🚀 Production Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment
1. Set production environment variables
2. Build frontend: `yarn build`
3. Configure reverse proxy (nginx)
4. Set up MongoDB replica set
5. Configure SSL/TLS certificates
6. Set up monitoring and logging

## 🔒 Security Considerations

- Input validation and sanitization
- Rate limiting for API endpoints
- MongoDB injection prevention
- CORS configuration
- Environment variable security

## 📈 Roadmap

### Short-term (1-2 months)
- Enhanced error handling and user feedback
- Performance optimization and caching
- Better mobile experience
- Configuration management improvements

### Medium-term (3-6 months)
- Batch processing capabilities
- Multi-language support
- Advanced analytics and reporting
- Integration with external services

### Long-term (6+ months)
- Enterprise features (multi-tenant, RBAC)
- Advanced AI/ML capabilities
- Microservices architecture
- Cloud-native deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sentence Transformers team for the embedding models
- FastAPI team for the excellent web framework
- React team for the frontend framework
- MongoDB team for the database solution
- OpenAI for inspiration on ethical AI evaluation

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the [documentation](PROJECT_DOCUMENTATION.md)
- Review the API documentation at `/docs`

---

**Status**: Production Ready | **Version**: 1.0.0 | **Last Updated**: January 2025
