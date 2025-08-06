# Changelog

All notable changes to the Ethical AI Testbed project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2025-08-06

### Added
- **Complete Adaptive Threshold Learning System**: Implemented perceptron-based threshold optimization with three variants (classic, averaged, voted)
- **Intent Hierarchy Normalization**: Empirical grounding of ethical scores using α=0.2 sensitivity parameter
- **Orthonormalized Feature Extraction**: QR decomposition ensures mathematical independence of ethical axes
- **Comprehensive Training Data Pipeline**: Synthetic data generation across 5 domains with manual annotation support
- **Frontend Integration**: Complete UI overhaul with new AdaptiveThresholdInterface component
- **Audit Logging**: Full transparency and traceability for all training and prediction operations
- **Model Persistence**: Save/load functionality for trained models with metadata
- **Repository Cleanup**: Removed obsolete test files and consolidated useful testing infrastructure

### Changed
- **Navigation Interface**: Replaced "Parameter Tuning" tab with "🧠 Adaptive Thresholds" tab
- **UI Paradigm**: Shifted from manual threshold adjustment to intelligent, ML-based optimization
- **Test Structure**: Reorganized tests into unit/, integration/, validation/, and operational/ directories
- **Documentation**: Complete overhaul of README and addition of comprehensive technical documentation

### Removed
- **Obsolete Parameter Tuning**: Removed manual slider-based threshold adjustment interface
- **Legacy Test Files**: Cleaned up redundant and obsolete test scripts and result files
- **Temporary Artifacts**: Removed debug files, logs, and intermediate result files

### Fixed
- **Syntax Errors**: Resolved all React JSX syntax errors in App.js
- **Import Issues**: Fixed component imports and navigation structure
- **Code Quality**: Improved error handling and type consistency

### Technical Details
- **Mathematical Framework**: Implemented s_P' = s_P * (1 + α * sim(intent_vec, E_P)) for intent normalization
- **Orthonormalization**: Q, R = qr(ethical_matrix) ensures axis independence with <1e-10 numerical error
- **Performance**: Achieved 67% accuracy on bootstrapped training data
- **Autonomy Preservation**: Maintains cognitive autonomy (D₂) through transparent, auditable learning

## [1.2.1] - 2025-08-05

### Added
- **Perceptron Threshold Learner**: Initial implementation of adaptive threshold learning
- **Mathematical Framework Validation**: Comprehensive testing of orthonormalization
- **Training Data Bootstrapping**: Initial pipeline for generating training examples

### Changed
- **Feature Extraction**: Enhanced with intent normalization capabilities
- **API Structure**: Added endpoints for adaptive threshold management

### Fixed
- **Orthonormalization Issues**: Resolved axis score homogenization problems
- **Async Method Calls**: Fixed await patterns in feature extraction

## [1.2.0] - 2025-08-04

### Added
- **Intent Hierarchy Integration**: Connected ethical evaluation with intent classification
- **Orthonormalization Implementation**: Gram-Schmidt process for axis independence
- **Advanced Feature Extraction**: Multi-dimensional ethical feature vectors

### Changed
- **Evaluation Engine**: Enhanced with orthonormalized ethical axes
- **API Responses**: Added intent normalization metadata

## [1.1.x] - 2025-07-xx

### Added
- **Core Ethical Evaluation Engine**: Multi-perspective ethical analysis
- **FastAPI Backend**: RESTful API with comprehensive endpoints
- **React Frontend**: Modern web interface with real-time capabilities
- **Multi-Framework Analysis**: Virtue, deontological, and consequentialist ethics
- **Real-time Streaming**: WebSocket-based live evaluation

### Technical Foundation
- **Ethical Frameworks**: Mathematical modeling of three ethical perspectives
- **Text Analysis**: Semantic evaluation with embedding-based similarity
- **Threshold Management**: Basic manual threshold adjustment
- **Visualization**: Interactive charts and ethical vector displays

---

## Version Numbering

This project uses semantic versioning (SemVer):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

## Migration Guides

### Migrating from v1.1.x to v1.2.x
- **API Changes**: New adaptive threshold endpoints added, existing endpoints unchanged
- **Frontend**: New tab structure, old evaluation functionality preserved
- **Configuration**: New environment variables for adaptive learning (optional)

### Migrating from v1.2.1 to v1.2.2
- **Frontend**: Navigation updated, adaptive threshold interface replaces parameter tuning
- **Testing**: Test files reorganized, update test import paths if using custom tests
- **Documentation**: Updated README and added comprehensive guides
