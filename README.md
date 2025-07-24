# Ethical AI Developer Testbed

**An open-source platform for the multi-perspective ethical evaluation of text-based AI models.**

This testbed provides a robust, modular, and high-performance environment for analyzing AI-generated text against multiple established ethical frameworks. It is designed to serve as a research and development tool for building safer and more aligned AI systems.

---

## Table of Contents
- [Abstract](#abstract)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Key Features](#key-features)
- [Performance](#performance)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Abstract

The Ethical AI Developer Testbed is an enterprise-grade application designed to operationalize ethical analysis. It moves beyond single-metric evaluations by integrating multiple philosophical perspectives—including Virtue Ethics, Deontology, and Consequentialism—into a unified computational framework. The system uses a sophisticated, multi-layered architecture to provide nuanced, evidence-backed assessments of AI-generated text. Key technical features include a centralized orchestration engine, a high-performance asynchronous processing pipeline, a multi-level caching system for sub-second response times, and a Retrieval-Augmented Generation (RAG) layer for grounding evaluations in external knowledge sources. This testbed is built for developers and researchers who require a rigorous, scalable, and transparent tool for evaluating the ethical dimensions of their models.

---

## System Architecture

The v1.2 architecture is modeled on modern, distributed systems principles, emphasizing separation of concerns, scalability, and resilience. It functions as a coordinated ecosystem of specialized microservices managed by a central orchestrator.

```
+--------------------------------------------------------------------------+
|                                  User / Client                           |
+--------------------------------------------------------------------------+
                 | (React Frontend / API Call)
                 v
+--------------------------------------------------------------------------+
|   FastAPI Backend (`server.py`)                                          |
|   (API Gateway, Load Balancing, Health Checks, Legacy Endpoint Support)  |
+--------------------------------------------------------------------------+
                 | (Evaluation Request)
                 v
+--------------------------------------------------------------------------+
|   Unified Ethical Orchestrator (`unified_ethical_orchestrator.py`)       |
|   (Coordinates workflows, selects evaluation strategy)                   |
+--------------------------------------------------------------------------+
                 |
                 +---------------------> (To appropriate evaluation engine)
                 |
  +--------------v-------------+      +--------------------------+      +-------------------------+
  | Multi-Modal Evaluation     |      | Knowledge Integration    |      | Optimized Core Engine   |
  | (`multi_modal_evaluation`) |      | (`knowledge_integrator`) |      | (`evaluation_engine`)   |
  | - Pre/Post Safety Checks   |      | - RAG (Wikipedia, etc.)  |      | - Async Processing      |
  | - Circuit Breaker Pattern  |      | - Vector Search (FAISS)  |      | - High-Performance      |
  +----------------------------+      +--------------------------+      +-----------+-------------+
                                                                                    | (Embeddings)
                                                                                    v
                                                                      +-------------+-------------+
                                                                      | Embedding & Caching Layer |
                                                                      | (`embedding_service.py`)  |
                                                                      | (`caching_manager.py`)    |
                                                                      +---------------------------+
```

### Design Principles
- **Clean Architecture:** Enforces a strict separation of concerns. Business logic is independent of frameworks, databases, and UI.
- **Asynchronous Processing:** Built on FastAPI and `asyncio` to handle high-concurrency workloads without blocking.
- **Dependency Injection:** FastAPI's dependency injection system is used extensively to manage component lifecycles and facilitate testing.
- **Backward Compatibility:** The original v1.1 `EthicalEvaluator` is encapsulated and maintained as a valid evaluation strategy, ensuring a seamless upgrade path.

---

## Core Components

### Backend
- **`server.py`**: The main FastAPI application. Acts as the API gateway, handling incoming HTTP requests, routing, and dependency management. It also manages the lifecycle of all major components and exposes health check endpoints.
- **`unified_ethical_orchestrator.py`**: The central nervous system of the application. It receives evaluation requests from the server and coordinates the complex workflow between different analysis engines and knowledge layers.
- **`core/evaluation_engine.py`**: A high-performance, asynchronous evaluation engine designed to solve the performance bottlenecks of the original engine.
- **`utils/caching_manager.py`**: A sophisticated, thread-safe, multi-level caching system (L1: Embedding, L2: Evaluation, L3: Preprocessing) with TTL and LRU eviction policies. This component is critical for achieving high performance on repetitive tasks.
- **`core/embedding_service.py`**: A dedicated service for converting text to vector embeddings using `sentence-transformers`. It leverages the caching manager and a `ThreadPoolExecutor` to perform this CPU-intensive task asynchronously and efficiently.
- **`knowledge_integration_layer.py`**: Implements a Retrieval-Augmented Generation (RAG) pipeline. It fetches and processes data from external sources like Wikipedia to provide context and evidence for ethical evaluations.
- **`multi_modal_evaluation.py`**: An advanced pipeline that wraps evaluations with pre-flight safety checks and post-flight analyses. It uses the Strategy pattern to select different evaluation modes and a Circuit Breaker pattern for resilience.
- **`ethical_engine.py`**: The original, legacy v1.1 evaluation engine. It is preserved for backward compatibility and can be invoked by the orchestrator as a specific evaluation strategy.

### Frontend
- **`App.js`**: The main React component that manages the application layout, routing, and state.
- **`/components`**: A directory of reusable React components for rendering UI elements like the evaluation forms, heat-map visualizations, and real-time interfaces.
- **`Tailwind CSS`**: Used for utility-first styling to maintain a consistent and professional UI.

---

## Key Features

- **Multi-Framework Evaluation:** Analyzes text from the perspectives of Virtue Ethics, Deontology, and Consequentialism.
- **High-Performance Caching:** A multi-level cache dramatically reduces latency for repeated or similar queries, with observed speedups of over 2500x on embedding lookups.
- **Asynchronous Architecture:** The system is fully asynchronous, enabling it to handle a high volume of concurrent requests efficiently.
- **RAG for Grounding:** Integrates external knowledge to ground ethical judgments in verifiable data, reducing hallucinations and improving transparency.
- **Legacy System Integration:** Preserves the original evaluation engine, allowing for A/B testing and phased migration.
- **Modular and Extensible:** Designed with Clean Architecture, making it straightforward to add new ethical frameworks, knowledge sources, or evaluation engines.

---

## Performance

Performance is a core feature of the v1.2 architecture, achieved primarily through software engineering rather than hardware scaling.

- **Latency:** While response times are data-dependent, cached evaluation results are typically returned in **under 50 milliseconds**.
- **Throughput:** The asynchronous nature of the FastAPI backend allows for high throughput, limited primarily by CPU cores for uncached, compute-intensive requests.
- **Bottleneck Mitigation:** The primary performance bottleneck—text embedding—has been aggressively optimized via a dedicated, cached, and asynchronous service.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+ & `yarn`
- MongoDB
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/GlenABradley/EthicalAITestbed.git
cd EthicalAITestbed
```

### 2. Configure Environment
Create a `.env` file in the `/backend` directory:
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=ethical_ai_testbed
```
Create a `.env` file in the `/frontend` directory:
```
REACT_APP_BACKEND_URL=http://localhost:8001
```

### 3. Install Dependencies
```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
yarn install
```

### 4. Run the Application
```bash
# Run the backend server
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload

# In a new terminal, run the frontend
cd frontend
yarn start
```
The application should now be running at `http://localhost:3000`.

---

## Usage

### API Endpoints
The backend exposes a REST API for programmatic interaction. Key endpoints include:
- `GET /api/health`: Returns the health status of the application and its components.
- `POST /api/evaluate`: The primary endpoint for submitting text for ethical evaluation.
- `POST /api/evaluate/legacy`: An endpoint for accessing the original v1.1 engine directly.

### Frontend Interface
The React-based frontend provides an intuitive user interface for:
- Submitting text for evaluation.
- Viewing detailed, multi-faceted ethical analysis results.
- Visualizing ethical assessments via heat-maps.
- Configuring evaluation parameters.

---

## Project Structure

```
/
├── backend/
│   ├── server.py                   # Main FastAPI application
│   ├── unified_ethical_orchestrator.py # Central workflow coordinator
│   ├── core/                       # High-performance core components
│   │   ├── evaluation_engine.py
│   │   └── embedding_service.py
│   ├── utils/
│   │   └── caching_manager.py
│   ├── knowledge_integration_layer.py
│   ├── multi_modal_evaluation.py
│   ├── ethical_engine.py           # Legacy v1.1 engine
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   └── components/
│   └── package.json
│
└── README.md                       # This file
```

---

## Contributing
Contributions are welcome. Please adhere to the following guidelines:
1.  **Follow Architectural Patterns:** New features should respect the existing Clean Architecture and SOLID principles.
2.  **Write Tests:** Include unit or integration tests for new functionality.
3.  **Maintain Objectivity:** All documentation and code comments should be professional and objective.
4.  **Format Code:** Adhere to `PEP 8` for Python and standard `ESLint` rules for JavaScript/React.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.