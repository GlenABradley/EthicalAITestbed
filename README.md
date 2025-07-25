# Ethical AI Developer Testbed: A Framework for AI Alignment

**An advanced evaluation platform that generates structured, N-dimensional ethical vectors to teach machine learning models human values.**

---

## The Vision: Solving the AI Alignment Problem

The fundamental challenge in creating safe, human-aligned AI is a data problem. Large Language Models (LLMs) learn from vast datasets that teach them what is *true* but not what is *good*. Human values are implicit, contradictory, and lack a structured, machine-readable format.

This project is an attempt to solve that problem. It is not merely a text analysis tool; it is a **machine for generating structured ethical data**. By processing text, it produces a high-resolution, N-dimensional vector that represents the text's ethical content. This provides the precise kind of labeled data needed to align AI models with human values through fine-tuning or reinforcement learning (RLAIF).

## The Core Philosophy: A Mathematical Approach to Ethics

This system is built on a novel premise: that ethical perspectives can be modeled mathematically as independent vectors in a high-dimensional space. The core axiom is to **maximize human autonomy within the bounds of empirical truth**.

To achieve this, the system models the three primary schools of Western philosophy as orthogonal vectors:

1.  **Virtue Ethics**: Focuses on the character of the moral agent.
2.  **Deontology**: Focuses on the adherence to rules and duties.
3.  **Consequentialism**: Focuses on the outcomes of actions.

These vectors are made mathematically independent using **Gram-Schmidt orthogonalization**. This ensures that when the system evaluates a piece of text, it can project the text's meaning onto each philosophical axis independently, preventing the frameworks from interfering with one another and providing a clear, multi-faceted ethical analysis.

## How It Works: The Evaluation Pipeline

The system transforms raw text into a structured ethical vector through a sophisticated pipeline:

1.  **Text Ingestion**: Receives input text via API or UI.
2.  **Span Detection**: Identifies ethically salient spans of text for granular analysis.
3.  **Semantic Embedding**: Converts text spans into high-dimensional vectors using advanced sentence-transformer models.
4.  **Ethical Vector Projection**: Projects the semantic embedding onto the pre-calculated orthogonal ethical framework vectors.
5.  **N-Dimensional Output**: Produces a final vector representing the text's ethical signature across multiple philosophical, autonomy, and truth-based dimensions.

## Key Features

- **Orthogonal Ethical Frameworks**: Mathematically independent analysis of Virtue, Deontological, and Consequentialist ethics.
- **N-Dimensional Ethical Vectors**: Generates rich, machine-readable outputs perfect for AI training datasets.
- **Multi-Scale Analysis**: Evaluates text at the token, span, sentence, and document level for unparalleled resolution.
- **High-Performance Engine**: Built with an async-first FastAPI backend and a highly optimized core for processing at scale.
- **Bayesian Optimization**: Includes a framework for automatically tuning ethical scalar parameters to maximize cluster resolution.
- **Unified Architecture**: Implements Clean Architecture principles for maintainability, testability, and scalability.

## What's New in Version 1.2: Advanced Ethical Oversight

Version 1.2 represents a major architectural evolution, transforming the testbed into a dynamic, real-time ethical oversight platform. The system now includes a modular pipeline, multiple evaluation modes, and a sophisticated streaming analysis engine, enabling it to act as an intelligent ethical firewall for LLMs and AI agents.

### 1. Modular & Layered Architecture

The core has been re-engineered into a highly modular, multi-layered pipeline, ensuring separation of concerns and enhancing scalability.

-   **Knowledge Layer**: Integrates and indexes external knowledge sources into vector stores and relational data, providing a rich, queryable context for the evaluation engine.
-   **Ethics Evaluation Pipeline**: A three-tiered engine for comprehensive analysis:
    -   **Foundational Layer**: Establishes the meta-ethical framework and resolves conflicts between different ethical perspectives.
    -   **Normative Layer**: Performs the core evaluation using the orthogonal Virtue, Deontological, and Consequentialist models.
    -   **Applied Layer**: Specializes the analysis for specific domains, including Digital Ethics (Privacy, Transparency) and AI Ethics (Fairness, Accountability).
-   **Execution Pipeline**: Takes the ethical requirements and constraints from the evaluation and translates them into actionable decisions, including routing requests to LLMs and monitoring their outputs.

### 2. Dynamic Evaluation Modes

The system now operates in three distinct modes, offering flexible ethical oversight across the entire lifecycle of an AI interaction:

-   **Pre-Evaluation Mode**: Analyzes prompts *before* they are sent to an LLM to check for malicious intent, safety risks, and context violations.
-   **Post-Evaluation Mode**: Validates LLM outputs *after* generation to ensure they align with ethical boundaries and do not contain harmful content.
-   **Streaming Mode**: Provides real-time, continuous evaluation of token streams as they are generated, enabling immediate intervention and dynamic adaptation.

### 3. Real-Time Streaming with a Smart Buffer System

At the heart of the new streaming capability is a sophisticated **Smart Buffer System**. This is not a simple queue; it is an intelligent, adaptive mechanism designed for real-time analysis.

-   **Adaptive Analysis**: Uses pattern recognition and semantic understanding to intelligently chunk token streams, ensuring that evaluations happen at meaningful semantic boundaries (e.g., complete phrases or sentences).
-   **Resource Optimization**: Dynamically resizes the buffer, prioritizes critical content, and manages system load to ensure high performance and low latency.
-   **Control System Integration**: Feeds real-time performance metrics to a control system that can dynamically adjust thresholds and resource allocation, creating a robust feedback loop.

### 4. Expanded API for Ethical Oversight

To support these new capabilities, the API has been expanded with dedicated endpoints:

-   `/pre-evaluation/`: For analyzing input prompts.
-   `/post-evaluation/`: For validating LLM responses.
-   `/stream-evaluation/`: For managing real-time WebSocket connections.
-   `/configuration/`: For dynamically tuning the system's ethical parameters and buffer settings.


## For Researchers & Developers

This testbed is designed for two primary audiences:

-   **AI Alignment Researchers**: Provides a powerful tool for generating high-quality, structured ethical datasets. Use this system to create reward models for RLAIF or to fine-tune foundation models for safer, more aligned behavior.
-   **Software & Systems Engineers**: Offers a robust, production-ready example of implementing complex philosophical concepts within a modern software architecture (FastAPI, React, MongoDB). Explore advanced concepts like dependency injection, asynchronous processing, and multi-level caching.

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB (local or remote)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GlenABradley/EthicalAITestbed.git
    cd EthicalAITestbed
    ```

2.  **Configure Environment:**
    Create `.env` files in both the `backend` and `frontend` directories. Use the `.env.example` files in each as a template.

3.  **Install Dependencies:**
    ```bash
    # Install backend dependencies
    cd backend
    pip install -r requirements.txt

    # Install frontend dependencies
    cd ../frontend
    npm install
    ```

4.  **Run the Application:**
    The application is managed by `supervisor`.
    ```bash
    # Start backend and frontend services
    sudo supervisorctl restart all
    ```
    - The backend will be available at `http://localhost:8001`.
    - The frontend will be available at `http://localhost:3000`.

## System Architecture

This project uses a modern, decoupled architecture designed for performance and scalability.

-   **Backend**: A fully asynchronous API built with **FastAPI**, featuring dependency injection, Pydantic data validation, and a high-performance `core` evaluation engine.
-   **Frontend**: A responsive user interface built with **React** and **Tailwind CSS**, providing tools for evaluation, visualization, and parameter tuning.
-   **Database**: **MongoDB** for storing evaluation results, caching, and knowledge integration.
-   **Orchestration**: A `UnifiedEthicalOrchestrator` manages the complex evaluation pipeline, ensuring clean separation of concerns.

For a complete file listing, see [FILELIST.md](./FILELIST.md).

## Contributing

Contributions are welcome. Please adhere to the principles of Clean Architecture and ensure comprehensive testing and documentation for any new features. See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.