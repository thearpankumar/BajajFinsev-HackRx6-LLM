# BajajFinsev-HackRx6-LLM: Multi-LLM RAG Decision Engine

## 🚀 Project Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-aware answers by leveraging multiple Large Language Models (LLMs) and a hybrid database approach. It's built with FastAPI for high performance and asynchronous operations, making it suitable for high-stakes domains like finance and legal document analysis.

The system addresses the challenge of LLM "hallucination" by grounding responses in a trusted document corpus, ensuring factual accuracy and explainability.

## ✨ Key Features

*   **Multi-Stage Query Processing:**
    *   **LLM-powered Query Clarification:** Transforms vague user prompts into detailed, structured queries using advanced LLMs (e.g., GPT-4o).
    *   **Hybrid Database Search:** Combines relational database (PostgreSQL) for document summaries and Pinecone (vector database) for precise clause retrieval.
*   **Asynchronous Document Ingestion:**
    *   Supports PDF, DOCX, and EML document uploads.
    *   Asynchronous parsing, semantic segmentation, summarization, embedding generation, and storage to both PostgreSQL and Pinecone.
*   **Multi-LLM Orchestration:** Intelligently routes sub-tasks to optimal LLMs (e.g., GPT-4o for synthesis, Mistral Medium for interpretation).
*   **Robust Authentication:** Secure API access using a strict Bearer token.
*   **Modular and Scalable Architecture:** Designed for maintainability, testability, and future expansion.
*   **Containerized Deployment:** Utilizes Docker and Docker Compose for consistent environments.
*   **Automated CI/CD:** GitHub Actions pipeline for continuous integration and deployment to an Azure VM.

## 🛠️ Technology Stack

*   **Backend Framework:** FastAPI (Python)
*   **Relational Database:** PostgreSQL
*   **Vector Database:** Pinecone
*   **Large Language Models (LLMs):** OpenAI (GPT-4o), Mistral AI (Mistral Medium)
*   **Document Parsing:** PyMuPDF, python-docx, Python's `email` library
*   **Text Processing:** spaCy (for semantic segmentation)
*   **Embeddings:** Sentence Transformers
*   **Authentication:** FastAPI's `APIKeyHeader`
*   **Containerization:** Docker, Docker Compose
*   **CI/CD:** GitHub Actions
*   **Deployment Target:** Azure Cloud VM

## 📂 Project Structure

The project follows a modular and organized structure to separate concerns and enhance maintainability:

```
BajajFinsev/
├── .git/
├── .github/
│   └── workflows/
│       └── main.yml             # GitHub Actions CI/CD workflow
├── config/
│   └── nginx/
│       ├── Dockerfile           # Dockerfile for custom Nginx image
│       └── nginx.conf           # Nginx configuration for reverse proxy
├── src/
│   ├── __init__.py              # Python package initializer
│   ├── main.py                  # Main FastAPI app instance and router
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── endpoints/
│   │       │   ├── __init__.py
│   │       │   ├── analysis.py  # POST /hackrx/run endpoint logic
│   │       │   └── documents.py # POST /documents/upload endpoint logic
│   │       └── schemas.py       # Pydantic models for API requests/responses
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management (e.g., env vars)
│   │   └── security.py          # Authentication and security dependencies
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py              # SQLAlchemy base model and session management
│   │   └── models.py            # SQLAlchemy ORM models (Document, Clause, etc.)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion_service.py # Logic for document parsing, embedding, DB storage
│   │   └── rag_workflow.py      # Core multi-agent RAG workflow logic
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_main.py         # Unit tests for FastAPI app
│   └── utils/
│       ├── __init__.py
│       └── document_parsers.py  # Parsers for PDF, DOCX, EML
├── tasks/                       # Detailed breakdown of project implementation tasks
│   ├── task1.md
│   ├── task2.md
│   └── task3.md
├── Dockerfile                   # Dockerfile for the FastAPI application
├── docker-compose.yml           # Docker Compose configuration for FastAPI and Nginx
├── requirements.txt             # Python dependencies
├── README.md                    # Project README file
├── Process Report.pdf           # Original project documentation
├── .gitignore                   # Git ignore file
└── BajajFinsevHarckRx_key.pem   # SSH Private Key (should be in .gitignore and removed from history)
```

## 🚀 Getting Started

### Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thearpankumar/BajajFinsev-HackRx6-LLM.git
    cd BajajFinsev-HackRx6-LLM
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    Create a `.env` file in the root directory with your API keys and database credentials.
    ```
    # .env example
    OPENAI_API_KEY=your_openai_key
    MISTRAL_API_KEY=your_mistral_key
    PINECONE_API_KEY=your_pinecone_key
    PINECONE_ENVIRONMENT=your_pinecone_env
    DATABASE_URL=postgresql://user:password@host:port/dbname
    API_AUTH_TOKEN=589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68
    ```
5.  **Run with Docker Compose (recommended for local testing):**
    ```bash
    docker compose up --build
    ```
    Your FastAPI app will be accessible at `http://localhost:80`.

### Deployment to Azure VM (via GitHub Actions)

This project is configured for automated CI/CD to an Azure VM using GitHub Actions.

1.  **Azure VM Setup (Manual Steps on VM):**
    *   SSH into your Azure VM (`ssh -i <private-key-file-path> azureuser@74.225.254.124`).
    *   **Uninstall K3s (if previously installed):**
        ```bash
        sudo /usr/local/bin/k3s-uninstall.sh
        ```
    *   **Install Docker & Docker Compose:**
        ```bash
        sudo apt update
        sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
        sudo bash -c 'curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg'
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt update
        sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        sudo usermod -aG docker azureuser
        newgrp docker # Log out and back in if changes don't apply immediately
        ```
    *   **Open Ports:** Ensure **Port 80 (HTTP)** and **Port 443 (HTTPS - if using SSL)** are open in your Azure Network Security Group (NSG) for the VM.

2.  **GitHub Repository Configuration:**
    *   Go to your GitHub repository settings: `https://github.com/thearpankumar/BajajFinsev-HackRx6-LLM/settings/secrets/actions`
    *   Add the following **Repository Secrets**:
        *   `DOCKER_USERNAME`: Your Docker Hub username (`arpankumar1119`).
        *   `DOCKER_PASSWORD`: Your Docker Hub Access Token (generate from Docker Hub settings, with "Write" permissions).
        *   `SSH_PRIVATE_KEY`: The content of your SSH private key (`BajajFinsevHarckRx_key.pem`). **Ensure this file is NOT committed to your repository.**

3.  **Trigger Deployment:**
    *   Any `git push` to the `master` branch will trigger the CI/CD pipeline.
    *   The pipeline will:
        *   Run build checks (linting with Ruff, testing with Pytest).
        *   Build and push Docker images (FastAPI app and Nginx) to Docker Hub.
        *   SSH into your VM, pull the latest code, and deploy the services using `docker compose up -d`.

## 🔑 API Endpoints

### `POST /hackrx/run`

Processes documents and answers questions using the RAG Decision Engine.

*   **URL:** `http://<YOUR_VM_PUBLIC_IP>/hackrx/run` (e.g., `http://74.225.254.124/hackrx/run`)
*   **Method:** `POST`
*   **Headers:**
    *   `Content-Type: application/json`
    *   `Accept: application/json`
*   **Request Body Example:**
    ```json
    {
      "documents": [
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T099"
      ],
      "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
      ]
    }
    ```
*   **Response Body Example:**
    ```json
    {
      "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or co",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first poli",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination"
      ]
    }
    ```

### `POST /documents/upload` (Future Endpoint)

This endpoint will handle the ingestion of new documents into the RAG system.

## 🔒 Authentication

API access is secured using a Bearer token. The `Authorization` header must contain the exact string:

`Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68`

Any other token or its absence will result in a `401 Unauthorized` response.

## 📈 Future Enhancements

*   **Trusted SSL/HTTPS:** Implement a publicly trusted SSL certificate (e.g., via Let's Encrypt and Certbot) for secure communication. This requires a registered domain name.
*   **Asynchronous Ingestion Queue:** Implement a message queue (e.g., RabbitMQ, Redis Queue) for robust asynchronous document ingestion.
*   **LLM Router Implementation:** Fully implement the dynamic LLM routing based on query characteristics and performance tiers.
*   **Comprehensive Error Handling & Logging:** Enhance error handling and implement structured logging.
*   **Monitoring & Alerting:** Set up Prometheus/Grafana for monitoring application and infrastructure metrics.
*   **Scalability:** Explore Kubernetes deployment for horizontal scaling of services.
*   **User Management:** Implement user authentication and authorization beyond API key.
*   **Document Versioning:** Support for managing different versions of documents.
*   **Evaluation Framework:** Integrate an automated RAG evaluation framework ("LLM-as-a-Judge").
