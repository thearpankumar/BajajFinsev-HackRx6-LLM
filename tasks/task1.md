### Task 1: Core API and Authentication

**Objective:** Set up the main FastAPI application, define the `/hackrx/run` endpoint, and implement the strict Bearer token authentication.

**Development Status:** Initial setup. This task is foundational and should be completed first to establish the API contract and security.

---

#### Task 1.1: Define Pydantic Schemas

* **Objective:** Create the data models for API requests and responses.
* **File:** `src/api/v1/schemas.py`
* **Details:**
  * Define an `AnalysisRequest` model with `documents: List[HttpUrl]` and `questions: List[str]`.
  * Define an `AnalysisResponse` model with `answers: List[str]`.
  * Ensure models match the exact input/output structure from the swappy-20250725-150151.png image.
* **Tech Stack:** Pydantic.

#### Task 1.2: Implement Comprehensive Security (Authentication & HTTPS Plan)

* **Objective:** Define and implement the complete security model, covering application-level authentication and the production plan for transport-layer encryption.
* **File(s):** `src/core/security.py` (for code), N/A (for infrastructure plan).
* **Details:**
  * API Key Authentication (Code):
    * Implement a `validate_api_key` function in `src/core/security.py` using FastAPI's `APIKeyHeader`.
    * The required token is the static string: `589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68`.
    * Raise an `HTTPException` with `status.HTTP_401_UNAUTHORIZED` for invalid tokens.
  * HTTPS Encryption (Plan):
    * This is a deployment-level concern, not an application code task.
    * For production on AWS, the application will be deployed behind an Application Load Balancer (ALB) configured with an AWS Certificate Manager (ACM) certificate to handle HTTPS termination. No action is needed for local development.
* **Tech Stack:** FastAPI, AWS Application Load Balancer (ALB), AWS Certificate Manager (ACM).

#### Task 1.3: Create Core API Endpoint

* **Objective:** Define the main RAG execution endpoint.
* **File:** `src/api/v1/endpoints/analysis.py`
* **Details:**
  * Create an `APIRouter` instance.
  * Define the `POST /hackrx/run` endpoint.
  * Secure the endpoint by adding `validate_api_key` as a dependency.
  * The endpoint will accept `AnalysisRequest` and return `AnalysisResponse`.
  * Start with a placeholder response to confirm the setup works.
* **Tech Stack:** FastAPI.

#### Task 1.4: Initialize FastAPI Application

* **Objective:** Set up the main FastAPI application instance and include the API router.
* **File:** `src/main.py`
* **Details:**
  * Create the main FastAPI app instance.
  * Include the `APIRouter` from `src/api/v1/endpoints/analysis.py`.
* **Tech Stack:** FastAPI.
