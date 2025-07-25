### Task 1: Core API and Authentication

**Objective:** Set up the main FastAPI application, define the `/hackrx/run` endpoint, and implement the strict Bearer token authentication.

**Development Status:** Initial setup. This task is foundational and should be completed first to establish the API contract and security.

---

#### Task 1.1: Define Pydantic Schemas

*   **Objective:** Create the data models for API requests and responses.
*   **File:** `src/api/v1/schemas.py`
*   **Details:**
    *   Define `AnalysisRequest` model with `documents: List[HttpUrl]` and `questions: List[str]`.
    *   Define `AnalysisResponse` model with `answers: List[str]`.
    *   Ensure models match the exact input/output structure provided in the `swappy-20250725-150151.png` image.
*   **Tech Stack:** Pydantic.

#### Task 1.2: Implement API Key Security

*   **Objective:** Create the authentication logic for the API.
*   **File:** `src/core/security.py`
*   **Details:**
    *   Implement `validate_api_key` function using FastAPI's `APIKeyHeader`.
    *   The `EXPECTED_TOKEN` constant must be the exact string: `589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68`.
    *   If the token is missing or incorrect, raise `HTTPException` with `status.HTTP_401_UNAUTHORIZED`.
*   **Tech Stack:** FastAPI.

#### Task 1.3: Create Core API Endpoint

*   **Objective:** Define the main RAG execution endpoint.
*   **File:** `src/api/v1/endpoints/analysis.py`
*   **Details:**
    *   Create an `APIRouter` instance.
    *   Define the `POST /hackrx/run` endpoint.
    *   Integrate `validate_api_key` as a dependency to secure the endpoint.
    *   The endpoint should accept `AnalysisRequest` as its request body and be configured to return `AnalysisResponse`.
    *   Initially, implement a placeholder response (e.g., returning the input questions as answers) to confirm authentication and routing work.
*   **Tech Stack:** FastAPI.

#### Task 1.4: Initialize FastAPI Application

*   **Objective:** Set up the main FastAPI application instance and include the API router.
*   **File:** `src/main.py`
*   **Details:**
    *   Create the main `FastAPI` app instance.
    *   Include the `APIRouter` from `src/api/v1/endpoints/analysis.py`.
*   **Tech Stack:** FastAPI.