from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from src.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variable, with fallback for testing
VALID_API_KEY = settings.API_KEY
if VALID_API_KEY == "12345678901":
    logger.warning("Using default API key. Set API_KEY environment variable for production.")

# Create HTTPBearer instance with proper OpenAPI integration
security = HTTPBearer(
    scheme_name="BearerAuth",
    description="Enter your bearer token (use: 12345678901)",
    auto_error=True  # This ensures 401 errors are raised automatically
)

async def validate_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Validate the bearer token against the expected API key.
    
    Args:
        credentials: The HTTP authorization credentials containing the bearer token
        
    Returns:
        str: The validated token
        
    Raises:
        HTTPException: If the token is invalid or missing
    """
    if not credentials:
        logger.error("❌ Authentication FAILED - No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token. Please provide a valid token in the Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials.strip()  # Remove any whitespace
    
    # Remove sensitive logging in production
    logger.info(f"🔍 Validating token (length: {len(token)})")
    # Don't log actual token values in production
    
    if not token:
        logger.error("❌ Authentication FAILED - Empty token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token. Please provide a valid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if token != VALID_API_KEY:
        logger.error(f"❌ Authentication FAILED - Invalid token: '{token}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token. Wrong code.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info("✅ Authentication PASSED - Valid token provided")
    return token