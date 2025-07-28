"""
Tests for security module.
"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from src.core.security import validate_api_key, verify_api_key, validate_bearer_token, verify_bearer_token, VALID_API_KEY


class TestSecurity:
    """Test cases for security functions."""
    
    def test_valid_api_key(self):
        """Test that valid API key passes verification."""
        assert verify_api_key(VALID_API_KEY) is True
    
    def test_invalid_api_key(self):
        """Test that invalid API key fails verification."""
        assert verify_api_key("invalid-key") is False
    
    def test_valid_bearer_token(self):
        """Test that valid bearer token passes verification."""
        assert verify_bearer_token(VALID_API_KEY) is True
    
    def test_invalid_bearer_token(self):
        """Test that invalid bearer token fails verification."""
        assert verify_bearer_token("invalid-token") is False
    
    @pytest.mark.asyncio
    async def test_validate_api_key_success(self):
        """Test successful API key validation."""
        result = await validate_api_key(VALID_API_KEY)
        assert result == VALID_API_KEY
    
    @pytest.mark.asyncio
    async def test_validate_api_key_failure(self):
        """Test failed API key validation raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            await validate_api_key("invalid-key")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_validate_bearer_token_success(self):
        """Test successful bearer token validation."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=VALID_API_KEY)
        result = await validate_bearer_token(credentials)
        assert result == VALID_API_KEY
    
    @pytest.mark.asyncio
    async def test_validate_bearer_token_failure(self):
        """Test failed bearer token validation raises HTTPException."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")
        with pytest.raises(HTTPException) as exc_info:
            await validate_bearer_token(credentials)
        assert exc_info.value.status_code == 401
