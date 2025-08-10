#!/usr/bin/env python3
"""
HackRx Challenge Solver MCP Server
Handles the Sachin's Parallel World challenge using FastAPI
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="HackRx Challenge Solver", description="MCP Server for HackRx challenges")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Landmark mapping data from PDF
LANDMARK_MAPPINGS = {
    # Indian Cities - Current locations in parallel world
    "Gateway of India": "Delhi",
    "India Gate": "Mumbai", 
    "Charminar": "Chennai",
    "Marina Beach": "Hyderabad",
    "Howrah Bridge": "Ahmedabad",
    "Golconda Fort": "Mysuru",
    "Qutub Minar": "Kochi",
    "Taj Mahal": "Hyderabad",  # Also appears in international section as Paris
    "Meenakshi Temple": "Pune",
    "Lotus Temple": "Nagpur",
    "Mysore Palace": "Chandigarh",
    "Rock Garden": "Kerala",
    "Victoria Memorial": "Bhopal",
    "Vidhana Soudha": "Varanasi",
    "Sun Temple": "Jaisalmer",
    "Golden Temple": "Pune",
    
    # International Cities - Current locations in parallel world
    "Eiffel Tower": "New York",
    "Statue of Liberty": "London",
    "Big Ben": "Tokyo",  # Also appears as Istanbul
    "Colosseum": "Beijing",
    "Sydney Opera House": "London",
    "Christ the Redeemer": "Bangkok",
    "Burj Khalifa": "Toronto",
    "CN Tower": "Dubai",
    "Petronas Towers": "Amsterdam",
    "Leaning Tower of Pisa": "Cairo",
    "Mount Fuji": "San Francisco",
    "Niagara Falls": "Berlin",
    "Louvre Museum": "Barcelona",
    "Stonehenge": "Moscow",
    "Sagrada Familia": "Seoul",
    "Acropolis": "Cape Town",
    "Machu Picchu": "Riyadh",
    "Moai Statues": "Dubai Airport",
    "Christchurch Cathedral": "Singapore",
    "The Shard": "Jakarta",
    "Blue Mosque": "Vienna",
    "Neuschwanstein Castle": "Kathmandu",
    "Buckingham Palace": "Los Angeles",
    "Space Needle": "Mumbai",
    "Times Square": "Seoul"
}

# Reverse mapping: city -> landmark
CITY_TO_LANDMARK = {}
for landmark, city in LANDMARK_MAPPINGS.items():
    if city not in CITY_TO_LANDMARK:
        CITY_TO_LANDMARK[city] = []
    CITY_TO_LANDMARK[city].append(landmark)

# API endpoints for flight numbers
FLIGHT_ENDPOINTS = {
    "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
    "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber",
    "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
    "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber",
    "default": "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"
}

class ChallengeState:
    """Manages the state of the challenge solving process"""
    def __init__(self):
        self.favorite_city: Optional[str] = None
        self.landmark: Optional[str] = None
        self.flight_number: Optional[str] = None
        self.steps_completed: List[str] = []
    
    def reset(self):
        """Reset the challenge state"""
        self.__init__()


challenge_state = ChallengeState()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx Challenge Solver Server",
        "version": "1.0.0",
        "available_endpoints": [
            "POST /call/fetch_url_data",
            "POST /call/get_favorite_city", 
            "POST /call/decode_city_landmark",
            "POST /call/get_flight_number",
            "POST /call/solve_complete_challenge",
            "POST /call/get_challenge_status",
            "POST /call/get_landmark_mappings"
        ],
        "description": "Server for handling HackRx Parallel World challenges"
    }


async def _fetch_url_data(url: str) -> Dict[str, Any]:
    """Internal function to fetch URL data"""
    logger.info(f"Fetching data from URL: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Try to parse as JSON first
            try:
                data = response.json()
                content_type = "json"
            except json.JSONDecodeError:
                # Fall back to text
                data = response.text
                content_type = "text"
            
            result = {
                "status": "success",
                "status_code": response.status_code,
                "content_type": content_type,
                "data": data,
                "headers": dict(response.headers),
                "url": str(response.url)
            }
            
            logger.info(f"Successfully fetched data from {url}: {content_type}")
            return result
            
    except httpx.TimeoutException:
        error_result = {
            "status": "error",
            "error_type": "timeout",
            "message": f"Request to {url} timed out after 30 seconds",
            "url": url
        }
        logger.error(error_result["message"])
        return error_result
        
    except httpx.HTTPStatusError as e:
        error_result = {
            "status": "error",
            "error_type": "http_error",
            "message": f"HTTP {e.response.status_code} error for {url}",
            "status_code": e.response.status_code,
            "url": url
        }
        logger.error(error_result["message"])
        return error_result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error_type": "general_error", 
            "message": f"Failed to fetch {url}: {str(e)}",
            "url": url
        }
        logger.error(error_result["message"])
        return error_result


@app.post("/call/fetch_url_data")
async def fetch_url_data(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch data from any URL with error handling and response parsing.
    """
    url = request.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    return await _fetch_url_data(url)


@app.post("/call/get_favorite_city")
async def get_favorite_city(request: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Step 1: Get the favorite city from the HackRx API.
    
    Returns:
        Dictionary containing the favorite city and challenge state
    """
    logger.info("Getting favorite city from HackRx API...")
    
    url = "https://register.hackrx.in/submissions/myFavouriteCity"
    result = await _fetch_url_data(url)
    
    if result["status"] == "success":
        # Extract city name from response
        if result["content_type"] == "json":
            # The API returns {"success": true, "data": {"city": "New York"}}
            api_response = result["data"]
            if isinstance(api_response, dict) and "data" in api_response:
                city = api_response["data"]["city"]
            else:
                city = str(api_response).strip().strip('"')
        else:
            # Handle text response
            city = result["data"].strip().strip('"')
        
        challenge_state.favorite_city = city
        challenge_state.steps_completed.append("favorite_city_retrieved")
        
        logger.info(f"Retrieved favorite city: {city}")
        
        return {
            "status": "success",
            "favorite_city": city,
            "step": "1_favorite_city_retrieved",
            "next_step": "decode_landmark"
        }
    else:
        return {
            "status": "error",
            "message": f"Failed to get favorite city: {result.get('message', 'Unknown error')}",
            "error_details": result
        }


@app.post("/call/decode_city_landmark") 
async def decode_city_landmark(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Decode which landmark belongs to the given city in the parallel world.
    """
    city = request.get("city")
    if not city:
        raise HTTPException(status_code=400, detail="City is required")
    logger.info(f"Decoding landmark for city: {city}")
    
    # Find landmarks in this city
    landmarks = CITY_TO_LANDMARK.get(city, [])
    
    if not landmarks:
        return {
            "status": "error",
            "message": f"No landmarks found for city: {city}",
            "city": city,
            "available_cities": list(CITY_TO_LANDMARK.keys())
        }
    
    challenge_state.landmark = landmarks  # Store all landmarks
    challenge_state.steps_completed.append("landmark_decoded")
    
    logger.info(f"Decoded {len(landmarks)} landmark(s) for city '{city}': {landmarks}")
    
    return {
        "status": "success",
        "city": city,
        "landmarks": landmarks,  # Return all landmarks
        "landmark_count": len(landmarks),
        "step": "2_landmark_decoded",
        "next_step": "get_flight_numbers"
    }


@app.post("/call/get_flight_numbers")
async def get_flight_numbers(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Get flight numbers for multiple landmarks.
    """
    landmarks = request.get("landmarks", [])
    if not landmarks:
        raise HTTPException(status_code=400, detail="Landmarks list is required")
    
    if isinstance(landmarks, str):
        landmarks = [landmarks]  # Handle single landmark
    
    logger.info(f"Getting flight numbers for {len(landmarks)} landmark(s): {landmarks}")
    
    flight_results = []
    
    for landmark in landmarks:
        # Determine which endpoint to call based on landmark
        endpoint_url = FLIGHT_ENDPOINTS.get(landmark, FLIGHT_ENDPOINTS["default"])
        
        logger.info(f"Using endpoint for '{landmark}': {endpoint_url}")
        
        result = await _fetch_url_data(endpoint_url)
        
        if result["status"] == "success":
            # Extract flight number from response
            if result["content_type"] == "json":
                # The API returns {"success": true, "data": {"flightNumber": "7df399"}}
                api_response = result["data"]
                if isinstance(api_response, dict) and "data" in api_response:
                    flight_number = api_response["data"]["flightNumber"]
                elif isinstance(api_response, dict) and "flightNumber" in api_response:
                    flight_number = api_response["flightNumber"]
                else:
                    flight_number = str(api_response)
            else:
                flight_number = result["data"].strip()
            
            flight_results.append({
                "landmark": landmark,
                "flight_number": flight_number,
                "endpoint_used": endpoint_url,
                "status": "success"
            })
            
            logger.info(f"Retrieved flight number for '{landmark}': {flight_number}")
        else:
            flight_results.append({
                "landmark": landmark,
                "endpoint_used": endpoint_url,
                "status": "error",
                "error": result.get("message", "Unknown error")
            })
            logger.error(f"Failed to get flight number for '{landmark}': {result.get('message')}")
    
    # Store results
    challenge_state.flight_number = flight_results
    challenge_state.steps_completed.append("flight_numbers_retrieved")
    
    successful_flights = [f for f in flight_results if f["status"] == "success"]
    
    return {
        "status": "success",
        "landmarks": landmarks,
        "flight_results": flight_results,
        "successful_count": len(successful_flights),
        "total_count": len(landmarks),
        "step": "3_flight_numbers_retrieved",
        "challenge_completed": True
    }


@app.post("/call/get_flight_number")
async def get_flight_number(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Get the flight number based on the landmark.
    """
    landmark = request.get("landmark")
    if not landmark:
        raise HTTPException(status_code=400, detail="Landmark is required")
    logger.info(f"Getting flight number for landmark: {landmark}")
    
    # Determine which endpoint to call based on landmark
    endpoint_url = FLIGHT_ENDPOINTS.get(landmark, FLIGHT_ENDPOINTS["default"])
    
    logger.info(f"Using endpoint: {endpoint_url}")
    
    result = await _fetch_url_data(endpoint_url)
    
    if result["status"] == "success":
        # Extract flight number from response
        if result["content_type"] == "json":
            # The API returns {"success": true, "data": {"flightNumber": "7df399"}}
            api_response = result["data"]
            if isinstance(api_response, dict) and "data" in api_response:
                flight_number = api_response["data"]["flightNumber"]
            elif isinstance(api_response, dict) and "flightNumber" in api_response:
                flight_number = api_response["flightNumber"]
            else:
                flight_number = str(api_response)
        else:
            flight_number = result["data"].strip()
        
        challenge_state.flight_number = flight_number
        challenge_state.steps_completed.append("flight_number_retrieved")
        
        logger.info(f"Retrieved flight number: {flight_number}")
        
        return {
            "status": "success",
            "landmark": landmark,
            "flight_number": flight_number,
            "endpoint_used": endpoint_url,
            "step": "3_flight_number_retrieved",
            "challenge_completed": True
        }
    else:
        return {
            "status": "error",
            "message": f"Failed to get flight number for {landmark}: {result.get('message', 'Unknown error')}",
            "landmark": landmark,
            "endpoint_used": endpoint_url,
            "error_details": result
        }


@app.post("/call/solve_complete_challenge")
async def solve_complete_challenge(request: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Solve the complete HackRx challenge automatically.
    Executes all steps: get city, decode landmark, get flight number.
    
    Returns:
        Dictionary containing complete challenge solution
    """
    logger.info("Starting complete HackRx challenge solution...")
    
    challenge_state.reset()
    
    try:
        # Step 1: Get favorite city
        city_result = await get_favorite_city({})
        if city_result["status"] != "success":
            return {
                "status": "error",
                "step_failed": "get_favorite_city",
                "error": city_result["message"]
            }
        
        city = city_result["favorite_city"]
        
        # Step 2: Decode landmarks
        landmark_result = await decode_city_landmark({"city": city})
        if landmark_result["status"] != "success":
            return {
                "status": "error", 
                "step_failed": "decode_city_landmark",
                "error": landmark_result["message"],
                "city": city
            }
        
        landmarks = landmark_result["landmarks"]
        
        # Step 3: Get flight numbers for all landmarks
        flight_result = await get_flight_numbers({"landmarks": landmarks})
        if flight_result["status"] != "success":
            return {
                "status": "error",
                "step_failed": "get_flight_numbers", 
                "error": flight_result.get("message", "Failed to get flight numbers"),
                "city": city,
                "landmarks": landmarks
            }
        
        flight_results = flight_result["flight_results"]
        successful_flights = [f for f in flight_results if f["status"] == "success"]
        
        # Success!
        logger.info(f"Challenge completed successfully! Got {len(successful_flights)} flight number(s)")
        
        # Create response message
        if len(successful_flights) == 1:
            flight_number = successful_flights[0]["flight_number"]
            landmark = successful_flights[0]["landmark"]
            message = f"Sachin can return to the real world using flight {flight_number} (via {landmark})!"
        else:
            flight_list = [f"{f['flight_number']} (via {f['landmark']})" for f in successful_flights]
            message = f"Sachin has {len(successful_flights)} flight options to return to the real world: {', '.join(flight_list)}"
        
        return {
            "status": "success",
            "challenge_completed": True,
            "solution": {
                "favorite_city": city,
                "landmarks": landmarks,
                "flight_results": flight_results,
                "successful_flights": successful_flights
            },
            "steps_completed": challenge_state.steps_completed,
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Challenge solution failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Challenge solution failed: {str(e)}",
            "steps_completed": challenge_state.steps_completed
        }


@app.post("/call/get_challenge_status")
async def get_challenge_status(request: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Get the current status of the challenge solving process.
    
    Returns:
        Dictionary containing current challenge state and progress
    """
    return {
        "challenge_state": {
            "favorite_city": challenge_state.favorite_city,
            "landmark": challenge_state.landmark,
            "flight_number": challenge_state.flight_number,
            "steps_completed": challenge_state.steps_completed
        },
        "available_tools": [
            "fetch_url_data",
            "get_favorite_city", 
            "decode_city_landmark",
            "get_flight_number",
            "solve_complete_challenge",
            "get_challenge_status",
            "get_landmark_mappings"
        ],
        "challenge_progress": f"{len(challenge_state.steps_completed)}/3 steps completed"
    }


@app.post("/call/get_landmark_mappings")
async def get_landmark_mappings(request: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Get the complete landmark mappings from the parallel world.
    
    Returns:
        Dictionary containing all landmark-to-city mappings
    """
    return {
        "landmark_to_city": LANDMARK_MAPPINGS,
        "city_to_landmarks": CITY_TO_LANDMARK,
        "flight_endpoints": FLIGHT_ENDPOINTS,
        "total_landmarks": len(LANDMARK_MAPPINGS),
        "total_cities": len(CITY_TO_LANDMARK)
    }


if __name__ == "__main__":
    import sys
    
    # Default port
    port = 8001
    
    # Check if port provided as argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    logger.info(f"Starting HackRx Challenge Solver Server on port {port}")
    
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")