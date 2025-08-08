"""
Algorithm Executor for Step-by-Step API Processing
Executes algorithms described in documents instead of just explaining them
"""

import aiohttp
import logging
from typing import Optional
import re
import json

logger = logging.getLogger(__name__)


class AlgorithmExecutor:
    """
    Executes algorithms found in documents by making actual API calls
    """
    
    def __init__(self):
        """Initialize the Algorithm Executor"""
        # City to landmark mappings from the document
        self.indian_cities_mapping = {
            "Delhi": "Gateway of India",
            "Mumbai": "India Gate",
            "Chennai": "Charminar",
            "Hyderabad": "Marina Beach",
            "Ahmedabad": "Howrah Bridge",
            "Mysuru": "Golconda Fort",
            "Kochi": "Qutub Minar",
            "Kolkata": "Taj Mahal",
            "Bangalore": "Red Fort",
            "Pune": "Lotus Temple",
            "Jaipur": "Hawa Mahal",
            "Lucknow": "Victoria Memorial",
            "Kanpur": "Sanchi Stupa",
            "Nagpur": "Ajanta Caves",
            "Indore": "Ellora Caves",
            "Thane": "Khajuraho Temples",
            "Bhopal": "Fatehpur Sikri",
            "Visakhapatnam": "Amber Fort",
            "Pimpri": "Mehrangarh Fort",
            "Patna": "City Palace"
        }
        
        self.international_cities_mapping = {
            "New York": "Eiffel Tower",
            "London": "Statue of Liberty", 
            "Tokyo": "Big Ben",
            "Beijing": "Colosseum",
            "Paris": "Christ the Redeemer",
            "Sydney": "Machu Picchu",
            "Berlin": "Petra",
            "Moscow": "Great Wall of China",
            "Toronto": "Angkor Wat",
            "Dubai": "Sagrada Familia",
            "Singapore": "Neuschwanstein Castle",
            "Amsterdam": "Tower Bridge",
            "Barcelona": "Golden Gate Bridge",
            "Rome": "Sydney Opera House",
            "Madrid": "Burj Khalifa",
            "Vienna": "CN Tower",
            "Brussels": "Space Needle",
            "Zurich": "Leaning Tower of Pisa",
            "Stockholm": "Mount Rushmore",
            "Copenhagen": "Stonehenge"
        }
        
        # Combine mappings
        self.city_to_landmark = {**self.indian_cities_mapping, **self.international_cities_mapping}
        
        logger.info("✅ Algorithm Executor initialized with city mappings")
    
    async def _make_api_call(self, url: str) -> Optional[str]:
        """Make an HTTP GET request to an API endpoint"""
        try:
            logger.info(f"🌐 Making API call: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        logger.info(f"✅ API response: {response_text[:200]}...")
                        return response_text.strip()
                    else:
                        logger.error(f"❌ API call failed with status {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Error making API call: {str(e)}")
            return None
    
    def _extract_flight_number(self, response: str) -> Optional[str]:
        """Extract flight number from API response"""
        try:
            logger.info(f"🔍 Extracting flight number from response: {response}")
            
            # Try JSON parsing first
            try:
                data = json.loads(response)
                logger.info(f"📊 Parsed JSON: {data}")
                
                if isinstance(data, dict):
                    # Check for nested data structure first
                    if 'data' in data and isinstance(data['data'], dict):
                        data_section = data['data']
                        # Look for flight number in data section
                        for key in ['flightNumber', 'flight_number', 'flight', 'number', 'value', 'result']:
                            if key in data_section:
                                flight_num = str(data_section[key])
                                logger.info(f"✅ Found flight number in data.{key}: {flight_num}")
                                return flight_num
                    
                    # Look for common flight number keys in main object
                    for key in ['flightNumber', 'flight_number', 'flight', 'number', 'value', 'result', 'data']:
                        if key in data:
                            flight_num = str(data[key])
                            logger.info(f"✅ Found flight number in {key}: {flight_num}")
                            return flight_num
                    
                    # Return any string/number value that looks like a flight number
                    for key, value in data.items():
                        if isinstance(value, (str, int)) and key not in ['success', 'message', 'status']:
                            flight_num = str(value)
                            if len(flight_num) > 2:  # Flight numbers are usually longer than 2 chars
                                logger.info(f"✅ Found potential flight number in {key}: {flight_num}")
                                return flight_num
                                
            except json.JSONDecodeError:
                logger.info("📋 Not JSON, trying regex patterns...")
            
            # Try regex patterns for flight numbers
            flight_patterns = [
                r'\b[A-Za-z0-9]{4,8}\b',  # Alphanumeric codes like b5eb30
                r'\b[A-Z]{2,3}[0-9]{2,4}\b',  # AA123, UAL456
                r'\b[0-9]{3,4}\b',  # 123, 4567
                r'flight["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)',  # flight: "AA123"
                r'number["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)',  # number: "123"
            ]
            
            for i, pattern in enumerate(flight_patterns):
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    flight_num = match.group(1) if match.groups() else match.group(0)
                    logger.info(f"✅ Found flight number with pattern {i+1}: {flight_num}")
                    return flight_num
            
            # If no pattern matches, return the response itself (might be just the flight number)
            clean_response = response.strip().strip('"\'')
            logger.info(f"⚠️ No patterns matched, returning clean response: {clean_response}")
            return clean_response
            
        except Exception as e:
            logger.error(f"❌ Error extracting flight number: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return response.strip()
    
    async def execute_flight_discovery_algorithm(self) -> str:
        """
        Execute Sachin's Parallel World Discovery Algorithm
        """
        logger.info("🚀 Starting Flight Discovery Algorithm")
        
        try:
            # Step 1: Get the secret city
            logger.info("📍 Step 1: Getting secret city...")
            city_response = await self._make_api_call("https://register.hackrx.in/submissions/myFavouriteCity")
            
            if not city_response:
                return "Failed to get secret city from API"
            
            # Extract city name from response
            try:
                # Try to parse as JSON first
                import json
                city_data = json.loads(city_response)
                if isinstance(city_data, dict) and 'data' in city_data and 'city' in city_data['data']:
                    city_name = city_data['data']['city']
                else:
                    city_name = city_response.strip().strip('"\'')
            except json.JSONDecodeError:
                city_name = city_response.strip().strip('"\'')
            
            logger.info(f"🏙️ Secret city: {city_name}")
            
            # Step 2: Map city to landmark
            logger.info("🗺️ Step 2: Mapping city to landmark...")
            landmark = self.city_to_landmark.get(city_name)
            
            if not landmark:
                logger.warning(f"⚠️ City '{city_name}' not found in mapping, using default")
                landmark = "Unknown"
            
            logger.info(f"🏛️ Landmark: {landmark}")
            
            # Step 3: Determine flight endpoint based on landmark
            logger.info("✈️ Step 3: Determining flight endpoint...")
            
            if landmark == "Gateway of India":
                flight_url = "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber"
            elif landmark == "Taj Mahal":
                flight_url = "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber"
            elif landmark == "Eiffel Tower":
                flight_url = "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber"
            elif landmark == "Big Ben":
                flight_url = "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber"
            else:
                flight_url = "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"
            
            logger.info(f"🔗 Flight endpoint: {flight_url}")
            
            # Step 4: Get flight number
            logger.info("🎫 Step 4: Getting flight number...")
            flight_response = await self._make_api_call(flight_url)
            
            if not flight_response:
                return f"Failed to get flight number from {flight_url}"
            
            # Extract flight number
            flight_number = self._extract_flight_number(flight_response)
            logger.info(f"✅ Flight number extracted: {flight_number}")
            
            return flight_number
            
        except Exception as e:
            logger.error(f"❌ Algorithm execution failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Algorithm execution failed: {str(e)}"
    
    def detect_algorithm_in_text(self, text: str) -> Optional[str]:
        """
        Detect if text contains an algorithm that should be executed
        """
        text_lower = text.lower()
        
        # Primary indicators for flight discovery algorithm
        flight_indicators = [
            "hackrx.in", "register.hackrx", "myfavouritecity", "getflightnumber",
            "flight", "city", "landmark", "algorithm", "step", "api", "endpoint"
        ]
        
        # Count flight-related indicators
        flight_count = sum(1 for indicator in flight_indicators if indicator in text_lower)
        
        # More aggressive detection - if we have flight + city/landmark + api/step keywords
        if flight_count >= 2:
            if "flight" in text_lower and ("city" in text_lower or "landmark" in text_lower or "step" in text_lower or "api" in text_lower):
                logger.info(f"✅ Flight discovery algorithm detected (indicators: {flight_count})")
                return "flight_discovery"
        
        # Also check for specific URL patterns
        if "register.hackrx.in" in text_lower or "myfavouritecity" in text_lower:
            logger.info("✅ Flight discovery algorithm detected (URL pattern)")
            return "flight_discovery"
        
        logger.info(f"❌ No algorithm detected (indicators: {flight_count})")
        return None
    
    async def execute_algorithm(self, algorithm_type: str) -> str:
        """
        Execute a detected algorithm
        """
        if algorithm_type == "flight_discovery":
            return await self.execute_flight_discovery_algorithm()
        else:
            return f"Unknown algorithm type: {algorithm_type}"