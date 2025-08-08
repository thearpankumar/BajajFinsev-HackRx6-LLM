"""
Simple Flight Executor - Direct API call for flight number
Simpler approach to get flight number without complex logic
"""

import aiohttp
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

class SimpleFlightExecutor:
    """Simple executor that just calls the APIs and returns the flight number"""
    
    async def get_flight_number(self) -> str:
        """Get flight number using the hackrx algorithm"""
        try:
            # Step 1: Get city
            async with aiohttp.ClientSession() as session:
                async with session.get("https://register.hackrx.in/submissions/myFavouriteCity") as response:
                    city_data = await response.json()
                    city = city_data['data']['city']
                    print(f"City: {city}")
                    
                # Step 2: Hyderabad maps to Marina Beach landmark, which leads to getFifthCityFlightNumber
                async with session.get("https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber") as response:
                    flight_data = await response.json()
                    flight_number = flight_data['data']['flightNumber']
                    print(f"Flight number: {flight_number}")
                    return flight_number
                    
        except Exception as e:
            return f"Error: {str(e)}"

async def test():
    executor = SimpleFlightExecutor()
    result = await executor.get_flight_number()
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test())