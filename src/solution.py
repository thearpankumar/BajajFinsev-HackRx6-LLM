
import os

import requests

# Data from the PDF, mapping current location to the landmark
LANDMARK_MAP = {
    # Indian Cities
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate",
    "Chennai": "Charminar",
    "Hyderabad": "Marina Beach",
    "Ahmedabad": "Howrah Bridge",
    "Mysuru": "Golconda Fort",
    "Kochi": "Qutub Minar",
    "Hyderabad": "Taj Mahal",
    "Pune": "Meenakshi Temple",
    "Nagpur": "Lotus Temple",
    "Chandigarh": "Mysore Palace",
    "Kerala": "Rock Garden",
    "Bhopal": "Victoria Memorial",
    "Varanasi": "Vidhana Soudha",
    "Jaisalmer": "Sun Temple",
    "Pune": "Golden Temple",

    # International Cities
    "New York": "Eiffel Tower",
    "London": "Statue of Liberty",
    "Tokyo": "Big Ben",
    "Beijing": "Colosseum",
    "London": "Sydney Opera House",
    "Bangkok": "Christ the Redeemer",
    "Toronto": "Burj Khalifa",
    "Dubai": "CN Tower",
    "Amsterdam": "Petronas Towers",
    "Cairo": "Leaning Tower of Pisa",
    "San Francisco": "Mount Fuji",
    "Berlin": "Niagara Falls",
    "Barcelona": "Louvre Museum",
    "Moscow": "Stonehenge",
    "Seoul": "Sagrada Familia",
    "Cape Town": "Acropolis",
    "Istanbul": "Big Ben",
    "Riyadh": "Machu Picchu",
    "Paris": "Taj Mahal",
    "Dubai Airport": "Moai Statues",
    "Singapore": "Christchurch Cathedral",
    "Jakarta": "The Shard",
    "Vienna": "Blue Mosque",
    "Kathmandu": "Neuschwanstein Castle",
    "Los Angeles": "Buckingham Palace",
    "Mumbai": "Space Needle",
    "Seoul": "Times Square",
}

# Endpoints from the PDF
BASE_URL = "https://register.hackrx.in/teams/public/flights"
GET_CITY_URL = "https://register.hackrx.in/submissions/myFavouriteCity"

FLIGHT_PATHS = {
    "Gateway of India": f"{BASE_URL}/getFirstCityFlightNumber",
    "Taj Mahal": f"{BASE_URL}/getSecondCityFlightNumber",
    "Eiffel Tower": f"{BASE_URL}/getThirdCityFlightNumber",
    "Big Ben": f"{BASE_URL}/getFourthCityFlightNumber",
    "Other": f"{BASE_URL}/getFifthCityFlightNumber",
}

def get_secret_city():
    """Step 1: Query the Secret City"""
    try:
        response = requests.get(GET_CITY_URL, headers={
            "Authorization": f"Bearer {os.environ.get('SECRET_TOKEN')}"
        })
        response.raise_for_status()
        return response.json().get("data", {}).get("city")
    except requests.exceptions.RequestException as e:
        print(f"Error getting secret city: {e}")
        return None

def decode_city(city_name):
    """Step 2: Decode the City"""
    return LANDMARK_MAP.get(city_name)

def choose_flight_path(landmark):
    """Step 3: Choose Your Flight Path"""
    if landmark in FLIGHT_PATHS:
        return FLIGHT_PATHS[landmark]
    return FLIGHT_PATHS["Other"]

def get_flight_number(flight_path_url):
    """Step 4: Get the flight number"""
    try:
        response = requests.get(flight_path_url, headers={
            "Authorization": f"Bearer {os.environ.get('SECRET_TOKEN')}"
        })
        response.raise_for_status()
        # Assuming the flight number is also in a nested 'data' object
        data = response.json().get("data", {})
        return data.get("flightNumber")
    except requests.exceptions.RequestException as e:
        print(f"Error getting flight number: {e}")
        return None

if __name__ == "__main__":

    city = get_secret_city()
    if city:
        print(f"Step 1: Got secret city: {city}")

        # Step 2
        landmark = decode_city(city)
        if landmark:
            print(f"Step 2: Decoded landmark: {landmark}")

            # Step 3
            flight_path = choose_flight_path(landmark)
            print(f"Step 3: Chosen flight path URL: {flight_path}")

            # Step 4
            flight_number = get_flight_number(flight_path)
            if flight_number:
                print(f"Step 4: Final Flight Number: {flight_number}")
            else:
                print("Could not retrieve the flight number.")
        else:
            print(f"Could not find a landmark for the city: {city}")
    else:
        print("Could not retrieve the secret city.")

