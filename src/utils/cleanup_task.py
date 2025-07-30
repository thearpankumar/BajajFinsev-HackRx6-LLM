# src/utils/cleanup_task.py
import os
import sys
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging for the cleanup task
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='cleanup.log',  # Log to a separate file
    filemode='a'
)

def main():
    """
    Initializes the Gemini client and deletes a file based on a
    command-line argument.
    """
    # Ensure a file name is provided
    if len(sys.argv) < 2:
        logging.error("No file name provided for cleanup.")
        sys.exit(1)
        
    file_name = sys.argv[1]
    logging.info(f"Starting cleanup process for file: {file_name}")

    try:
        # Load environment variables from .env file
        # This is crucial because this script runs in a separate process
        # and won't inherit the environment from the FastAPI app.
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY not found in environment.")
            sys.exit(1)
            
        # Initialize the client with the API key
        genai.configure(api_key=api_key)
        
        # Perform the blocking delete operation
        genai.delete_file(name=file_name)
        
        logging.info(f"Successfully deleted file: {file_name}")
        
    except Exception as e:
        logging.error(f"Failed to delete file {file_name}: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
