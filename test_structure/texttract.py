

import boto3
import requests
import os
import time
import io
from dotenv import load_dotenv
from pdf2image import convert_from_path

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Load configuration from environment variables
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# The URL of the document you want to process.
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"

# --- Main Workflow Functions ---

def download_file(url, download_dir="."):
    """Downloads a file from a URL to a local temporary directory."""
    try:
        filename = "temp_document.pdf"
        local_path = os.path.join(download_dir, filename)

        print(f"--- Downloading document from URL... ---")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded to: {local_path}")
        return local_path
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not download file. Reason: {e}")
        return None

def perform_fast_textract_ocr(local_pdf_path, textract_client):
    """
    Performs fast, synchronous OCR on a local PDF file.
    Converts each page to an image and sends it directly to Textract.
    """
    print("\n--- Starting Fast Synchronous Textract OCR ---")
    full_text = ""
    
    try:
        # 1. Convert PDF pages to a list of in-memory images
        print("Converting PDF to images...")
        images = convert_from_path(local_pdf_path)
        print(f"PDF has {len(images)} pages.")
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        print("Please ensure 'poppler' is installed on your system.")
        print("  - On Debian/Ubuntu: sudo apt-get install poppler-utils")
        print("  - On macOS (Homebrew): brew install poppler")
        return None

    # 2. Process each page image with Textract's synchronous API
    for i, page_image in enumerate(images):
        print(f"Processing page {i + 1}/{len(images)}...")
        
        # Convert PIL image to bytes
        with io.BytesIO() as image_bytes:
            page_image.save(image_bytes, format="PNG")
            image_bytes = image_bytes.getvalue()

        try:
            # Call Textract's synchronous endpoint
            response = textract_client.detect_document_text(
                Document={'Bytes': image_bytes}
            )
            
            # Extract text from the response
            for item in response["Blocks"]:
                if item["BlockType"] == "LINE":
                    full_text += item["Text"] + "\n"
        except Exception as e:
            print(f"  Error processing page {i + 1} with Textract: {e}")
            continue
            
    print("--- OCR Processing Complete ---")
    return full_text

def cleanup(local_path):
    """Deletes the local file."""
    print("\n--- Cleaning up resources ---")
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"Deleted local file: {local_path}")

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # Check if all required environment variables are set
    if not all([AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        print("Error: Please ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are set in your .env file.")
        exit()

    # Initialize AWS Textract client
    textract_client = boto3.client(
        "textract",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    # 1. Download the document
    local_path = download_file(DOCUMENT_URL)
    if not local_path:
        exit()

    # 2. Perform fast OCR
    extracted_text = perform_fast_textract_ocr(local_path, textract_client)
    
    # 3. Process and print results
    if extracted_text:
        print("\n--- Extracted Text ---")
        print(extracted_text)
        print("--- End of Extracted Text ---")
    else:
        print("\nCould not extract text from the document.")

    # 4. Clean up the downloaded file
    cleanup(local_path)
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
    print("Workflow complete.")
