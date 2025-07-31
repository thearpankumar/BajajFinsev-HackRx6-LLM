import requests
import PyPDF2
import tiktoken
import os
import base64
from groq import Groq
from pdf2image import convert_from_path
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file or environment variables. Please set it.")

PDF_URL = "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
DOWNLOAD_DIR = "sample_docs"
OUTPUT_TEXT_FILE = "document_text_output.txt" # Generic output name
ENCODING_MODEL = "cl100k_base"
GROQ_MULTIMODAL_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 

# --- Function to Extract Text Directly from PDF ---
def extract_text_directly_from_pdf(pdf_path):
    """
    Extracts text from a PDF if it's text-based. Returns None if no text is found.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                print(f"File '{os.path.basename(pdf_path)}' is encrypted. Cannot extract text.")
                return None
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        if text.strip(): # Return text only if it's not empty
            return text
        else:
            return None # Indicate no extractable text was found
            
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred during direct text extraction from {pdf_path}: {e}")
        return None

# --- Function to Convert PDF Pages to Images ---
def convert_pdf_to_images(pdf_path, output_folder="pdf_images"):
    """Converts each page of a PDF into an image file."""
    print(f"--- Converting PDF to images: {os.path.basename(pdf_path)} ---")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        pdf_images = convert_from_path(pdf_path, dpi=200, output_folder=output_folder, fmt='png', thread_count=4)
        
        image_paths = []
        for i, img in enumerate(pdf_images):
            img_path = os.path.join(output_folder, f"page_{i+1}.png")
            img.save(img_path, 'PNG')
            image_paths.append(img_path)
        print(f"Converted {len(image_paths)} pages to images in '{output_folder}'.")
        return image_paths
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        print("Please ensure Poppler is installed and in your PATH if you encounter issues.")
        return None

# --- Function to Encode Image to Base64 ---
def encode_image_to_base64(image_path):
    """Encodes an image file into a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# --- Groq OCR Function using Groq SDK ---
def get_ocr_text_from_groq(image_paths, groq_api_key, model_name):
    """
    Performs OCR on a list of image paths using Groq's multimodal API.
    """
    if not image_paths:
        print("No image paths provided for OCR.")
        return None

    client = Groq(api_key=groq_api_key)
    all_extracted_text = ""
    
    print(f"\n--- Performing OCR with Groq API ({model_name}) ---")

    for i, img_path in enumerate(image_paths):
        base64_image = encode_image_to_base64(img_path)
        if not base64_image:
            continue 

        print(f"Processing image: {os.path.basename(img_path)}...")

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Provide the output clearly, indicating if any part is illegible or if the image is empty. If there's no text, say 'No text detected'."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ],
                    }
                ],
                model=model_name,
                max_tokens=1024, 
                temperature=0.1, 
            )

            page_text = chat_completion.choices[0].message.content
            all_extracted_text += f"\n--- Page {i+1} OCR Result ---\n{page_text}\n"
            print(f"Successfully processed page {i+1}.")

        except Exception as e:
            print(f"Error processing image {os.path.basename(img_path)} with Groq API: {e}")
            all_extracted_text += f"\n--- Page {i+1} OCR Result ---\n[OCR failed for this page: {e}]\n"

    return all_extracted_text

# --- Token Counting Function (same as before) ---
def count_tokens_in_file(file_path, encoding_name):
    """
    Counts the tokens in a given text file using a specified encoding.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError:
        print(f"Error: Encoding '{encoding_name}' not found. Please check tiktoken documentation.")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        tokens = encoding.encode(text)
        return len(tokens)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while counting tokens in {file_path}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # Create directories
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    IMAGE_EXTRACTION_DIR = "pdf_images" # Folder to store intermediate images
    if not os.path.exists(IMAGE_EXTRACTION_DIR):
        os.makedirs(IMAGE_EXTRACTION_DIR)

    pdf_filename = PDF_URL.split('/')[-1].split('?')[0].replace('%20', '_')
    pdf_path = os.path.join(DOWNLOAD_DIR, pdf_filename)

    # Download the PDF
    print(f"--- Downloading PDF: {pdf_filename} ---")
    try:
        response = requests.get(PDF_URL, stream=True)
        response.raise_for_status() 
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {pdf_filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {PDF_URL}: {e}")
        exit()

    # --- Core Logic: Attempt Direct Text Extraction First ---
    print(f"\n--- Attempting direct text extraction from PDF: {pdf_filename} ---")
    extracted_text = extract_text_directly_from_pdf(pdf_path)

    final_text_to_tokenize = ""

    if extracted_text:
        print("Found extractable text in PDF. Using direct text extraction.")
        final_text_to_tokenize = extracted_text
    else:
        print("No extractable text found via direct extraction. Proceeding with OCR using Groq.")
        # 2. Convert PDF to images for OCR
        image_paths = convert_pdf_to_images(pdf_path, IMAGE_EXTRACTION_DIR)

        if image_paths:
            # 3. Get OCR text from Groq API using the images
            llm_ocr_output = get_ocr_text_from_groq(image_paths, GROQ_API_KEY, GROQ_MULTIMODAL_MODEL)
            if llm_ocr_output:
                final_text_to_tokenize = llm_ocr_output
            else:
                print("Groq OCR process failed to return any text.")
        else:
            print("PDF to image conversion failed. Cannot perform Groq OCR.")

    # 4. Save the final text (either directly extracted or from Groq OCR) to a file
    if final_text_to_tokenize:
        with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(final_text_to_tokenize)
        print(f"Final document text saved to: {OUTPUT_TEXT_FILE}")

        # 5. Calculate token count of the saved text file
        print(f"\n--- Calculating Token Count for: {OUTPUT_TEXT_FILE} ---")
        token_count = count_tokens_in_file(OUTPUT_TEXT_FILE, ENCODING_MODEL)

        if token_count is not None:
            print(f"The file '{OUTPUT_TEXT_FILE}' contains approximately {token_count} tokens.")
            print(f"(Using encoding: '{ENCODING_MODEL}')")
    else:
        print("No text was obtained from the document. Cannot save or count tokens.")

    # --- Cleanup (Optional) ---
    # Uncomment the following lines to automatically clean up files and directories.
    # print("\n--- Cleaning up files ---")
    # if os.path.exists(pdf_path):
    #     os.remove(pdf_path)
    #     print(f"Removed: '{pdf_path}'.")
    # if image_paths:
    #     for img_path in image_paths:
    #         if os.path.exists(img_path):
    #             os.remove(img_path)
    #             print(f"Removed: '{img_path}'.")
    # if os.path.exists(IMAGE_EXTRACTION_DIR) and not os.listdir(IMAGE_EXTRACTION_DIR):
    #     os.rmdir(IMAGE_EXTRACTION_DIR)
    #     print(f"Removed empty directory: '{IMAGE_EXTRACTION_DIR}'.")
