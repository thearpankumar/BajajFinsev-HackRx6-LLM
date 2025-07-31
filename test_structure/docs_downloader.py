import requests
import PyPDF2
import tiktoken
import os

def download_and_calculate_total_tokens(urls, directory="sample_docs"):
    """
    Downloads PDF documents into a specified directory, calculates the token count
    for each, and returns the total token count.

    Args:
        urls (list): A list of URLs pointing to PDF documents.
        directory (str): The folder to save the downloaded files in.
    """
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: '{directory}'")

    downloaded_files = []
    total_tokens = 0

    # 2. Download all files from the URLs
    print("\n--- Downloading Files ---")
    for url in urls:
        try:
            # Extract a clean filename from the URL
            file_name = url.split('/')[-1].split('?')[0].replace('%20', '_')
            file_path = os.path.join(directory, file_name)

            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded_files.append(file_path)
            print(f"Successfully downloaded: {file_name}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")

    # 3. Calculate tokens for each downloaded file
    print("\n--- Calculating Tokens ---")
    
    # Use the cl100k_base encoding for models like gpt-4 and gpt-3.5-turbo
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Could not get tiktoken encoding. Please ensure 'tiktoken' is installed. Error: {e}")
        return

    for file_path in downloaded_files:
        try:
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                # Handle encrypted PDFs that can't be opened
                if reader.is_encrypted:
                    print(f"Skipping encrypted file: {os.path.basename(file_path)}")
                    continue
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

            # Count the tokens and add to total
            num_tokens = len(encoding.encode(text))
            total_tokens += num_tokens
            print(f"File: {os.path.basename(file_path):<70} | Tokens: {num_tokens}")

        except Exception as e:
            print(f"Could not process file {os.path.basename(file_path)}. Error: {e}")

    # 4. Print the final total
    print("\n-------------------------------------------------------------------------")
    print(f"Total estimated tokens for all documents: {total_tokens}")
    print("-------------------------------------------------------------------------")


# List of document URLs
document_urls = [
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
    "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
    "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
]

# Run the function
if __name__ == "__main__":
    download_and_calculate_total_tokens(document_urls)
