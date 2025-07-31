

import fitz  # PyMuPDF
import os
import time
import concurrent.futures

# --- Configuration ---
# Path to your large PDF file
PDF_PATH = "path/to/your/20mb_document.pdf" 
# Directory to save the output images
OUTPUT_DIR = "pdf_images"
# The resolution of the output images. Lower is faster but lower quality.
# 150 is a good balance for OCR. 72 is very fast. 300 is high quality.
DPI = 150 

def render_page(args):
    """
    Renders a single page of a PDF to an image.
    Designed to be called by a parallel processor.
    """
    doc_path, page_num, output_dir, dpi = args
    try:
        # Each worker must open its own document instance
        doc = fitz.open(doc_path)
        page = doc.load_page(page_num)
        
        # Render page to an image (pixmap)
        pix = page.get_pixmap(dpi=dpi)
        
        output_path = os.path.join(output_dir, f"page_{page_num + 1:04d}.png")
        pix.save(output_path)
        
        doc.close()
        return f"Rendered page {page_num + 1}"
    except Exception as e:
        return f"Error on page {page_num + 1}: {e}"

def main():
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at '{PDF_PATH}'")
        print("Please update the 'PDF_PATH' variable in the script.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Starting conversion of '{os.path.basename(PDF_PATH)}'...")
    start_time = time.time()

    # Open the document once to get the page count
    doc = fitz.open(PDF_PATH)
    num_pages = len(doc)
    doc.close()
    
    if num_pages == 0:
        print("PDF has no pages.")
        return

    # Create a list of arguments for each page rendering task
    tasks = [(PDF_PATH, i, OUTPUT_DIR, DPI) for i in range(num_pages)]

    # Use a ThreadPoolExecutor to render pages in parallel
    # The number of workers will default to a reasonable number for your system
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # The list() call ensures that we wait for all futures to complete.
        list(executor.map(render_page, tasks))

    end_time = time.time()
    
    print("\n--- Conversion Complete ---")
    print(f"Processed {num_pages} pages.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    if num_pages > 0:
        print(f"Average time per page: {((end_time - start_time) / num_pages) * 1000:.2f} ms.")
    print(f"Images saved in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()

