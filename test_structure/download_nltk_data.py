#!/usr/bin/env python3
"""
Download required NLTK data for the RAG system
"""

import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download all required NLTK data"""
    
    print("📥 Downloading NLTK data...")
    
    # List of required NLTK data
    required_data = [
        'punkt',
        'punkt_tab', 
        'stopwords'
    ]
    
    for data_name in required_data:
        try:
            print(f"  📦 Downloading {data_name}...")
            nltk.download(data_name, quiet=True)
            print(f"  ✅ {data_name} downloaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to download {data_name}: {str(e)}")
    
    print("✅ NLTK data download completed!")

if __name__ == "__main__":
    download_nltk_data()
