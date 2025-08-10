# 🚀 Getting Started with BajajFinsev RAG System

## 📋 For New Users - Complete Setup Guide

This guide will help you set up and run the **Advanced RAG System with HackRx Challenge Integration** from scratch.

## 🎯 What This System Does

- **Advanced RAG Pipeline**: Analyzes documents and answers questions using AI
- **HackRx Challenge Solver**: Automatically detects and solves the "Sachin's Parallel World" challenge
- **Intelligent Detection**: Knows when to use normal RAG vs challenge solving
- **Multi-format Support**: PDF, DOCX, XLSX, images, and more

## 📋 Prerequisites

### 1. System Requirements
- **Linux/macOS/Windows** (Linux preferred)
- **Python 3.11** or newer
- **NVIDIA GPU** (optional, falls back to CPU)
- **8GB+ RAM** (16GB+ recommended)

### 2. Software Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install git python3 python3-pip tesseract-ocr tesseract-ocr-mal

# macOS (with Homebrew)
brew install git python tesseract tesseract-lang

# Windows (with Chocolatey)
choco install git python tesseract
```

## 🛠️ Installation Steps

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd BajajFinsev
```

### Step 2: Install Conda/Miniconda
If you don't have conda installed:
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### Step 3: Create Environment
```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate gpu-genai

# Install PyTorch with CUDA support (for GPU acceleration)
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install all other requirements
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "
import torch
import faiss
import fastapi
import transformers
print('✅ PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())
print('✅ All dependencies installed successfully!')
"
```

## 🚀 Running the System

### Option 1: Quick Start (Recommended for Beginners)

#### 1. Start the Challenge Solver Server
```bash
# Make sure you're in the project directory
cd /path/to/BajajFinsev

# Activate the environment
conda activate gpu-genai

# Start the challenge solver (handles HackRx PDF challenges)
python3 mcp_servers/challenge_solver.py 8004

# You should see:
# INFO: Starting HackRx Challenge Solver Server on port 8004
# INFO: Uvicorn running on http://0.0.0.0:8004
```

Keep this terminal open - this is your **Challenge Solver Server**.

#### 2. Start the Main RAG Server (New Terminal)
```bash
# Open a new terminal
cd /path/to/BajajFinsev
conda activate gpu-genai

# Start the main RAG server
python3 src/main.py

# You should see:
# 🚀 Initializing BajajFinsev Advanced RAG System...
# ✅ System configured with Advanced RAG mode
# INFO: Uvicorn running on http://0.0.0.0:8000
```

Keep this terminal open - this is your **Main RAG Server**.

### Option 2: Using Startup Scripts

#### 1. Make scripts executable
```bash
chmod +x start_mcp_server.sh
```

#### 2. Start the challenge server
```bash
./start_mcp_server.sh 8004
```

#### 3. Start the main server (in another terminal)
```bash
conda activate gpu-genai
python3 src/main.py
```

## 🧪 Testing the System

### Test 1: Basic Health Check
```bash
# Test main server
curl http://localhost:8000/

# Test challenge server
curl http://localhost:8004/
```

### Test 2: HackRx Challenge (The Magic!)
```bash
# Test the challenge solver directly
curl -X POST http://localhost:8004/call/solve_complete_challenge \
     -H "Content-Type: application/json" \
     -d "{}"

# Expected output:
# {"status":"success","message":"Sachin can return to the real world using flight xxxxx (via Eiffel Tower)!"}
```

### Test 3: Full RAG Pipeline with Challenge Detection
```bash
# Replace YOUR_API_KEY with a test key (check src/core/config.py for default)
curl -X POST http://localhost:8000/api/v1/hackrx/run \
     -H "Authorization: Bearer your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "path/to/FinalRound4SubmissionPDF.pdf",
       "questions": [
         "What flight number will take Sachin back to the real world?",
         "How can Sachin escape the parallel world?"
       ]
     }'
```

## 🎯 Understanding the System Architecture

### 1. Main RAG Server (Port 8000)
- **Purpose**: Handles document analysis and Q&A
- **Features**: GPU acceleration, multi-language support, intelligent routing
- **Endpoints**: `/api/v1/hackrx/run`, `/api/v1/hackrx/health`

### 2. Challenge Solver Server (Port 8004)
- **Purpose**: Solves HackRx parallel world challenges
- **Features**: Automatic API calls, landmark mapping, flight number retrieval
- **Endpoints**: `/call/solve_complete_challenge`, `/call/get_favorite_city`

### 3. Intelligent Detection
- **How it works**: Analyzes user queries for challenge patterns
- **Triggers on**: Keywords like "flight number", "parallel world", "Sachin"
- **Result**: Automatic routing to challenge solver

## 📚 Common Use Cases

### Use Case 1: Regular Document Analysis
```bash
# Ask questions about normal documents
curl -X POST http://localhost:8000/api/v1/hackrx/run \
     -H "Authorization: Bearer your_api_key" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://example.com/document.pdf",
       "questions": ["What is the main topic of this document?"]
     }'
```

### Use Case 2: HackRx Challenge (Automatic!)
```bash
# Ask challenge-related questions - system auto-detects!
curl -X POST http://localhost:8000/api/v1/hackrx/run \
     -H "Authorization: Bearer your_api_key" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "path/to/FinalRound4SubmissionPDF.pdf",
       "questions": ["What is my flight number to escape the parallel world?"]
     }'

# System automatically:
# 1. Detects this is a challenge question
# 2. Calls the challenge solver
# 3. Returns: "Sachin can return using flight XXXXX"
```

## 🔧 Configuration

### API Keys
Edit `src/core/config.py` to set your API key:
```python
API_KEY = "your_secure_api_key_here"
```

### Challenge Server URL
If you change the challenge server port, update `src/services/intelligent_challenge_handler.py`:
```python
def __init__(self, mcp_server_url: str = "http://localhost:8004"):
```

### GPU Settings
The system auto-detects GPU. To force CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""
python3 src/main.py
```

## 🚨 Troubleshooting

### Problem: Challenge server won't start (port in use)
```bash
# Find and kill process using port 8004
sudo lsof -i :8004
sudo kill -9 <PID>

# Or use a different port
python3 mcp_servers/challenge_solver.py 8005
# Then update intelligent_challenge_handler.py accordingly
```

### Problem: GPU out of memory
```bash
# Use CPU mode
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch size in config
```

### Problem: Dependencies not found
```bash
# Reinstall in clean environment
conda deactivate
conda env remove -n gpu-genai
conda env create -f environment.yml
conda activate gpu-genai
pip install -r requirements.txt
```

### Problem: Challenge not detected
Check that:
1. Challenge server is running on correct port
2. Query contains challenge keywords: "flight", "parallel world", "Sachin"
3. Document path includes the PDF or challenge-related content

## 📈 Performance Tips

### For Better Performance:
```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Increase worker processes
# Edit src/core/config.py and increase parallel workers
```

### For Lower Memory Usage:
```bash
# Use CPU mode
export CUDA_VISIBLE_DEVICES=""

# Reduce document size limits
# Edit config files to reduce max document size
```

## 🎉 Success! You're Ready

If both servers are running and tests pass, you have successfully set up:
- ✅ **Advanced RAG System** for document analysis
- ✅ **HackRx Challenge Solver** for automatic challenge detection
- ✅ **Intelligent Routing** between normal Q&A and challenge solving
- ✅ **Multi-format Document Processing** with GPU acceleration

## 🔗 Next Steps

1. **Try the Web Interface**: Build a simple frontend to interact with the APIs
2. **Add More Challenges**: Extend the challenge solver for other competition problems
3. **Scale Up**: Use Docker for production deployment
4. **Customize**: Modify the system for your specific use cases

## 📞 Need Help?

- **Check Logs**: Both servers output detailed logs for debugging
- **Test Individual Components**: Use the curl commands above to test each part
- **Review Configuration**: Double-check ports, API keys, and file paths

The system is designed to be **intelligent and user-friendly** - it automatically detects what kind of question you're asking and responds appropriately!