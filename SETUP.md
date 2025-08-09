# BajajFinsev RAG System - Conda-Only Setup

## 🎯 Simple Conda-Only Package Management

This project now uses **conda-only** package management to avoid conflicts and simplify dependency resolution.

## 📋 Prerequisites

- Anaconda or Miniconda installed
- NVIDIA GPU with CUDA 12.1+ support (optional, falls back to CPU)
- Git
- Tesseract OCR with Malayalam support (for bilingual OCR)

## 🚀 Installation

### 1. Install System Dependencies
```bash
# Install Tesseract OCR with Malayalam support (Ubuntu/Debian)
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-mal tesseract-ocr-eng

# For other systems:
# CentOS/RHEL: sudo yum install tesseract tesseract-langpack-mal tesseract-langpack-eng
# macOS: brew install tesseract tesseract-lang
```

### 2. Create Environment
```bash
# Create the conda environment with core dependencies
conda env create -f environment.yml

# Activate the environment
conda activate gpu-genai

# Install PyTorch 2.1.2 with CUDA 12.1 support (required for transformers)
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python packages via pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Test basic functionality
python -c "
import torch
import faiss
import fastapi
import transformers
print('✅ PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('✅ FAISS:', faiss.__version__, '| GPUs:', faiss.get_num_gpus())
print('✅ FastAPI imported successfully')
print('✅ Transformers:', transformers.__version__)
"
```

### 4. Run the Server
```bash
# Start the development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

## 📦 Package Management

### Hybrid Package Management
- **Conda**: System dependencies (PyTorch CUDA, FAISS GPU, NumPy, SciPy)
- **Pip**: Python packages (APIs, document processing, monitoring)
- **System**: Tesseract OCR with Malayalam language pack

### Why This Approach?
1. **GPU Support**: Conda handles CUDA dependencies (PyTorch, FAISS)
2. **Malayalam OCR**: System Tesseract provides proper Malayalam support
3. **No EasyOCR Conflicts**: Eliminates numpy/scipy compatibility issues
4. **Proven Setup**: Tested combination that works reliably

## 🔧 Development Commands

```bash
# Activate environment
conda activate gpu-genai

# Run tests
pytest

# Format code
black src/
ruff check src/ --fix

# Type checking
mypy src/

# Start server
uvicorn src.main:app --reload
```

## 📊 Environment Details

- **Python**: 3.11
- **CUDA**: 12.1 (with fallback to CPU)
- **NumPy**: 1.26.4 (FAISS compatible)
- **FAISS**: 1.11.0 (GPU accelerated)
- **PyTorch**: 2.1.2 (CUDA 12.1)
- **OCR**: Tesseract with Malayalam + English support

## 🔍 Monitoring

- **Health Check**: `GET /api/v1/hackrx/health`
- **Metrics**: `GET /metrics` (Prometheus format)
- **GPU Status**: Included in health check response

## 🐳 Docker Alternative

If you prefer Docker:
```bash
# Build and run with docker-compose
docker-compose up --build
```

## 📝 Notes

- **Malayalam OCR**: Full bilingual support via Tesseract OCR
- **FAISS GPU**: 1 GPU detected and working
- **EasyOCR**: Removed to eliminate compatibility conflicts
- **Language Detection**: Malayalam-English cross-lingual capabilities preserved
- **Translation**: Google Translate and Azure AI translation services available

## 🆘 Troubleshooting

### PyTorch Version Issues
```bash
# If transformers complain about PyTorch version
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Tesseract Malayalam Issues
```bash
# Check if Malayalam language pack is installed
tesseract --list-langs | grep mal

# Install Malayalam language pack (Ubuntu/Debian)
sudo apt install tesseract-ocr-mal

# Test Malayalam OCR
echo "മലയാളം ടെക്‌സ്റ്റ്" | tesseract stdin stdout -l mal
```

### FAISS GPU Issues
```bash
# Check GPU availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import faiss; print(faiss.get_num_gpus())"
```

### Environment Issues
```bash
# Recreate environment
conda env remove -n gpu-genai
conda env create -f environment.yml
conda activate gpu-genai
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```