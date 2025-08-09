#!/usr/bin/env python3
"""
Debug script to identify the exact source of 'Using CPU' warnings
"""

import os
import sys
import warnings
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Suppress all warnings first
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

print("🔍 Testing individual components for 'Using CPU' warnings...")

# Test 1: PyTorch
print("\n1. Testing PyTorch CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)
    else:
        print("   ❌ CUDA not available")
except Exception as e:
    print(f"   ❌ PyTorch error: {e}")

# Test 2: FAISS
print("\n2. Testing FAISS GPU...")
try:
    import faiss
    stdout_buf = StringIO()
    stderr_buf = StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        gpu_count = faiss.get_num_gpus()
    
    stdout_content = stdout_buf.getvalue()
    stderr_content = stderr_buf.getvalue()
    
    print(f"   GPU count: {gpu_count}")
    if stdout_content:
        print(f"   STDOUT: {stdout_content}")
    if stderr_content:
        print(f"   STDERR: {stderr_content}")
        
except Exception as e:
    print(f"   ❌ FAISS error: {e}")

# Test 3: SentenceTransformers
print("\n3. Testing SentenceTransformers...")
try:
    stdout_buf = StringIO()
    stderr_buf = StringIO()
    
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        from sentence_transformers import SentenceTransformer
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('intfloat/multilingual-e5-base', device=device)
    
    stdout_content = stdout_buf.getvalue()
    stderr_content = stderr_buf.getvalue()
    
    if stdout_content and 'Using CPU' in stdout_content:
        print(f"   ⚠️ Found 'Using CPU' in STDOUT: {stdout_content}")
    if stderr_content and 'Using CPU' in stderr_content:
        print(f"   ⚠️ Found 'Using CPU' in STDERR: {stderr_content}")
        
    if not stdout_content and not stderr_content:
        print("   ✅ No output captured")
    
    # Check actual device
    model_device = next(model.parameters()).device
    print(f"   Model device: {model_device}")
    
except Exception as e:
    print(f"   ❌ SentenceTransformers error: {e}")

print("\n✅ Debug test complete")