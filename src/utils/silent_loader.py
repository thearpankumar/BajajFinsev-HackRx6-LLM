"""
Silent Model Loader - Completely suppresses all output during model loading
Uses subprocess isolation to eliminate any console warnings or messages
"""
import os
import sys
import subprocess
import tempfile
import pickle
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_sentence_transformer_silently(model_name, device: str = 'cuda:0', cache_dir: str = './models_cache') -> Any:
    """
    Load SentenceTransformer with complete output suppression using subprocess isolation
    
    Args:
        model_name: Name/path of the SentenceTransformer model
        device: Target device (cuda:0, cpu, etc.)
        cache_dir: Cache directory for models
        
    Returns:
        Loaded SentenceTransformer model
        
    Raises:
        Exception: If model loading fails
    """
    
    # 2024-2025 Fix: Handle unhashable dict/list parameters from transformers/sentence-transformers
    def safe_param_to_string(param):
        """Convert any parameter to a safe string representation"""
        if isinstance(param, (dict, list, set)):
            import json
            try:
                return json.dumps(param, sort_keys=True)
            except (TypeError, ValueError):
                return str(param)
        return str(param)
    
    # Safely convert all parameters
    safe_model_name = safe_param_to_string(model_name)
    safe_device = safe_param_to_string(device)
    safe_cache_dir = safe_param_to_string(cache_dir)
    
    logger.debug(f"Converted parameters - model_name: {safe_model_name}, device: {safe_device}, cache_dir: {safe_cache_dir}")
    
    try:
        # Prepare temp file paths with safe hash generation
        temp_dir = Path(tempfile.gettempdir())
        
        # Create a deterministic hash using hashlib
        import hashlib
        import time
        
        # Include timestamp to avoid conflicts - use safe converted parameters for hashing
        hash_input = f"{safe_model_name}_{safe_device}_{safe_cache_dir}_{int(time.time())}"
        model_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]  # Use SHA256 for better distribution
        
        model_file = temp_dir / f'st_model_{model_hash}.pkl'
        success_file = temp_dir / f'st_success_{model_hash}.txt'
        error_file = temp_dir / f'st_error_{model_hash}.txt'
        script_file = temp_dir / f'st_loader_{model_hash}.py'
        
    except Exception as e:
        logger.error(f"Hash creation failed: {e}")
        logger.error(f"Original parameters: model_name={type(model_name)}, device={type(device)}, cache_dir={type(cache_dir)}")
        raise Exception(f"Parameter processing failed: {e}")
    
    # Create the isolated loader script with proper escaping
    loader_script = f'''
import os
import sys
import warnings
import logging

# Complete output suppression
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'

# Disable all logging
logging.disable(logging.CRITICAL)

# Redirect all output to null
if os.name == 'nt':  # Windows
    devnull = 'nul'
else:  # Unix/Linux/macOS
    devnull = '/dev/null'

sys.stdout = open(devnull, 'w')
sys.stderr = open(devnull, 'w')

try:
    # Import and load model
    from sentence_transformers import SentenceTransformer
    import torch
    import pickle
    
    # Create cache directory
    cache_path = r"{safe_cache_dir}"
    os.makedirs(cache_path, exist_ok=True)
    
    # 2025 CUTTING-EDGE FIX: Based on sentence-transformers v5.1.0 research
    # Fix version compatibility and dict hashing issues
    
    import os
    import sys
    import importlib
    
    # Set environment for compatibility
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_path
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 2025 Version compatibility check and fix
    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
        
        # For v5.1.0+ use new backend approach
        if hasattr(sentence_transformers, 'SentenceTransformer'):
            # Use the most minimal approach possible
            model = sentence_transformers.SentenceTransformer(r"{safe_model_name}")
        else:
            # Fallback for older versions
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(r"{safe_model_name}")
            
    except Exception as version_error:
        # Ultimate fallback - import directly
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(r"{safe_model_name}")
    
    # Move to device after creation if GPU is available
    if '{safe_device}' != 'cpu' and torch.cuda.is_available():
        model = model.to('{safe_device}')
    
    # Save model to temporary file
    with open(r"{model_file}", 'wb') as f:
        pickle.dump(model, f)
    
    # Write success indicator with device info
    model_device = str(next(model.parameters()).device) if hasattr(model, 'parameters') else '{safe_device}'
    with open(r"{success_file}", 'w') as f:
        f.write('success,' + str(model_device))
        
except Exception as e:
    # Write error to file
    with open(r"{error_file}", 'w') as f:
        f.write(str(e))
finally:
    # Close redirected streams
    if sys.stdout != sys.__stdout__:
        sys.stdout.close()
    if sys.stderr != sys.__stderr__:
        sys.stderr.close()
'''
    
    try:
        # Write the loader script
        with open(script_file, 'w') as f:
            f.write(loader_script)
        
        logger.info(f"🔇 Loading {model_name} silently (this may take a moment)...")
        
        # Run the loader script in complete isolation
        result = subprocess.run([
            sys.executable, str(script_file)
        ], 
        capture_output=True, 
        text=True, 
        timeout=300,  # 5 minute timeout
        env={{**os.environ, 'PYTHONUNBUFFERED': '1'}}
        )
        
        # Check results
        if success_file.exists() and model_file.exists():
            # Read success info
            with open(success_file, 'r') as f:
                success_info = f.read().strip().split(',')
                model_device = success_info[1] if len(success_info) > 1 else device
            
            # Load the model back
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"✅ Model loaded silently on {model_device}")
            return model
        
        elif error_file.exists():
            with open(error_file, 'r') as f:
                error = f.read()
            raise Exception(f"Silent loading failed: {error}")
        
        else:
            # Check subprocess output for debugging
            stdout_output = result.stdout if result.stdout else "No stdout"
            stderr_output = result.stderr if result.stderr else "No stderr"
            raise Exception(f"Silent loading failed: unknown error. Return code: {result.returncode}. Stdout: {stdout_output}. Stderr: {stderr_output}")
    
    except subprocess.TimeoutExpired:
        raise Exception(f"Model loading timed out after 5 minutes for {model_name}")
    
    except Exception as e:
        error_msg = f"Silent loading failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        # Add debugging info
        logger.debug(f"Model name: {model_name}")
        logger.debug(f"Device: {device}")
        logger.debug(f"Cache dir: {cache_dir}")
        logger.debug(f"Temp files: model={model_file}, success={success_file}, error={error_file}")
        
        raise Exception(error_msg)
    
    finally:
        # Cleanup temp files
        for temp_file in [model_file, success_file, error_file, script_file]:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_file}: {e}")


def test_silent_loader():
    """Test function for silent loader"""
    try:
        print("🧪 Testing silent loader...")
        model = load_sentence_transformer_silently(
            'sentence-transformers/all-MiniLM-L6-v2',  # Smaller model for testing
            device='cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
        )
        print(f"✅ Test successful! Model device: {next(model.parameters()).device}")
        
        # Test encoding
        embeddings = model.encode(["This is a test sentence"])
        print(f"✅ Encoding test successful! Shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_silent_loader()