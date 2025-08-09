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


def load_sentence_transformer_silently(model_name: str, device: str = 'cuda:0', cache_dir: str = './models_cache') -> Any:
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
    
    # Prepare temp file paths
    temp_dir = Path(tempfile.gettempdir())
    model_file = temp_dir / f'model_{hash(model_name)}.pkl'
    success_file = temp_dir / f'success_{hash(model_name)}.txt'
    error_file = temp_dir / f'error_{hash(model_name)}.txt'
    script_file = temp_dir / f'loader_{hash(model_name)}.py'
    
    # Create the isolated loader script
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
    cache_path = "{cache_dir}"
    os.makedirs(cache_path, exist_ok=True)
    
    # Load model with optimizations
    model_kwargs = {{
        'trust_remote_code': True,
        'device': '{device}',
        'cache_folder': cache_path
    }}
    
    model = SentenceTransformer('{model_name}', **model_kwargs)
    
    # Verify GPU usage if CUDA device specified
    if '{device}' != 'cpu' and torch.cuda.is_available():
        # Ensure model is on correct device
        model = model.to('{device}')
    
    # Save model to temporary file
    with open('{model_file}', 'wb') as f:
        pickle.dump(model, f)
    
    # Write success indicator with device info
    model_device = str(next(model.parameters()).device) if hasattr(model, 'parameters') else '{device}'
    with open('{success_file}', 'w') as f:
        f.write(f'success,{{model_device}}')
        
except Exception as e:
    # Write error to file
    with open('{error_file}', 'w') as f:
        f.write(str(e))
finally:
    # Close redirected streams
    sys.stdout.close()
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
        logger.error(f"❌ Silent loading failed: {str(e)}")
        raise
    
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