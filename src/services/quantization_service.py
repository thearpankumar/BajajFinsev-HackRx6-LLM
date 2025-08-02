import logging
from typing import Dict, Any
import torch

logger = logging.getLogger(__name__)

class ModelQuantizationService:
    """
    Service for model quantization to reduce model size and improve inference speed
    for resource-constrained environments.
    """
    
    def __init__(self):
        self.logger = logger
        self.quantized_models = {}
    
    def quantize_model(self, model: torch.nn.Module, 
                      quantization_type: str = "dynamic",
                      bits: int = 8) -> torch.nn.Module:
        """
        Quantize a PyTorch model to reduce size and improve inference speed.
        
        Args:
            model: PyTorch model to quantize
            quantization_type: Type of quantization ("dynamic", "static", "qat")
            bits: Number of bits for quantization (4 or 8)
            
        Returns:
            Quantized model
        """
        try:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, quantization may be slower")
            
            # Prepare model for quantization
            model.eval()
            model.cpu()  # Move to CPU for quantization
            
            if quantization_type == "dynamic":
                quantized_model = self._dynamic_quantization(model, bits)
            elif quantization_type == "static":
                quantized_model = self._static_quantization(model, bits)
            elif quantization_type == "qat":
                quantized_model = self._quantization_aware_training(model, bits)
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            self.logger.info(f"Model quantized using {quantization_type} quantization with {bits} bits")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Failed to quantize model: {e}")
            raise
    
    def _dynamic_quantization(self, model: torch.nn.Module, 
                             bits: int = 8) -> torch.nn.Module:
        """
        Apply dynamic quantization to a model.
        
        Args:
            model: PyTorch model to quantize
            bits: Number of bits for quantization
            
        Returns:
            Dynamically quantized model
        """
        try:
            if bits == 8:
                # 8-bit dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            elif bits == 4:
                # 4-bit dynamic quantization (if supported)
                if hasattr(torch.quantization, 'quantize_dynamic_4bit'):
                    quantized_model = torch.quantization.quantize_dynamic_4bit(
                        model, {torch.nn.Linear}
                    )
                else:
                    self.logger.warning("4-bit dynamic quantization not supported, using 8-bit")
                    quantized_model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
            else:
                raise ValueError(f"Unsupported bit width: {bits}")
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Dynamic quantization failed: {e}")
            raise
    
    def _static_quantization(self, model: torch.nn.Module, 
                            bits: int = 8) -> torch.nn.Module:
        """
        Apply static quantization to a model.
        
        Args:
            model: PyTorch model to quantize
            bits: Number of bits for quantization
            
        Returns:
            Statically quantized model
        """
        try:
            # Prepare model for static quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate the model (this would typically use calibration data)
            # For now, we'll just run a dummy calibration
            self._calibrate_model(model)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Static quantization failed: {e}")
            raise
    
    def _quantization_aware_training(self, model: torch.nn.Module, 
                                    bits: int = 8) -> torch.nn.Module:
        """
        Apply quantization-aware training to a model.
        
        Args:
            model: PyTorch model to quantize
            bits: Number of bits for quantization
            
        Returns:
            Model prepared for quantization-aware training
        """
        try:
            # Prepare model for QAT
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            
            self.logger.info("Model prepared for quantization-aware training")
            return model
            
        except Exception as e:
            self.logger.error(f"Quantization-aware training preparation failed: {e}")
            raise
    
    def _calibrate_model(self, model: torch.nn.Module):
        """
        Calibrate a model for static quantization.
        
        Args:
            model: Model to calibrate
        """
        # This is a dummy calibration function
        # In practice, you would run sample data through the model
        self.logger.info("Calibrating model for static quantization")
    
    def save_quantized_model(self, model: torch.nn.Module, 
                            filepath: str) -> bool:
        """
        Save a quantized model to disk.
        
        Args:
            model: Quantized model to save
            filepath: Path to save the model
            
        Returns:
            Success status
        """
        try:
            torch.save(model.state_dict(), filepath)
            self.logger.info(f"Quantized model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save quantized model: {e}")
            return False
    
    def load_quantized_model(self, model_class: type, 
                            filepath: str) -> torch.nn.Module:
        """
        Load a quantized model from disk.
        
        Args:
            model_class: Class of the model to load
            filepath: Path to load the model from
            
        Returns:
            Loaded quantized model
        """
        try:
            # Create model instance
            model = model_class()
            
            # Load state dict
            model.load_state_dict(torch.load(filepath))
            
            self.logger.info(f"Quantized model loaded from {filepath}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load quantized model: {e}")
            raise
    
    def compare_model_sizes(self, original_model: torch.nn.Module, 
                           quantized_model: torch.nn.Module) -> Dict[str, Any]:
        """
        Compare the sizes of original and quantized models.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            
        Returns:
            Dictionary with size comparison results
        """
        try:
            # Calculate sizes
            original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
            quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
            
            # Calculate reduction
            size_reduction = (original_size - quantized_size) / original_size * 100
            
            return {
                'original_size_bytes': original_size,
                'quantized_size_bytes': quantized_size,
                'size_reduction_percent': size_reduction,
                'size_reduction_ratio': f"1:{original_size/quantized_size:.2f}"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare model sizes: {e}")
            raise
    
    def measure_inference_speed(self, model: torch.nn.Module,
                                  input_data: torch.Tensor,
                                  iterations: int = 100) -> float:
        """
        Measure inference speed of a model.
        
        Args:
            model: Model to test
            input_data: Input data for inference
            iterations: Number of iterations to run
            
        Returns:
            Average inference time in milliseconds
        """
        import time
        
        try:
            # Warm up
            for _ in range(10):
                _ = model(input_data)
            
            # Measure inference time
            start_time = time.time()
            for _ in range(iterations):
                _ = model(input_data)
            end_time = time.time()
            
            # Calculate average time in milliseconds
            avg_time_ms = (end_time - start_time) / iterations * 1000
            
            return avg_time_ms
            
        except Exception as e:
            self.logger.error(f"Failed to measure inference speed: {e}")
            raise

# Global instance
quantization_service = ModelQuantizationService()