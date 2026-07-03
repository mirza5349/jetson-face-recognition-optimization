# tests/test_preprocessing.py
import os
import sys
import numpy as np
from PIL import Image

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.preprocessor import FacePreprocessor

def test_base_preprocessing():
    """
    Verifies that the standard PIL-based preprocessing operates correctly 
    and outputs a float32 tensor of shape (3, 160, 160) normalized to [-1, 1].
    """
    preprocessor = FacePreprocessor(target_size=(160, 160))
    
    # Create random PIL Image representing input face crop
    dummy_img = Image.fromarray(np.random.randint(0, 256, (180, 180, 3), dtype=np.uint8))
    
    # Apply preprocessing
    out_tensor = preprocessor.preprocess_base(dummy_img)
    
    # Check assertions
    assert isinstance(out_tensor, np.ndarray), "Output must be a numpy array"
    assert out_tensor.dtype == np.float32, "Output array must be float32"
    assert out_tensor.shape == (3, 160, 160), f"Output shape was {out_tensor.shape}, expected (3, 160, 160)"
    assert np.min(out_tensor) >= -1.0, f"Min value {np.min(out_tensor)} is outside expected standard range"
    assert np.max(out_tensor) <= 1.0, f"Max value {np.max(out_tensor)} is outside expected standard range"

def test_optimized_preprocessing_with_clahe():
    """
    Verifies that the OpenCV-based optimized preprocessing works correctly, 
    performs color conversion, CLAHE equalization, and outputs a float32 tensor 
    of shape (3, 160, 160) normalized to [-1, 1].
    """
    preprocessor = FacePreprocessor(target_size=(160, 160))
    
    # Create random numpy BGR image (OpenCV standard)
    dummy_bgr = np.random.randint(0, 256, (180, 180, 3), dtype=np.uint8)
    
    # Apply CLAHE independently
    clahe_bgr = preprocessor.apply_clahe(dummy_bgr)
    assert clahe_bgr.shape == dummy_bgr.shape, "CLAHE should preserve image shape"
    assert clahe_bgr.dtype == np.uint8, "CLAHE should preserve image dtype"
    
    # Apply complete optimized preprocessing (with CLAHE)
    out_tensor = preprocessor.preprocess_optimized(dummy_bgr, enable_clahe=True)
    
    # Check assertions
    assert isinstance(out_tensor, np.ndarray), "Output must be a numpy array"
    assert out_tensor.dtype == np.float32, "Output array must be float32"
    assert out_tensor.shape == (3, 160, 160), f"Output shape was {out_tensor.shape}, expected (3, 160, 160)"
    assert np.min(out_tensor) >= -1.0, f"Min value {np.min(out_tensor)} is outside normalized range"
    assert np.max(out_tensor) <= 1.0, f"Max value {np.max(out_tensor)} is outside normalized range"

def test_optimized_preprocessing_without_clahe():
    """
    Verifies optimized preprocessing when CLAHE illumination normalization is bypassed (ablation study).
    """
    preprocessor = FacePreprocessor(target_size=(160, 160))
    dummy_bgr = np.random.randint(0, 256, (180, 180, 3), dtype=np.uint8)
    
    out_tensor = preprocessor.preprocess_optimized(dummy_bgr, enable_clahe=False)
    
    assert out_tensor.shape == (3, 160, 160)
    assert np.min(out_tensor) >= -1.0
    assert np.max(out_tensor) <= 1.0
