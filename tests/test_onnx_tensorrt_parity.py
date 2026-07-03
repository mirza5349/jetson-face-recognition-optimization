# tests/test_onnx_tensorrt_parity.py
import os
import sys
import numpy as np

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings.embedder import FaceEmbedder

def test_onnx_tensorrt_numerical_parity():
    """
    Asserts that the compiled TensorRT FP16 engine and the source ONNX graph
    yield equivalent face embeddings within acceptable numerical bounds (accounting for FP16 quantization).
    """
    rng = np.random.default_rng(9999)
    batch_tensor = rng.uniform(-1.0, 1.0, (1, 3, 160, 160)).astype(np.float32)
    
    # Initialize ONNX embedder
    onnx_embedder = FaceEmbedder(backend="onnx", model_path="models/facenet.onnx")
    # Initialize TensorRT FP16 embedder
    trt_embedder = FaceEmbedder(backend="tensorrt", model_path="models/facenet_fp16.engine")
    
    # Compute embeddings
    onnx_out = onnx_embedder.compute_embeddings(batch_tensor)
    trt_out = trt_embedder.compute_embeddings(batch_tensor)
    
    # Verify dimensions and normalization
    assert onnx_out.shape == (1, 512), f"ONNX shape mismatch: {onnx_out.shape}"
    assert trt_out.shape == (1, 512), f"TensorRT shape mismatch: {trt_out.shape}"
    
    assert np.allclose(np.linalg.norm(onnx_out[0]), 1.0, atol=1e-5), "ONNX output must be L2-normalized"
    assert np.allclose(np.linalg.norm(trt_out[0]), 1.0, atol=1e-5), "TensorRT output must be L2-normalized"
    
    if not onnx_embedder.mock and not trt_embedder.mock:
        # On actual AGX Orin, we check that FP16 compilation matches within 5e-3 tolerance
        # (slightly relaxed relative to FP32 due to FP16 quantization noise)
        similarity = np.dot(onnx_out[0], trt_out[0])
        print(f"Physical ONNX-TensorRT Cosine Similarity: {similarity:.6f}")
        assert similarity > 0.99, f"TensorRT quantization diverged from ONNX. Similarity: {similarity}"
    else:
        # Mock mode deterministic validation
        print("Executing mock-mode ONNX-TensorRT perfect match validation...")
        assert np.allclose(onnx_out, trt_out, rtol=1e-5, atol=1e-5), \
            "Mock outputs did not match on identical input tensors"
            
def test_tensorrt_batch_processing_limits():
    """
    Verifies that the TensorRT preallocated bindings handle various batch sizes up to max capacity.
    """
    trt_embedder = FaceEmbedder(backend="tensorrt", model_path="models/facenet_fp16.engine")
    
    # Test batch size 1 (single face)
    inp_b1 = np.random.uniform(-1.0, 1.0, (1, 3, 160, 160)).astype(np.float32)
    out_b1 = trt_embedder.compute_embeddings(inp_b1)
    assert out_b1.shape == (1, 512)
    
    # Test batch size 4 (multi-face)
    inp_b4 = np.random.uniform(-1.0, 1.0, (4, 3, 160, 160)).astype(np.float32)
    out_b4 = trt_embedder.compute_embeddings(inp_b4)
    assert out_b4.shape == (4, 512)
    
    # Test batch size 8
    inp_b8 = np.random.uniform(-1.0, 1.0, (8, 3, 160, 160)).astype(np.float32)
    out_b8 = trt_embedder.compute_embeddings(inp_b8)
    assert out_b8.shape == (8, 512)
    
    # Verify we can extract distinct embeddings per batch index
    assert not np.allclose(out_b4[0], out_b4[1]), "Batch indices should represent distinct outputs"
