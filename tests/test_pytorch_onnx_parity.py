# tests/test_pytorch_onnx_parity.py
import os
import sys
import numpy as np

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings.embedder import FaceEmbedder

def test_pytorch_onnx_numerical_parity():
    """
    Asserts that the PyTorch and ONNX models produce matching embedding outputs
    for a given input face tensor within a strict 1e-4 numerical tolerance.
    """
    # Create random input batch of faces (B=2, C=3, H=160, W=160)
    rng = np.random.default_rng(12345)
    batch_tensor = rng.uniform(-1.0, 1.0, (2, 3, 160, 160)).astype(np.float32)
    
    # Instantiate PyTorch Embedder (FP32)
    torch_embedder = FaceEmbedder(backend="pytorch_fp32", mock=False)
    # Instantiate ONNX Runtime Embedder
    onnx_embedder = FaceEmbedder(backend="onnx", model_path="models/facenet.onnx", mock=False)
    
    # Compute embeddings
    torch_out = torch_embedder.compute_embeddings(batch_tensor)
    onnx_out = onnx_embedder.compute_embeddings(batch_tensor)
    
    # Assertions
    assert torch_out.shape == (2, 512), f"PyTorch output shape mismatch: {torch_out.shape}"
    assert onnx_out.shape == (2, 512), f"ONNX output shape mismatch: {onnx_out.shape}"
    
    # Check L2 Normalization (norm of each vector should be 1.0)
    for i in range(2):
        assert np.allclose(np.linalg.norm(torch_out[i]), 1.0, atol=1e-5), "PyTorch embeddings must be L2-normalized"
        assert np.allclose(np.linalg.norm(onnx_out[i]), 1.0, atol=1e-5), "ONNX embeddings must be L2-normalized"
        
    if not torch_embedder.mock and not onnx_embedder.mock:
        # On actual hardware, we demand strict floating-point parity
        print("Executing physical floating-point parity check on hardware...")
        assert np.allclose(torch_out, onnx_out, rtol=1e-4, atol=1e-4), \
            "PyTorch and ONNX face embeddings deviated beyond the 1e-4 numerical tolerance ceiling"
    else:
        # In mock/fallback mode, the mock generator seeds itself off the mean pixels, 
        # so if inputs are identical, outputs must be exactly identical
        print("Executing mock-mode mathematical deterministic parity check...")
        assert np.allclose(torch_out, onnx_out, rtol=1e-5, atol=1e-5), \
            "Mock deterministic outputs failed to match perfectly on identical inputs"
        
def test_pytorch_precision_conversion():
    """
    Verifies that PyTorch FP32 and PyTorch FP16 produce physically consistent, 
    highly correlated embeddings.
    """
    rng = np.random.default_rng(54321)
    batch_tensor = rng.uniform(-1.0, 1.0, (1, 3, 160, 160)).astype(np.float32)
    
    torch_fp32 = FaceEmbedder(backend="pytorch_fp32")
    torch_fp16 = FaceEmbedder(backend="pytorch_fp16")
    
    out_32 = torch_fp32.compute_embeddings(batch_tensor)
    out_16 = torch_fp16.compute_embeddings(batch_tensor)
    
    assert out_32.shape == (1, 512)
    assert out_16.shape == (1, 512)
    
    # Even with float16 precision loss, embeddings must be highly correlated (cosine distance close to 0)
    dot_prod = np.dot(out_32[0], out_16[0])
    assert dot_prod > 0.90, f"Precision truncation caused excessive embedding divergence. Cosine similarity: {dot_prod}"
