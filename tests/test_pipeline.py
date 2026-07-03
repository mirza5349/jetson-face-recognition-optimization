# tests/test_pipeline.py
import os
import sys
import numpy as np

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipelines.base_pipeline import BaseFaceNetPipeline
from src.pipelines.optimized_pipeline import OptimizedFaceNetPipeline
from src.pipelines.onnx_pipeline import ONNXFaceNetPipeline
from src.pipelines.tensorrt_pipeline import TensorRTFaceNetPipeline

def verify_pipeline_output_contract(output_dict):
    """
    Verifies that a pipeline output follows the strict contract expected by benchmarks.
    """
    assert isinstance(output_dict, dict), "Pipeline output must be a dictionary"
    
    expected_keys = ["face_detected", "detections", "embeddings", "matches", "metrics"]
    for k in expected_keys:
        assert k in output_dict, f"Missing key '{k}' in pipeline output"
        
    # Check metrics fields
    m = output_dict["metrics"]
    metric_fields = ["detection_latency_ms", "preprocessing_latency_ms", "embedding_latency_ms", "matching_latency_ms"]
    for f in metric_fields:
        assert f in m, f"Missing metric field '{f}' in metrics dictionary"
        assert isinstance(m[f], float), f"Metric field '{f}' must be a float"
        
    # Validate structure match
    faces_detected = output_dict["face_detected"]
    assert isinstance(faces_detected, int), "face_detected must be an integer"
    
    if faces_detected > 0:
        assert len(output_dict["detections"]) == faces_detected
        assert len(output_dict["embeddings"]) == faces_detected
        assert len(output_dict["matches"]) == faces_detected

def test_base_pipeline():
    """
    Tests standard synchronous Base PyTorch FP32 Pipeline.
    """
    pipeline = BaseFaceNetPipeline(device="cpu", mock=True)
    
    # Create dummy RGB frame
    dummy_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Simple registry database (10 identities)
    db = np.random.normal(0, 1, (10, 512)).astype(np.float32)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    
    out = pipeline.process_frame(dummy_rgb, database_embs=db)
    verify_pipeline_output_contract(out)

def test_optimized_pipeline():
    """
    Tests Optimized PyTorch Pipeline with queue-based frame processing.
    """
    pipeline = OptimizedFaceNetPipeline(device="cpu", mock=True)
    
    # OpenCV standard image is BGR
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    db = np.random.normal(0, 1, (10, 512)).astype(np.float32)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    
    out = pipeline.process_frame(dummy_bgr, database_embs=db)
    verify_pipeline_output_contract(out)

def test_onnx_pipeline():
    """
    Tests ONNX Runtime Pipeline with CUDA providers.
    """
    pipeline = ONNXFaceNetPipeline(device="cpu", model_path="models/facenet.onnx", mock=True)
    
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    db = np.random.normal(0, 1, (10, 512)).astype(np.float32)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    
    out = pipeline.process_frame(dummy_bgr, database_embs=db)
    verify_pipeline_output_contract(out)

def test_tensorrt_pipeline():
    """
    Tests TensorRT FP16 compiled engine pipeline with page-locked buffers.
    """
    pipeline = TensorRTFaceNetPipeline(device="cpu", model_path="models/facenet_fp16.engine", mock=True)
    
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    db = np.random.normal(0, 1, (10, 512)).astype(np.float32)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    
    out = pipeline.process_frame(dummy_bgr, database_embs=db)
    verify_pipeline_output_contract(out)
