# src/pipelines/onnx_pipeline.py
from src.pipelines.optimized_pipeline import OptimizedFaceNetPipeline
from src.embeddings.embedder import FaceEmbedder

class ONNXFaceNetPipeline(OptimizedFaceNetPipeline):
    def __init__(self, device="cpu", model_path=None, mock=False):
        """
        ONNX Runtime Acceleration Pipeline
        
        Optimization Layer:
        - Inherits high-performance OpenCV preprocessing, CLAHE, and 5-point alignment from Optimized Pipeline
        - Overrides the embedding backend to run ONNX Runtime using CUDAExecutionProvider
        - Maintains absolute functional parity to ensure fair comparison
        """
        super().__init__(device=device, model_path=model_path, mock=mock)
        
        # Override the PyTorch embedder with ONNX Runtime backend
        # Note: If model_path is None, FaceEmbedder automatically exports a standard ONNX FaceNet
        self.embedder = FaceEmbedder(backend="onnx", model_path=model_path, device=device, mock=mock)
