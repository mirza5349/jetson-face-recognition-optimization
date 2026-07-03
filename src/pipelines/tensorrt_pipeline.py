# src/pipelines/tensorrt_pipeline.py
from src.pipelines.optimized_pipeline import OptimizedFaceNetPipeline
from src.embeddings.embedder import FaceEmbedder

class TensorRTFaceNetPipeline(OptimizedFaceNetPipeline):
    def __init__(self, device="cuda", model_path=None, mock=False):
        """
        NVIDIA TensorRT Accelerated FaceNet Pipeline
        
        Optimization Layer:
        - Inherits high-performance OpenCV preprocessing, CLAHE, and similarity alignment
        - Overrides the embedding backend to native compiled TensorRT FP16
        - Employs preallocated zero-copy GPU buffers (pinned memory) via pycuda/ctypes
        - Minimizes H2D and D2H copies, using async stream copies for overlapped executions
        """
        # TensorRT runs on GPU, so default device is 'cuda'
        super().__init__(device=device, model_path=model_path, mock=mock)
        
        # Override the embedder with TensorRT backend
        self.embedder = FaceEmbedder(backend="tensorrt", model_path=model_path, device=device, mock=mock)
