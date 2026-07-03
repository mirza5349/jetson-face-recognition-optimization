# src/pipelines/base_pipeline.py
import time
import numpy as np
from PIL import Image

from src.detection.detector import FaceDetector
from src.alignment.aligner import FaceAligner
from src.preprocessing.preprocessor import FacePreprocessor
from src.embeddings.embedder import FaceEmbedder
from src.matching.matcher import FaceMatcher

class BaseFaceNetPipeline:
    def __init__(self, device="cpu", model_path=None, mock=False):
        """
        Base FaceNet Pipeline (Standard PyTorch FP32)
        
        Reference Implementation:
        - Uses original facenet-pytorch synchronous flow
        - Uses PIL image processing (slow on CPUs and edge devices)
        - Synchronous single-threaded processing
        - Euclidean similarity matching
        """
        self.detector = FaceDetector(device=device, mock=mock)
        self.aligner = FaceAligner()
        self.preprocessor = FacePreprocessor()
        self.embedder = FaceEmbedder(backend="pytorch_fp32", model_path=model_path, device=device, mock=mock)
        self.matcher = FaceMatcher(metric="euclidean", threshold=0.6)

    def process_frame(self, frame_rgb, database_embs=None):
        """
        Processes a single video frame or image synchronously.
        Args:
            frame_rgb: numpy.ndarray, shape [H, W, 3] (RGB format, or PIL Image)
            database_embs: numpy.ndarray [N, 512], registered identity database
        Returns:
            results: dict containing latencies, bounding boxes, and embeddings
        """
        t_start = time.perf_counter()
        
        # 1. Image Format Setup
        if isinstance(frame_rgb, np.ndarray):
            pil_img = Image.fromarray(frame_rgb)
            img_np_bgr = frame_rgb[:, :, ::-1] # Save BGR for OpenCV subtests if needed
        else:
            pil_img = frame_rgb
            frame_rgb = np.array(pil_img)
            img_np_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if 'cv2' in globals() else frame_rgb

        # 2. Face Detection
        t_det_0 = time.perf_counter()
        boxes, probs, landmarks = self.detector.detect(pil_img)
        t_det = (time.perf_counter() - t_det_0) * 1000.0 # to ms

        crops_latency = 0.0
        embedding_latency = 0.0
        matching_latency = 0.0
        
        detected_embeddings = []
        matching_results = []
        
        if boxes is not None and len(boxes) > 0:
            # 3. Preprocessing (Crop + Resize + Normalize)
            t_prep_0 = time.perf_counter()
            cropped_tensors = []
            for box in boxes:
                # PIL crop (No alignment in Base version)
                crop = self.aligner.crop_base(pil_img, box)
                # PIL preprocess
                tensor = self.preprocessor.preprocess_base(crop)
                cropped_tensors.append(tensor)
                
            # Stack to batch tensor [B, 3, 160, 160]
            batch_tensor = np.stack(cropped_tensors, axis=0)
            crops_latency = (time.perf_counter() - t_prep_0) * 1000.0

            # 4. Feature Embedding
            t_emb_0 = time.perf_counter()
            detected_embeddings = self.embedder.compute_embeddings(batch_tensor)
            embedding_latency = (time.perf_counter() - t_emb_0) * 1000.0

            # 5. Distance Matching
            if database_embs is not None and len(database_embs) > 0:
                t_match_0 = time.perf_counter()
                for emb in detected_embeddings:
                    distances, best_idx = self.matcher.search_database(emb, database_embs)
                    matching_results.append({
                        "best_index": int(best_idx),
                        "distance": float(distances[best_idx]),
                        "match": bool(distances[best_idx] < self.matcher.threshold)
                    })
                matching_latency = (time.perf_counter() - t_match_0) * 1000.0

        total_latency = (time.perf_counter() - t_start) * 1000.0

        return {
            "face_detected": len(boxes) if boxes is not None else 0,
            "detections": boxes.tolist() if boxes is not None else [],
            "boxes": boxes.tolist() if boxes is not None else [],
            "scores": probs.tolist() if probs is not None else [],
            "embeddings": detected_embeddings if len(detected_embeddings) > 0 else np.zeros((0, 512)),
            "matches": matching_results,
            "matching": matching_results,
            "metrics": {
                "detection_latency_ms": t_det,
                "preprocessing_latency_ms": crops_latency,
                "embedding_latency_ms": embedding_latency,
                "matching_latency_ms": matching_latency,
                "total_latency_ms": total_latency,
                "fps": 1000.0 / total_latency if total_latency > 0 else 0.0
            }
        }
