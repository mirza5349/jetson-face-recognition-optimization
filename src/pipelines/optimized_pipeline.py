# src/pipelines/optimized_pipeline.py
import time
import queue
import threading
import numpy as np
import cv2
from PIL import Image

from src.detection.detector import FaceDetector
from src.alignment.aligner import FaceAligner
from src.preprocessing.preprocessor import FacePreprocessor
from src.embeddings.embedder import FaceEmbedder
from src.matching.matcher import FaceMatcher

class OptimizedFaceNetPipeline:
    def __init__(self, device="cpu", model_path=None, mock=False, num_workers=1):
        """
        Optimized FaceNet Pipeline (PyTorch FP16 with Asynchronous Video Stream Execution)
        
        Optimization Layer:
        - OpenCV & NumPy vectorized image operations
        - CLAHE L-channel local contrast correction
        - Landmark-based geometric eye alignment
        - PyTorch FP16 GPU inference (half precision)
        - Vectorized Cosine similarity database lookups
        - Multi-threaded producer-consumer queues overlapping Preprocessing, Detection, and Inference
        """
        self.device = device
        self.mock = mock
        
        # Backends initialization
        # PyTorch FP16 for the optimized backend
        backend = "pytorch_fp16" if (device == "cuda" and not mock) else "pytorch_fp32"
        
        self.detector = FaceDetector(device=device, mock=mock)
        self.aligner = FaceAligner()
        self.preprocessor = FacePreprocessor()
        self.embedder = FaceEmbedder(backend=backend, model_path=model_path, device=device, mock=mock)
        self.matcher = FaceMatcher(metric="cosine", threshold=0.4) # Cosine threshold is normally ~0.4

        # Asynchronous multi-threading components
        self.input_queue = queue.Queue(maxsize=16)
        self.output_queue = queue.Queue()
        self.running = False
        self.worker_thread = None

    def start_async_processing(self, database_embs=None):
        """
        Starts the background worker thread for asynchronous frame pipelining.
        """
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._async_worker_loop, 
            args=(database_embs,),
            daemon=True
        )
        self.worker_thread.start()
        print("Asynchronous frame processing worker thread started.")

    def stop_async_processing(self):
        """
        Stops the asynchronous worker.
        """
        self.running = False
        if self.worker_thread is not None:
            # Wake up thread if it's waiting on empty queue
            try:
                self.input_queue.put(None, block=False)
                self.worker_thread.join(timeout=1.0)
            except Exception:
                pass
            self.worker_thread = None
        print("Asynchronous frame processing worker thread stopped.")

    def enqueue_frame(self, frame_bgr, frame_id=0):
        """
        Pushes a raw BGR frame to the input queue for async processing.
        """
        if not self.running:
            raise RuntimeError("Async processing is not running. Call start_async_processing() first.")
        try:
            self.input_queue.put_nowait((frame_id, frame_bgr))
            return True
        except queue.Full:
            return False # Drop frame to avoid backpressure build up in real-time streams

    def get_async_results(self):
        """
        Retrieves processed results from the output queue.
        """
        results = []
        while not self.output_queue.empty():
            try:
                results.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _async_worker_loop(self, database_embs):
        """
        Background worker that processes enqueued frames.
        Overlaps file reading/camera capture with neural network execution.
        """
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
                if item is None:
                    break
                frame_id, frame_bgr = item
                
                # Process the frame using our optimized pipeline
                res = self.process_frame(frame_bgr, database_embs)
                res["frame_id"] = frame_id
                
                # Put in results queue
                self.output_queue.put(res)
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async worker loop: {e}")

    def process_frame(self, frame_bgr, database_embs=None, enable_clahe=True):
        """
        Processes a single BGR frame synchronously.
        """
        t_start = time.perf_counter()
        
        # 1. Image Format (Ensure BGR numpy)
        if isinstance(frame_bgr, Image.Image):
            # If PIL, convert to BGR OpenCV format
            img_rgb = np.array(frame_bgr)
            frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_rgb_pil = frame_bgr
        else:
            img_rgb_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # 2. Face Detection
        t_det_0 = time.perf_counter()
        boxes, probs, landmarks = self.detector.detect(img_rgb_pil)
        t_det = (time.perf_counter() - t_det_0) * 1000.0

        align_latency = 0.0
        embedding_latency = 0.0
        matching_latency = 0.0
        
        detected_embeddings = []
        matching_results = []

        if boxes is not None and len(boxes) > 0:
            # 3. Geometric Alignment & Preprocessing (CLAHE + Normalize)
            t_prep_0 = time.perf_counter()
            cropped_tensors = []
            
            for i, box in enumerate(boxes):
                if landmarks is not None and len(landmarks) > i:
                    # Landmark-based similarity transform (Optimized alignment)
                    aligned_face = self.aligner.align_optimized(frame_bgr, landmarks[i])
                else:
                    # Fallback to direct crop if landmarks missing
                    x1, y1, x2, y2 = map(int, box)
                    aligned_face = frame_bgr[max(0, y1):y2, max(0, x1):x2]
                
                # OpenCV preprocessing (CLAHE + Norm + Transpose)
                tensor = self.preprocessor.preprocess_optimized(aligned_face, enable_clahe=enable_clahe)
                cropped_tensors.append(tensor)
                
            # Vectorized Batch preparation
            batch_tensor = np.stack(cropped_tensors, axis=0)
            align_latency = (time.perf_counter() - t_prep_0) * 1000.0

            # 4. FP16 Inference
            t_emb_0 = time.perf_counter()
            detected_embeddings = self.embedder.compute_embeddings(batch_tensor)
            
            # Explicit L2 normalization of embeddings (essential for Cosine Similarity)
            norms = np.linalg.norm(detected_embeddings, axis=1, keepdims=True)
            detected_embeddings = detected_embeddings / np.maximum(norms, 1e-12)
            embedding_latency = (time.perf_counter() - t_emb_0) * 1000.0

            # 5. Cosine Database Matching (Matrix-Vector dot product)
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
                "preprocessing_latency_ms": align_latency,
                "embedding_latency_ms": embedding_latency,
                "matching_latency_ms": matching_latency,
                "total_latency_ms": total_latency,
                "fps": 1000.0 / total_latency if total_latency > 0 else 0.0
            }
        }
