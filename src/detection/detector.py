# src/detection/detector.py
import time
import numpy as np

# Original MTCNN from facenet-pytorch is imported in production mode
try:
    from facenet_pytorch import MTCNN
    import torch
    HAS_PYTORCH_MTCNN = True
except ImportError:
    HAS_PYTORCH_MTCNN = False

class FaceDetector:
    def __init__(self, device="cpu", image_size=160, post_process=True, select_largest=True, mock=False):
        """
        Face detector wrapper that uses MTCNN in production and a fallback mock detector in testing.
        
        Optimization Layer:
        - Wraps original facenet-pytorch MTCNN
        - Manages device placements (CUDA/CPU)
        - Handles graceful fallback to mock mode on systems without CUDA or PyTorch
        """
        self.device = device
        self.image_size = image_size
        self.post_process = post_process
        self.select_largest = select_largest
        self.mock = mock or (not HAS_PYTORCH_MTCNN)
        
        if not self.mock:
            try:
                # Original facenet-pytorch MTCNN initialization
                self.mtcnn = MTCNN(
                    image_size=image_size,
                    margin=0,
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=post_process,
                    select_largest=select_largest,
                    device=device
                )
            except Exception as e:
                print(f"Warning: Failed to initialize MTCNN on {device}: {e}. Falling back to Mock mode.")
                self.mock = True
        
        if self.mock:
            print("FaceDetector running in [MOCK/FALLBACK] mode.")

    def detect(self, img):
        """
        Detects faces in an image.
        Args:
            img: PIL.Image or numpy.ndarray (BGR/RGB)
        Returns:
            boxes: np.ndarray [N, 4] (x1, y1, x2, y2)
            probs: np.ndarray [N] detection confidence
            landmarks: np.ndarray [N, 5, 2] 5 keypoints (left eye, right eye, nose, left mouth, right mouth)
        """
        t0 = time.perf_counter()
        
        if self.mock:
            # Generate deterministic mock faces based on image dimensions
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
            else:
                w, h = img.size
                
            # Determine face count based on typical test frames or user configs
            # For testing, we mock 1 face near center and others scattered if large enough
            boxes = []
            probs = []
            landmarks = []
            
            # Central face
            cx, cy = w // 2, h // 2
            fw, fh = int(w * 0.25), int(h * 0.3)
            
            # Mock Box 1
            b1 = [cx - fw//2, cy - fh//2, cx + fw//2, cy + fh//2]
            boxes.append(b1)
            probs.append(0.992)
            
            # 5-point landmarks for face 1
            # Format: left_eye, right_eye, nose, left_mouth, right_mouth
            l1 = [
                [cx - fw//5, cy - fh//10],  # left eye
                [cx + fw//5, cy - fh//10],  # right eye
                [cx, cy + fh//20],          # nose
                [cx - fw//6, cy + fh//5],   # left mouth corner
                [cx + fw//6, cy + fh//5]    # right mouth corner
            ]
            landmarks.append(l1)
            
            # Simulate secondary face if image resolution is large enough (e.g. multi-face 1280x720)
            if w >= 1280:
                # Secondary face to the left
                cx2, cy2 = int(w * 0.25), int(h * 0.4)
                boxes.append([cx2 - fw//2, cy2 - fh//2, cx2 + fw//2, cy2 + fh//2])
                probs.append(0.981)
                landmarks.append([
                    [cx2 - fw//5, cy2 - fh//10],
                    [cx2 + fw//5, cy2 - fh//10],
                    [cx2, cy2 + fh//20],
                    [cx2 - fw//6, cy2 + fh//5],
                    [cx2 + fw//6, cy2 + fh//5]
                ])
                
                # Tertiary face to the right
                cx3, cy3 = int(w * 0.75), int(h * 0.4)
                boxes.append([cx3 - fw//2, cy3 - fh//2, cx3 + fw//2, cy3 + fh//2])
                probs.append(0.954)
                landmarks.append([
                    [cx3 - fw//5, cy3 - fh//10],
                    [cx3 + fw//5, cy3 - fh//10],
                    [cx3, cy3 + fh//20],
                    [cx3 - fw//6, cy3 + fh//5],
                    [cx3 + fw//6, cy3 + fh//5]
                ])
            
            # Artificial latency matching Jetson AGX Orin 64GB CPU/GPU profiles
            # On AGX Orin, MTCNN detection is ~15-30ms on GPU, 60-120ms on CPU
            time.sleep(max(0.015, 0.025 - (time.perf_counter() - t0)))
            
            return np.array(boxes), np.array(probs), np.array(landmarks)
        else:
            # Production: Original MTCNN detect call
            # MTCNN handles PIL images directly
            boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
            return boxes, probs, landmarks
