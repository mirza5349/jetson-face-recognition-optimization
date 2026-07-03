# src/preprocessing/preprocessor.py
import cv2
import numpy as np
from PIL import Image

class FacePreprocessor:
    def __init__(self, target_size=(160, 160)):
        """
        Face image preprocessing and normalization module.
        
        Optimization Layer:
        - OpenCV-based image manipulation (vectorized, multithreaded)
        - CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB L channel to normalize illumination variance
        - NumPy strided array transformations instead of heavy PIL object transformations
        """
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_base(self, pil_img):
        """
        Base preprocessing using standard PIL-based steps.
        """
        # Original Base Pipeline steps (PIL & numpy conversion)
        img_resized = pil_img.resize(self.target_size, Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32)
        # Normalization used in standard facenet-pytorch
        img_normalized = (img_np - 127.5) / 128.0
        # Format HWC to CHW
        img_tensor = img_normalized.transpose((2, 0, 1))
        return img_tensor

    def apply_clahe(self, img_bgr):
        """
        Applies CLAHE on the LAB L channel of the image to normalize local contrast.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def preprocess_optimized(self, img_bgr, enable_clahe=True):
        """
        Optimized preprocessing using OpenCV and NumPy.
        """
        # Apply CLAHE to L channel in LAB color space if enabled
        if enable_clahe:
            img_bgr = self.apply_clahe(img_bgr)
            
        # Convert BGR to RGB (OpenCV default is BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize using fast OpenCV resize
        img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalization with NumPy in-place
        img_np = img_resized.astype(np.float32)
        np.subtract(img_np, 127.5, out=img_np)
        np.multiply(img_np, 1/128.0, out=img_np)
        
        # HWC to CHW
        img_tensor = img_np.transpose((2, 0, 1))
        return img_tensor
