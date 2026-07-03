# src/alignment/aligner.py
import cv2
import numpy as np
from PIL import Image

class FaceAligner:
    def __init__(self, desired_face_width=160, desired_face_height=160, desired_left_eye=(0.3, 0.3)):
        """
        Face aligner that performs similarity transformations based on 5 facial landmark coordinates.
        
        Optimization Layer:
        - Geometric eye-alignment stabilizing facial roll/tilt
        - Desired left/right eye locations mapped to stable spatial coordinates in final 160x160 tensor
        - Direct affine mapping bypassing un-aligned cropping to reduce distortion
        """
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        self.desired_left_eye = desired_left_eye
        self.desired_right_eye_x = 1.0 - desired_left_eye[0]

    def crop_base(self, img, box):
        """
        Base cropping (no alignment, standard PIL-based box crop).
        Args:
            img: PIL.Image
            box: List/numpy array [x1, y1, x2, y2]
        Returns:
            cropped_img: PIL.Image
        """
        # Original Base FaceNet cropping behavior
        x1, y1, x2, y2 = map(int, box)
        # Ensure coordinates are within image boundaries
        if isinstance(img, Image.Image):
            w, h = img.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return img.crop((x1, y1, x2, y2))
        else: # Numpy/OpenCV array
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop_np = img[y1:y2, x1:x2]
            return Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))

    def align_optimized(self, img_np, landmarks):
        """
        Optimized alignment using 2D similarity transform (translation, scaling, rotation).
        Args:
            img_np: numpy.ndarray, representing BGR or RGB image
            landmarks: numpy.ndarray of shape [5, 2] representing 5 face keypoints
        Returns:
            aligned_face: numpy.ndarray of shape [desired_face_height, desired_face_width, 3]
        """
        # Extract eye coordinates
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Compute the angle between the eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Desired right eye x-coordinate
        desired_right_eye_x = self.desired_right_eye_x

        # Calculate scale factor based on eye distances
        dist = np.sqrt(dx**2 + dy**2)
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.desired_face_width
        scale = desired_dist / max(1e-6, dist)

        # Center of eyes in source image
        eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

        # Get the transformation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle_deg, scale)

        # Translate eyes center to desired center
        t_x = self.desired_face_width * 0.5
        t_y = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (t_x - eyes_center[0])
        M[1, 2] += (t_y - eyes_center[1])

        # Apply affine warp to align and crop
        aligned_face = cv2.warpAffine(
            img_np, 
            M, 
            (self.desired_face_width, self.desired_face_height),
            flags=cv2.INTER_CUBIC
        )
        return aligned_face
