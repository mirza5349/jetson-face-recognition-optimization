# scripts/prepare_dataset.py
import os
import sys
import json
import cv2
import numpy as np

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def draw_synthetic_face(img, center, radius, color, label):
    """
    Draws a stylized face shape on a canvas for detection and preprocessing tests.
    """
    cx, cy = center
    # Draw face oval
    cv2.circle(img, (cx, cy), radius, color, -1)
    cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
    
    # Draw eyes
    eye_offset_x = int(radius * 0.3)
    eye_offset_y = int(radius * 0.25)
    cv2.circle(img, (cx - eye_offset_x, cy - eye_offset_y), int(radius*0.1), (0, 0, 0), -1)
    cv2.circle(img, (cx + eye_offset_x, cy - eye_offset_y), int(radius*0.1), (0, 0, 0), -1)
    
    # Draw nose
    cv2.polygon = np.array([
        [cx, cy - int(radius*0.05)],
        [cx - int(radius*0.08), cy + int(radius*0.15)],
        [cx + int(radius*0.08), cy + int(radius*0.15)]
    ], dtype=np.int32)
    cv2.polylines(img, [cv2.polygon], True, (0, 0, 0), 2)
    
    # Draw mouth (smiling)
    cv2.ellipse(img, (cx, cy + int(radius*0.35)), (int(radius*0.25), int(radius*0.15)), 0, 0, 180, (0, 0, 0), 2)
    
    # Label
    cv2.putText(img, label, (cx - radius, cy - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def create_synthetic_image(width, height, face_count, pose_angle=0, lighting="normal", is_blur=False, is_occlude=False):
    """
    Generates a synthetic image with drawn faces, lighting, and degradation profiles.
    """
    # Create dark background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (0,0), (width, height), (35, 30, 30), -1)
    
    # Draw grids for background detail
    for x in range(0, width, 80):
        cv2.line(img, (x, 0), (x, height), (45, 40, 40), 1)
    for y in range(0, height, 80):
        cv2.line(img, (0, y), (width, y), (45, 40, 40), 1)

    # Determine face positions based on face_count
    # Distribute them spaced out
    face_radius = int(min(width, height) * 0.1)
    centers = []
    if face_count == 1:
        centers.append((width // 2, height // 2))
    elif face_count == 3:
        centers = [
            (width // 4, height // 2),
            (width // 2, height // 2),
            (3 * width // 4, height // 2)
        ]
    elif face_count >= 5:
        # 5 or 10 faces distributed in grid rows
        rows = 2 if face_count == 10 else 1
        cols = 5
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < face_count:
                    cx = int((c + 0.5) * (width / cols))
                    cy = int((r + 0.5) * (height / rows))
                    centers.append((cx, cy))
                    idx += 1

    # Draw the faces
    for i, center in enumerate(centers):
        # Change face color per face index for variance
        color = (120, 160, 240) if i % 2 == 0 else (180, 120, 240)
        label = f"ID-{i+1:03d}"
        draw_synthetic_face(img, center, face_radius, color, label)
        
        # Apply occlusion to first face if enabled
        if is_occlude and i == 0:
            cx, cy = center
            # Draw a black blocking bar (occlusion) over the nose/mouth
            cv2.rectangle(img, (cx - face_radius, cy), (cx + face_radius, cy + face_radius), (10, 10, 10), -1)
            cv2.putText(img, "OCCLUDED", (cx - face_radius, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Apply lighting effects
    if lighting == "low_light":
        # Scale brightness down severely
        img = (img.astype(np.float32) * 0.25).astype(np.uint8)
    elif lighting == "side_lighting":
        # Create a horizontal brightness gradient (left bright, right dark)
        gradient = np.tile(np.linspace(1.2, 0.2, width).reshape(1, width, 1), (height, 1, 3))
        img = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    elif lighting == "strong_shadow":
        # Cut half the image brightness
        img[:, width//2:] = (img[:, width//2:].astype(np.float32) * 0.15).astype(np.uint8)

    # Apply blur
    if is_blur:
        img = cv2.GaussianBlur(img, (15, 15), 0)

    # Label pose/lighting text on canvas for clarity
    text_info = f"Res: {width}x{height} | Faces: {face_count} | Pose: {pose_angle}deg | Light: {lighting}"
    cv2.putText(img, text_info, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img

def create_synthetic_video(video_path, width=640, height=480, num_frames=120):
    """
    Creates a valid synthetic dynamic face-tracking MP4 video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
    
    face_radius = int(height * 0.12)
    
    # Generate frame-by-frame movement
    for frame_idx in range(num_frames):
        # Black background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (width, height), (25, 20, 20), -1)
        
        # Moving face 1 (sinusoidal)
        cx1 = int(width * 0.3 + np.sin(frame_idx * 0.08) * width * 0.15)
        cy1 = int(height * 0.5 + np.cos(frame_idx * 0.05) * height * 0.1)
        draw_synthetic_face(img, (cx1, cy1), face_radius, (100, 150, 250), "Track-01")
        
        # Moving face 2 (diagonal)
        cx2 = int(width * 0.7 - (frame_idx / num_frames) * width * 0.2)
        cy2 = int(height * 0.3 + (frame_idx / num_frames) * height * 0.4)
        draw_synthetic_face(img, (cx2, cy2), int(face_radius*0.9), (200, 100, 250), "Track-02")
        
        # Text log
        cv2.putText(img, f"STREAM BENCHMARK | Frame: {frame_idx+1}/{num_frames}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        writer.write(img)
        
    writer.release()
    print(f"Synthetic benchmark video successfully written to: {video_path}")

def main():
    root_dir = "data/benchmark_dataset"
    gallery_dir = os.path.join(root_dir, "gallery")
    query_dir = os.path.join(root_dir, "query")
    
    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    print("=" * 60)
    print("      FACE_NET BENCHMARKING SUITE - SYNTHETIC DATASET GENERATOR")
    print("=" * 60)

    # 1. Create Gallery (Registered Identities)
    print("Generating registered identities (Gallery)...")
    gallery_manifest = []
    num_identities = 10
    
    for i in range(num_identities):
        g_id = f"ID-{i+1:03d}"
        img = create_synthetic_image(160, 160, face_count=1, pose_angle=0, lighting="normal")
        filename = f"person_{i+1:03d}.jpg"
        filepath = os.path.join(gallery_dir, filename)
        cv2.imwrite(filepath, img)
        
        gallery_manifest.append({
            "identity": g_id,
            "filepath": filepath,
            "label_idx": i
        })
    print(f"Created {num_identities} gallery identity profiles.")

    # 2. Create Queries covering various poses, lightings, resolutions, and face counts
    print("Generating test query images...")
    query_manifest = []
    
    # Resolutions matrix
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    # Poses & Lightings combinations
    test_cases = [
        {"pose": 0, "lighting": "normal", "blur": False, "occlude": False},
        {"pose": 0, "lighting": "low_light", "blur": False, "occlude": False},
        {"pose": 30, "lighting": "side_lighting", "blur": False, "occlude": False},
        {"pose": 50, "lighting": "strong_shadow", "blur": False, "occlude": True},
        {"pose": 0, "lighting": "normal", "blur": True, "occlude": False},
    ]
    face_counts = [1, 3, 5, 10]
    
    idx = 0
    for res in resolutions:
        w, h = res
        for case in test_cases:
            for count in face_counts:
                img = create_synthetic_image(
                    w, h, count, 
                    pose_angle=case["pose"], 
                    lighting=case["lighting"],
                    is_blur=case["blur"],
                    is_occlude=case["occlude"]
                )
                
                filename = f"query_{idx+1:04d}_{w}x{h}_face{count}_angle{case['pose']}_{case['lighting']}.jpg"
                filepath = os.path.join(query_dir, filename)
                cv2.imwrite(filepath, img)
                
                # Register ground truth metadata
                query_manifest.append({
                    "id": idx + 1,
                    "filepath": filepath,
                    "width": w,
                    "height": h,
                    "face_count": count,
                    "pose_angle": case["pose"],
                    "lighting": case["lighting"],
                    "blurred": case["blur"],
                    "occluded": case["occlude"],
                    "ground_truth_identities": [f"ID-{x+1:03d}" for x in range(count)] # first 'count' identities
                })
                idx += 1
                
    print(f"Created {len(query_manifest)} total query images.")

    # 3. Create Video Stream File
    video_path = os.path.join(root_dir, "video_sample.mp4")
    print("Generating synthetic live video stream (video_sample.mp4)...")
    create_synthetic_video(video_path)

    # 4. Save metadata manifests
    metadata = {
        "gallery": gallery_manifest,
        "queries": query_manifest,
        "video": video_path
    }
    
    metadata_path = os.path.join(root_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Metadata manifest successfully saved to: {metadata_path}")
    print("=" * 60)
    print("Dataset generation complete. Environment ready for benchmarking!")

if __name__ == "__main__":
    main()
