# scripts/export_onnx.py
import os
import sys
import argparse

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings.embedder import FaceEmbedder

def main():
    parser = argparse.ArgumentParser(description="Export FaceNet InceptionResnetV1 PyTorch Model to ONNX.")
    parser.add_argument("--output", type=str, default="models/facenet.onnx", help="Output path for exported ONNX model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for tracing (cpu or cuda).")
    parser.add_argument("--mock", action="store_true", help="Force export of a dummy trace model.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting FaceNet model to ONNX format at: {args.output}")
    print(f"Device: {args.device}")

    # Use FaceEmbedder's built-in robust export/bootstrap logic
    embedder = FaceEmbedder(backend="onnx", model_path=args.output, device=args.device, mock=args.mock)
    
    if os.path.exists(args.output):
        print(f"Success! Model successfully written to {args.output}")
        # Print file size
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"ONNX Model File Size: {size_mb:.2f} MB")
    else:
        print("Error: ONNX export failed. Output file not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
