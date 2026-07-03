# scripts/check_environment.py
import os
import sys
import subprocess
import platform

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error / Not Configured (Exit Code {e.returncode})"
    except Exception as e:
        return f"Error: {str(e)}"

def get_jetson_info():
    """
    Attempts to read L4T Tegra Release or jetson_release details.
    """
    info = {}
    if os.path.exists("/etc/nv_tegra_release"):
        info["is_jetson"] = True
        with open("/etc/nv_tegra_release", "r") as f:
            info["nv_tegra"] = f.readline().strip()
        # Parse nvpmodel status
        info["power_mode"] = run_cmd("nvpmodel -q")
    else:
        info["is_jetson"] = False
        info["power_mode"] = "N/A (Non-Tegra Hardware)"
    return info

def main():
    print("=" * 60)
    print("      FACE_NET BENCHMARKING SUITE - ENVIRONMENT AUDIT")
    print("=" * 60)

    # 1. Host OS and Hardware
    print(f"Host OS:             {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version:      {sys.version.split()[0]}")
    
    # 2. Jetson Specifics
    j_info = get_jetson_info()
    print(f"Is NVIDIA Jetson:    {'YES' if j_info['is_jetson'] else 'NO (Local Fallback Mode)'}")
    if j_info['is_jetson']:
        print(f"Tegra Release Info:  {j_info.get('nv_tegra', 'Unknown')}")
    print(f"Active Power Mode:   {j_info['power_mode']}")

    # 3. CUDA, cuDNN, and Compiler Versions
    cuda_path = "N/A"
    nvcc_version = run_cmd("nvcc --version")
    if "nvcc" in nvcc_version:
        for line in nvcc_version.split("\n"):
            if "release" in line:
                cuda_path = line.strip()
    print(f"CUDA Compiler (NVCC):{cuda_path}")

    # 4. PyTorch & GPU details
    try:
        import torch
        print(f"PyTorch Version:     {torch.__version__}")
        print(f"PyTorch CUDA Available: {'YES' if torch.cuda.is_available() else 'NO'}")
        if torch.cuda.is_available():
            print(f"  Primary GPU Name:  {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Runtime:      {torch.version.cuda}")
            print(f"  cuDNN Build:       {torch.backends.cudnn.version()}")
    except ImportError:
        print("PyTorch Version:     NOT INSTALLED")

    # 5. ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime:        {ort.__version__}")
        print(f"ORT Providers:       {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime:        NOT INSTALLED")

    # 6. TensorRT
    try:
        import tensorrt as trt
        print(f"TensorRT SDK:        {trt.__version__}")
    except ImportError:
        print("TensorRT SDK:        NOT INSTALLED")

    # 7. OpenCV
    try:
        import cv2
        print(f"OpenCV Library:      {cv2.__version__}")
    except ImportError:
        print("OpenCV Library:      NOT INSTALLED")

    print("=" * 60)
    print("Audit Complete.")

if __name__ == "__main__":
    main()
