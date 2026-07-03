# scripts/build_tensorrt.py
import os
import sys
import argparse

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

def build_engine(onnx_path, engine_path, max_batch_size=16, use_fp16=True):
    """
    Compiles an ONNX model into a serialized TensorRT Engine.
    Compatible with both TensorRT 8.x and TensorRT 10.x API standards.
    """
    print(f"Starting TensorRT Engine compilation...")
    print(f"ONNX Model: {onnx_path}")
    print(f"Engine Target: {engine_path}")
    print(f"Target Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Max Batch Size: {max_batch_size}")

    if not HAS_TRT:
        print("TensorRT library not found. Creating a [MOCK ENGINE] file for bootstrap.")
        # Create a mock engine file containing a tiny header text
        with open(engine_path, "w") as f:
            f.write(f"MOCK_TENSORRT_ENGINE_FP16_BATCH_16_FACENET_V2_0\nONNX:{onnx_path}\n")
        print(f"Mock engine successfully written to {engine_path}")
        return True

    # Real TensorRT Builder Setup
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) = 1 (explicit batch is required for ONNX)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model {onnx_path} does not exist. Run export_onnx.py first.")
        return False

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    config = builder.create_builder_config()
    
    # Set workspace limit (TRT 10.x uses set_memory_pool_limit, TRT 8.x uses max_workspace_size)
    # Give it 1GB of workspace
    workspace_bytes = 1 << 30 
    
    trt_version = [int(x) for x in trt.__version__.split('.')]
    if trt_version[0] >= 10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    # Configure precision flags
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 precision enabled on target hardware.")
        else:
            print("WARNING: Hardware does not support fast FP16. Compiling with default precision.")

    # Configure optimization profiles for dynamic batch sizes
    profile = builder.create_optimization_profile()
    # "input" is the standard tensor name for our FaceNet ONNX model
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        (1, 3, 160, 160),       # Minimum batch size
        (4, 3, 160, 160),       # Optimum batch size (typical load)
        (max_batch_size, 3, 160, 160) # Maximum batch size
    )
    config.add_optimization_profile(profile)

    # Compile and Serialize Network
    print("Compiling network. This may take a few minutes on Jetson Orin...")
    
    if trt_version[0] >= 10:
        # TRT 10.x Engine build & serialization
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Error: Engine build failed.")
            return False
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
    else:
        # TRT 8.x Engine build & serialization
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Error: Engine build failed.")
            return False
        serialized_engine = engine.serialize()
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

    print(f"Success! Compiled TensorRT engine successfully written to: {engine_path}")
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"Compiled Engine File Size: {size_mb:.2f} MB")
    return True

def main():
    parser = argparse.ArgumentParser(description="Compile exported FaceNet ONNX Model to FP16 TensorRT Engine.")
    parser.add_argument("--onnx", type=str, default="models/facenet.onnx", help="Path to input ONNX file.")
    parser.add_argument("--engine", type=str, default="models/facenet_fp16.engine", help="Path to write compiled TensorRT engine.")
    parser.add_argument("--max-batch", type=int, default=16, help="Maximum dynamic batch size to configure.")
    parser.add_argument("--fp32", action="store_true", help="Compile in FP32 instead of default FP16.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.engine), exist_ok=True)
    success = build_engine(args.onnx, args.engine, max_batch_size=args.max_batch, use_fp16=not args.fp32)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
