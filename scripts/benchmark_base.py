# scripts/benchmark_base.py
import os
import sys
import time
import json
import argparse
import numpy as np
import cv2

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_loader import load_combined_configs
from src.utils.jetson_monitor import JetsonStatsMonitor
from src.pipelines.base_pipeline import BaseFaceNetPipeline

def run_benchmark(config_path, hw_config_path, num_frames=50, mock=False):
    configs = load_combined_configs(config_path, hw_config_path)
    b_cfg = configs["benchmark"]
    h_cfg = configs["hardware"]
    
    print("=" * 60)
    print(f"  BENCHMARK - BASE PIPELINE (PyTorch FP32 Sync)")
    print("=" * 60)
    
    # Initialize hardware stats monitor
    monitor = JetsonStatsMonitor(interval_ms=100, mock=mock)
    monitor.set_workload_context(backend="pytorch_fp32", batch_size=1)
    monitor.start()

    # Initialize Base Pipeline
    device = "cuda" if (h_cfg["inference"]["use_gpu"] and not mock) else "cpu"
    pipeline = BaseFaceNetPipeline(device=device, mock=mock)

    # Prepare dummy database for search benchmarks
    db_size = 100
    db_embs = np.random.normal(0, 1, (db_size, 512)).astype(np.float32)
    # L2 normalize
    db_embs = db_embs / np.linalg.norm(db_embs, axis=1, keepdims=True)

    # Prepare a standard input image (640x480)
    img_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Warm-up iterations
    print("Warming up pipeline...")
    for _ in range(5):
        _ = pipeline.process_frame(img_rgb, database_embs=db_embs)

    print(f"Running benchmark for {num_frames} frames...")
    latencies = []
    det_latencies = []
    prep_latencies = []
    emb_latencies = []
    match_latencies = []

    for i in range(num_frames):
        t0 = time.perf_counter()
        out = pipeline.process_frame(img_rgb, database_embs=db_embs)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
        m = out["metrics"]
        det_latencies.append(m["detection_latency_ms"])
        prep_latencies.append(m["preprocessing_latency_ms"])
        emb_latencies.append(m["embedding_latency_ms"])
        match_latencies.append(m["matching_latency_ms"])
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_frames} frames... Current Avg Latency: {np.mean(latencies):.2f} ms")

    # Stop monitor and fetch hardware stats
    hw_history = monitor.stop()
    hw_summary = monitor.get_summary()

    avg_lat = np.mean(latencies)
    fps = 1000.0 / avg_lat

    results = {
        "backend": "pytorch_fp32",
        "precision": "fp32",
        "resolution": "640x480",
        "batch_size": 1,
        "number_of_faces": 1,
        "pose_angle": 0,
        "lighting_condition": "normal",
        "detection_latency_ms": float(np.mean(det_latencies)),
        "preprocessing_latency_ms": float(np.mean(prep_latencies)),
        "embedding_latency_ms": float(np.mean(emb_latencies)),
        "matching_latency_ms": float(np.mean(match_latencies)),
        "total_latency_ms": float(avg_lat),
        "fps": float(fps),
        "p50_latency": float(np.percentile(latencies, 50)),
        "p90_latency": float(np.percentile(latencies, 90)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "p99_latency": float(np.percentile(latencies, 99)),
        "cpu_usage_percent": hw_summary["cpu_avg_percent"],
        "gpu_usage_percent": hw_summary["gpu_avg_percent"],
        "ram_usage_mb": hw_summary["ram_avg_mb"],
        "gpu_memory_mb": hw_summary["gpu_mem_avg_mb"],
        "power_watts": hw_summary["power_avg_watts"],
        "temperature_c": hw_summary["temp_avg_c"],
        "thermal_throttling": bool(hw_summary["thermal_throttling_events"] > 0)
    }

    print("-" * 60)
    print(f"Average Pipeline Latency:  {avg_lat:.2f} ms")
    print(f"Average Pipeline FPS:      {fps:.2f}")
    print(f"Average CPU Load:          {results['cpu_usage_percent']:.1f}%")
    print(f"Average GPU Load:          {results['gpu_usage_percent']:.1f}%")
    print(f"Average Power Draw:        {results['power_watts']:.2f} W")
    print(f"Average Temp:              {results['temperature_c']:.1f} C")
    print("=" * 60)

    # Save results
    os.makedirs("results/raw", exist_ok=True)
    out_path = "results/raw/benchmark_base.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Raw benchmark metrics saved to {out_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark FaceNet Base PyTorch FP32 Pipeline.")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml", help="Path to benchmark YAML.")
    parser.add_argument("--hw-config", type=str, default="configs/jetson_agx_orin.yaml", help="Path to hardware YAML.")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to benchmark.")
    parser.add_argument("--mock", action="store_true", help="Force mock execution for local environments.")
    args = parser.parse_args()

    run_benchmark(args.config, args.hw_config, num_frames=args.frames, mock=args.mock)

if __name__ == "__main__":
    main()
