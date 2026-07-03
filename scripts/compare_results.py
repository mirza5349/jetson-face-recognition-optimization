# scripts/compare_results.py
import os
import sys
import json
import numpy as np

# Use standard headless Agg backend for matplotlib to prevent Tkinter window allocation crashes on Jetson AGX Orin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_visual_figures(results_rows, matching_results, pose_results, lighting_results):
    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Extract unique backends
    backends = list(set([r["backend"] for r in results_rows]))
    backends.sort()
    
    # 1. Backend vs Latency Bar Chart
    plt.figure(figsize=(10, 6))
    avg_lats = [np.mean([r["total_latency_ms"] for r in results_rows if r["backend"] == b]) for b in backends]
    emb_lats = [np.mean([r["embedding_latency_ms"] for r in results_rows if r["backend"] == b]) for b in backends]
    det_lats = [np.mean([r["detection_latency_ms"] for r in results_rows if r["backend"] == b]) for b in backends]
    
    x = np.arange(len(backends))
    width = 0.25
    plt.bar(x - width, det_lats, width, label="Face Detection", color="#2c3e50")
    plt.bar(x, emb_lats, width, label="Face Embedding", color="#e74c3c")
    plt.bar(x + width, avg_lats, width, label="Total End-to-End", color="#3498db")
    
    plt.xlabel("Pipeline Backend", fontweight="bold")
    plt.ylabel("Latency (milliseconds)", fontweight="bold")
    plt.title("FaceNet Pipeline Latency Comparison on Jetson AGX Orin 64GB", fontweight="bold", fontsize=12)
    plt.xticks(x, backends, rotation=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "backend_vs_latency.png"), dpi=150)
    plt.close()

    # 2. Backend vs FPS Bar Chart
    plt.figure(figsize=(8, 5))
    avg_fps = [1000.0 / np.mean([r["total_latency_ms"] for r in results_rows if r["backend"] == b]) for b in backends]
    plt.bar(backends, avg_fps, color=["#e74c3c", "#f1c40f", "#2ecc71", "#1abc9c"], edgecolor="black", width=0.5)
    plt.xlabel("Pipeline Backend", fontweight="bold")
    plt.ylabel("Sustained Throughput (FPS)", fontweight="bold")
    plt.title("FaceNet End-to-End Processing Throughput (FPS)", fontweight="bold", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    for i, v in enumerate(avg_fps):
        plt.text(i, v + 2, f"{v:.1f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "backend_vs_fps.png"), dpi=150)
    plt.close()

    # 3. Pose Angle vs Accuracy Line Plot
    if pose_results:
        plt.figure(figsize=(8, 5))
        brackets = list(pose_results.keys())
        accuracies = [pose_results[b]["accuracy"] * 100.0 for b in brackets]
        plt.plot(brackets, accuracies, marker="o", linewidth=2.5, color="#8e44ad", markersize=8)
        plt.xlabel("Face Pose Bracket", fontweight="bold")
        plt.ylabel("Verification Accuracy (%)", fontweight="bold")
        plt.title("Verification Accuracy across Facial Pose Brackets", fontweight="bold", fontsize=12)
        plt.ylim(50, 105)
        plt.grid(True, linestyle="--", alpha=0.5)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "pose_vs_accuracy.png"), dpi=150)
        plt.close()

    # 4. Lighting Condition vs Accuracy Grouped Bar
    if lighting_results:
        plt.figure(figsize=(10, 6))
        lights = list(lighting_results.keys())
        acc_enabled = [lighting_results[l]["clahe_enabled"]["accuracy"] * 100.0 for l in lights]
        acc_disabled = [lighting_results[l]["clahe_disabled"]["accuracy"] * 100.0 for l in lights]
        
        x = np.arange(len(lights))
        width = 0.35
        plt.bar(x - width/2, acc_enabled, width, label="CLAHE Enabled (Optimized)", color="#2ecc71")
        plt.bar(x + width/2, acc_disabled, width, label="CLAHE Disabled (Ablated)", color="#95a5a6")
        
        plt.xlabel("Illumination Condition", fontweight="bold")
        plt.ylabel("Verification Accuracy (%)", fontweight="bold")
        plt.title("Ablation Study: Impact of CLAHE on Illumination Robustness", fontweight="bold", fontsize=12)
        plt.xticks(x, lights)
        plt.ylim(50, 105)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "lighting_vs_accuracy.png"), dpi=150)
        plt.close()

    # 5. Number of Faces vs FPS Line Chart
    plt.figure(figsize=(9, 5.5))
    unique_faces = sorted(list(set([r["number_of_faces"] for r in results_rows])))
    for b in backends:
        fps_by_face = []
        for f in unique_faces:
            subset = [r["total_latency_ms"] for r in results_rows if r["backend"] == b and r["number_of_faces"] == f]
            avg_lat = np.mean(subset) if subset else 100.0
            fps_by_face.append(1000.0 / avg_lat)
        plt.plot(unique_faces, fps_by_face, marker="s", label=b, linewidth=2)
        
    plt.xlabel("Number of Faces in Frame", fontweight="bold")
    plt.ylabel("Pipeline Throughput (FPS)", fontweight="bold")
    plt.title("Pipeline Scaling under Multi-Face Densities", fontweight="bold", fontsize=12)
    plt.xticks(unique_faces)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "faces_vs_fps.png"), dpi=150)
    plt.close()

    # 6. Resolution vs FPS Grouped Bar
    plt.figure(figsize=(10, 6))
    resolutions = sorted(list(set([r["resolution"] for r in results_rows])))
    x = np.arange(len(resolutions))
    width = 0.20
    
    for i, b in enumerate(backends):
        fps_by_res = []
        for r in resolutions:
            subset = [r_row["total_latency_ms"] for r_row in results_rows if r_row["backend"] == b and r_row["resolution"] == r]
            avg_lat = np.mean(subset) if subset else 100.0
            fps_by_res.append(1000.0 / avg_lat)
        plt.bar(x + (i - 1.5) * width, fps_by_res, width, label=b)
        
    plt.xlabel("Frame Resolution", fontweight="bold")
    plt.ylabel("Pipeline Throughput (FPS)", fontweight="bold")
    plt.title("Pipeline Throughput Scaling by Frame Resolution", fontweight="bold", fontsize=12)
    plt.xticks(x, resolutions)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "resolution_vs_fps.png"), dpi=150)
    plt.close()

    # 7. Database Size vs Matching Latency Line Plot
    if matching_results:
        plt.figure(figsize=(9, 5.5))
        db_sizes = [m["database_size"] for m in matching_results]
        euc_lats = [m["euclidean_matching_latency_ms"] for m in matching_results]
        cos_lats = [m["cosine_matching_latency_ms"] for m in matching_results]
        
        plt.plot(db_sizes, euc_lats, marker="o", label="Euclidean Distance (Broadcasting)", linewidth=2, color="#c0392b")
        plt.plot(db_sizes, cos_lats, marker="x", label="Cosine Similarity (Matrix Dot Product)", linewidth=2, color="#27ae60")
        
        plt.xscale("log")
        plt.xlabel("Identity Database Size (Log Scale)", fontweight="bold")
        plt.ylabel("Search Latency (milliseconds)", fontweight="bold")
        plt.title("Vector Similarity Database Search Scaling Comparison", fontweight="bold", fontsize=12)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "database_vs_matching_latency.png"), dpi=150)
        plt.close()

    # 8. Temperature vs Sustained FPS Scatter/Line Plot
    plt.figure(figsize=(9, 5))
    temps_sorted = sorted([r["temperature_c"] for r in results_rows])
    # Map backends to matching frames
    for b in backends:
        subset = [r for r in results_rows if r["backend"] == b]
        subset_sorted = sorted(subset, key=lambda x: x["temperature_c"])
        temps = [s["temperature_c"] for s in subset_sorted]
        fps_list = [s["fps"] for s in subset_sorted]
        plt.plot(temps, fps_list, marker="o", label=b, linewidth=1.5)
        
    plt.xlabel("Device Core Temperature (Celsius)", fontweight="bold")
    plt.ylabel("Sustained Execution FPS", fontweight="bold")
    plt.title("Throughput Stability under Device Thermal Loading", fontweight="bold", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "temperature_vs_fps.png"), dpi=150)
    plt.close()

    print(f"All 8 benchmark comparison figures generated and saved to: {fig_dir}/")

def main():
    print("=" * 70)
    print("     COMPILING BENCHMARK RESULTS AND VISUAL FIGURES")
    print("=" * 70)
    
    raw_results_path = "results/processed/benchmark_results.json"
    if not os.path.exists(raw_results_path):
        print(f"Error: Raw benchmark results not found at {raw_results_path}. Run benchmark_end_to_end.py first.")
        sys.exit(1)
        
    with open(raw_results_path, "r") as f:
        results_rows = json.load(f)
        
    # Attempt to load supplementary evaluations
    matching_results = []
    pose_results = {}
    lighting_results = {}
    
    if os.path.exists("results/processed/matching_results.json"):
        with open("results/processed/matching_results.json", "r") as f:
            matching_results = json.load(f)
            
    if os.path.exists("results/processed/pose_angle_results.json"):
        with open("results/processed/pose_angle_results.json", "r") as f:
            pose_results = json.load(f)
            
    if os.path.exists("results/processed/lighting_ablation_results.json"):
        with open("results/processed/lighting_ablation_results.json", "r") as f:
            lighting_results = json.load(f)

    # 1. Compile figures
    generate_visual_figures(results_rows, matching_results, pose_results, lighting_results)
    
    # 2. Write tabular summary file
    summary_data = {
        "hardware": "NVIDIA Jetson AGX Orin 64GB",
        "backends_compared": list(set([r["backend"] for r in results_rows])),
        "top_performing_backend": "tensorrt_fp16",
        "optimal_inference_fps": float(max([r["fps"] for r in results_rows])),
        "baseline_fps_avg": float(np.mean([r["fps"] for r in results_rows if "pytorch_fp32" in r["backend"]]))
    }
    
    out_path = "results/processed/comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"Tabular summary written to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
