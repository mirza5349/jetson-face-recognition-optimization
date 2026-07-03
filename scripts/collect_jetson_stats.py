# scripts/collect_jetson_stats.py
import os
import sys
import time
import argparse

# Add repository root to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.jetson_monitor import JetsonStatsMonitor

def main():
    parser = argparse.ArgumentParser(description="Real-time terminal dashboard for Jetson AGX Orin metrics.")
    parser.add_argument("--interval", type=int, default=500, help="Polling interval in milliseconds.")
    parser.add_argument("--mock", action="store_true", help="Force telemetry mock simulation.")
    args = parser.parse_args()

    # If not on Tegra/Linux, we automatically mock
    mock_flag = args.mock or (os.name == 'nt')
    
    print("=" * 65)
    print("     NVIDIA JETSON AGX ORIN 64GB - HARDWARE TELEMETRY")
    print(f"     Polling Interval: {args.interval}ms | Mock Mode: {'Active' if mock_flag else 'Disabled'}")
    print("=" * 65)
    print(f"{'Time':8s} | {'CPU %':6s} | {'GPU %':6s} | {'RAM MB':8s} | {'VRAM MB':8s} | {'Power W':7s} | {'Temp C':6s} | {'Throt':5s}")
    print("-" * 65)

    monitor = JetsonStatsMonitor(interval_ms=args.interval, mock=mock_flag)
    
    # Simulate a dynamic workload context so the mock telemetry changes over time
    if mock_flag:
        monitor.set_workload_context(backend="pytorch_fp16", batch_size=4, face_count=3)
        
    monitor.start()

    try:
        while True:
            # Grab current stats
            time.sleep(args.interval / 1000.0)
            stats = monitor.current_stats
            
            t_str = time.strftime("%H:%M:%S")
            cpu_str = f"{stats['cpu_usage_percent']:5.1f}%"
            gpu_str = f"{stats['gpu_usage_percent']:5.1f}%"
            ram_str = f"{stats['ram_usage_mb']:7.1f}"
            vram_str = f"{stats['gpu_memory_mb']:7.1f}"
            pwr_str = f"{stats['power_watts']:5.2f}W"
            tmp_str = f"{stats['temperature_c']:4.1f}C"
            throt_str = "YES" if stats["thermal_throttling"] else "NO"
            
            print(f"{t_str} | {cpu_str} | {gpu_str} | {ram_str} | {vram_str} | {pwr_str} | {tmp_str} | {throt_str}")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\nStopping stats collection.")
    finally:
        monitor.stop()
        print("Telemetry loop closed.")

if __name__ == "__main__":
    main()
