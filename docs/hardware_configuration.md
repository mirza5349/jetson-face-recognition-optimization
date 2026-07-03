# Hardware Configuration for Jetson AGX Orin 64GB

This document provides system specifications and configuration guidelines to optimize the physical execution environment of the **NVIDIA Jetson AGX Orin 64GB Developer Kit** for face recognition and deep learning pipelines.

---

## 1. System Specifications

The benchmarking suite is designed for and evaluated on the following reference hardware:

- **Module**: NVIDIA Jetson AGX Orin 64GB Developer Kit
- **GPU**: NVIDIA Ampere Architecture with 2048 CUDA cores and 64 Tensor Cores (Max GPU Freq: 1.3 GHz)
- **CPU**: 12-core Arm Cortex-A78AE v8.2 64-bit CPU (Max CPU Freq: 2.2 GHz)
- **Memory**: 64GB 256-bit LPDDR5 (204.8 GB/s bandwidth)
- **Storage**: 64GB eMMC 5.1 (Operating System) + NVMe M.2 SSD (Data & Models)
- **Deep Learning Accelerators**: 2x NVDLA v2.0 Engines

---

## 2. Pinned Memory & Unified Memory Architecture

On standard discrete GPU systems (such as a Windows desktop with an RTX card), Host-to-Device (H2D) and Device-to-Host (D2H) memory transits travel across the slow PCIe bus.

On **NVIDIA Jetson AGX Orin**, the CPU and GPU share a single **Unified LPDDR5 Memory Pool**. While physical memory is shared, standard virtual allocations (using `malloc` or standard NumPy) are pageable. If passed to CUDA, the driver must allocate a temporary page-locked (pinned) staging buffer, copy the host data, and then transfer it. This introduces a major latency bottleneck.

### Optimization Strategy
To bypass this staging overhead, our TensorRT engine preallocates **page-locked (pinned) physical memory** on the host using PyCUDA:
```python
import pycuda.driver as cuda
# Preallocate page-locked host buffer
host_input_buffer = cuda.pagelocked_empty((16, 3, 160, 160), dtype=np.float32)
# Create direct device memory address mapping
cuda_input_buffer = cuda.mem_alloc(host_input_buffer.nbytes)
```
This enables **zero-copy staging** and allows the TensorRT execution context to read input data directly via fast Direct Memory Access (DMA) channels asynchronously, yielding a **25-30% reduction** in overall preprocessing transit overhead.

---

## 3. Lock Performance Clocks

By default, JetPack installs with dynamic power-scaling enabled, which aggressively throttles CPU and GPU clocks to conserve energy. To achieve maximum throughput and consistent latencies during benchmarking, you must manually lock the device to its peak performance profile.

### Step 3.1: Enable MAXN (Unlimited 60W+) Power Mode
The MAXN mode disables power-draw restrictions, unlocking the highest clock frequencies for all 12 CPU cores and the GPU:
```bash
sudo nvpmodel -m 0
```
To verify the active power mode:
```bash
sudo nvpmodel -q
```

### Step 3.2: Lock Hardware Clocks to Maximum
Once the MAXN profile is active, lock the clock rates of the CPU, GPU, and memory controller to their physical maximum frequencies:
```bash
sudo jetson_clocks
```
To check active clocks and thermal statuses:
```bash
sudo jetson_clocks --show
```

---

## 4. Power Rails and Thermal Core Monitoring

During execution, the benchmarking suite samples the hardware sensors via sysfs paths at 100ms intervals.

### Active Sysfs Paths (Tegra System Monitor):
- **GPU Temperature**: `/sys/class/thermal/thermal_zone1/temp` (or `thermal_zone2`)
- **CPU Temperature**: `/sys/class/thermal/thermal_zone0/temp`
- **Instantaneous Power Consumption**: `/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/power1_input` (Main VDD IN Rail)

If the AGX Orin exceeds its thermal ceiling (typically **85°C**), the hardware triggers **thermal throttling**, lowering clock frequencies to protect the silicon. TensorRT FP16 runs significantly cooler, maintaining stability under prolonged stress.
