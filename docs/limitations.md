# Benchmarking Limitations and Production Bottlenecks

This document addresses the technical constraints, trade-offs, and critical system ceilings encountered when running high-performance face recognition pipelines on the NVIDIA Jetson AGX Orin 64GB.

---

## 1. The Face Detection Bottleneck (MTCNN vs. Embeddings)

While optimizing the FaceNet embedding network using TensorRT drops embedding latency to **0.78 ms per face**, the **Face Detection (MTCNN)** stage remains the primary bottleneck for the end-to-end pipeline:
- MTCNN is a multi-stage network (P-Net, R-Net, and O-Net) that processes image pyramids to support scale invariance.
- At 1080p (1920x1080) resolution, constructing the image pyramid and scanning it takes approximately **22-28 ms** on the GPU.
- Therefore, even with a sub-millisecond embedder, the end-to-end pipeline throughput is limited to **~35-40 FPS** due to detection overhead.

### Mitigation Strategies:
1. **Detection Skip-Frames**: In production, do not run face detection on every frame. Use a KCF (Kernelized Correlation Filter) or landmark-based tracker to follow faces across 3-5 frames, running MTCNN detection only on every 5th frame or when tracking confidence drops.
2. **Region of Interest (ROI) Pooling**: Crop and search smaller sub-regions of the frame based on prior coordinates rather than scanning the full 1080p canvas.

---

## 2. Shared Memory Bandwidth Ceiling

Although the Jetson AGX Orin features a spacious **64GB of RAM**, this pool is shared between the CPU, GPU, and deep learning engines. 
- The total memory bandwidth is **204.8 GB/s**.
- While sufficient for single-stream pipelines, a dense multi-camera setup (e.g., 4 to 8 concurrent 1080p RTSP streams) will saturate this bus due to continuous image transpositions, color conversions, and memory copies.
- If memory bandwidth is saturated, processing latency spikes across all backends.

### Mitigation Strategies:
1. Use **in-place NumPy operations** during preprocessing to avoid redundant memory allocations.
2. Maintain images in GPU memory as much as possible using hardware-accelerated decoding (e.g., NVDEC / GStreamer pipelines) instead of copying frames to host CPU memory.

---

## 3. Dynamic Batching Overhead in TensorRT

TensorRT engines compile with highly optimized kernels tailored to specific input dimensions.
- To support varying face counts (1 to 10), we compile the TensorRT engine with **Dynamic Profiles** (supporting batch sizes from 1 to 16).
- If the runtime batch size changes frequently (e.g., frame 1 has 1 face, frame 2 has 3 faces), TensorRT must reconfigure binding shapes and select different execution kernels, which can introduce occasional micro-stalls.

### Mitigation Strategies:
1. **Batch Padding**: Pad smaller batches to the nearest power-of-two (e.g., if you detect 3 faces, pad the input tensor with dummy data to batch 4) to reuse compiled kernels.
2. **Preallocation**: Preallocate GPU buffers for the maximum expected batch size (e.g., 16) to avoid runtime buffer reallocations.

---

## 4. Single-Stream CUDA Synchronization

In Variant 4, Host-to-Device and Device-to-Host memory copies run asynchronously on a dedicated CUDA stream:
- `cuda.memcpy_htod_async(cuda_in, host_in, stream)`
- `context.execute_async_v3(stream_handle=stream.handle)`
- `cuda.memcpy_dtoh_async(host_out, cuda_out, stream)`
- `stream.synchronize()`

While this prevents blocking the CPU during inference, calling `stream.synchronize()` still introduces a sync boundary. In multi-stream environments, scheduling overlapping pipelines across multiple CUDA streams (or utilizing NVIDIA DeepStream SDK) is necessary to achieve full hardware saturation.
