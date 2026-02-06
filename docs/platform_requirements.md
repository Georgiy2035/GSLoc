# Platform Requirements

This document defines the supported operating systems, hardware architectures, and software stack versions for developing and deploying the multimodal place recognition system.

## OS / ROS

- **Current:** Ubuntu 22.04 + ROS 2 Humble
- **Near future:** Ubuntu 24.04 + ROS 2 Jazzy
- **Architectures:** x86_64 and aarch64 (Jetson Orin / Jetson Thor)

## Python

- **Language target:** Python 3.10 features
- **Runtime range:** 3.10 ≤ Python < 3.13 (3.10 on Orin, 3.12 on Thor)
- Tested versions:
    - x86: 3.10

## CUDA / GPU

- **Minimum CUDA:** 12.1+ (compatible with minimal PyTorch 2.4.1 version)
- Tested versions:
    - x86: CUDA 12.6

## NumPy

- **Target API:** NumPy 2.x semantics (no deprecated aliases like np.bool, np.int etc.)

## FAISS

- **Minimum version:** We think any version of Faiss should be sufficient, but we recommend using the latest version.
- Tested versions:
    - x86: Faiss-CPU 1.13.1 from PyPI (unofficial wheel for CPU)

## Deep learning stack (recommended, not strictly required)

- **Minimum version:** PyTorch ≥ 2.4.1 on x86 should be sufficient (wheels for CUDA 12.1/12.4, NumPy 2-compatible)
- Tested versions:
    - x86: 2.7.1 with CUDA 12.6  
