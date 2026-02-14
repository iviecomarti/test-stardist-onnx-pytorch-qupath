# EXPERIMENT: Running StarDist models in QuPath using ONNX and PyTorch

This repository is for experimentation only.

During my PhD, I used **StarDist** extensively for cell detection via the famous `qupath-extension-stardist`. However, GPU support for **StarDist** in QuPath is limited on Windows 11 and Apple Silicon.

The idea behind this experiment was to convert StarDist TensorFlow models to ONNX Runtime, use a DJL **Hybrid Engine** (ONNX Runtime + PyTorch), and make the StarDist extension for QuPath run inference through that engine.

On Windows 11, I managed to get it working on both CPU and GPU, although some changes to the QuPath source code are required.

On Ubuntu Linux, I also managed to run it (but I don’t currently have access to that workstation).

On macOS, I previously tested DJL with the **DANEELpath** U-Nets and it used the M1 GPU. However, I don’t have access to that laptop right now, and I don’t know yet whether the same approach is compatible with **StarDist**.


