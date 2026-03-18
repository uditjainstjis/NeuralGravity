# Neural Gravity: Hybrid Low-Rank Adaptation (HLRA)

A framework for running and fine-tuning 3B-parameter models (e.g., Qwen2.5-3B) within the strict 5.4GB Metal memory limit of an 8GB MacBook, using Apple MLX.

## 🚀 Key Innovation: HLRA
We developed **Hybrid Low-Rank Adaptation (HLRA)**, a dual-path adapter designed to compensate for weight destruction in ultra-low bit (4-bit/2-bit) quantized models.
- **Path 1 (DoRA)**: Standard weight-decomposed directional updates.
- **Path 2 (EoRA)**: Residual path explicitly initialized via SVD of the quantization error matrix.

## 🔬 Experiments
- **Successful Validation**: HLRA demonstrated deeper and more stable loss convergence than standard LoRA/DoRA on 2-bit models.
- **Test-time A* Search (TTA*)**: (WIP) A decoding wrapper recovering reasoning capacity through self-reflection heuristics.
- **Speculative Cascade**: (Pivoted) Explored speculative decoding with custom Metal kernels, currently focusing on pure HLRA logic for the ICLR 2026 submission.

## 🛠️ Performance & Scalability
- **Apple Silicon M3 Optimized**: Leverages Unified Memory Architecture (UMA).
- **Quantization Resilience**: Maintaining reasoning capabilities on 2-bit/4-bit substrates.

## 📂 Project Structure
- `neural_gravity/`: Implementation of HLRA and EGMP Optimizer.
- `reports/iclr_2026_submission.tex`: Academic paper (HLRA Focus).
- `reasoning_search.py`: Test-time A* Search implementation.
- `fast_metal_cascade.py`: Speculative decoding experimentation.

## 📜 Acknowledgements
This project is part of the SKIntern Team's AI research on hardware-constrained large language model optimization.
