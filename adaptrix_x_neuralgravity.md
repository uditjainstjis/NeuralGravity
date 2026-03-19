# Adaptrix × Neural Gravity: Toward Hardware-Aware Modular Intelligence
**Technical Synthesis & Research Proposal**

## 1. Executive Summary
This document synthesizes the architectural breakthroughs of **Project Neural Gravity** (HLRA, EGMP, Thermal-PID) with the modular domain-routing objectives of **Adaptrix**. We propose a unified framework for local LLM deployment where specialized adapters are not only domain-expert but also hardware-resilient.

## 2. Technical Synergies

### A. HLRA: The Substrate for "Unbreakable" Domain Adapters
*   **Adaptrix Problem:** Training domain adapters (Medical, Legal, Code) on sub-3B models typically requires quantized substrates (4-bit/2-bit) for edge deployment, leading to "weight destruction" and reasoning collapse.
*   **Neural Gravity Solution:** **Hybrid Low-Rank Adaptation (HLRA)**. By using a dual-path architecture—**DoRA** for directional updates and **EoRA** (Error-compensated LoRA) initialized via SVD of the exact quantization error matrix $E = W_{orig} - Q(W)$—HLRA recovers the lost high-frequency signals from quantization.
*   **Integration:** Adaptrix domain adapters should be architected as HLRA modules. This ensures that the Medical or Legal knowledge is injected into the *error manifold* of the base model, preventing the "forgetting" hypothesized in Adaptrix's *Reversibility Hypothesis*.

### B. RAD-Routing: Resource-Aware Dynamic Routing
*   **Adaptrix Problem:** Standard intent-based routing ignores the physical state of the device (MacBook/Edge). High-rank adapters may cause thermal throttling, spiking latency above the <200ms target.
*   **Neural Gravity Solution:** **Thermal PID Controller & EGMP**. Neural Gravity's `thermal_pid.py` monitors `kOSThermalNotificationPressureLevel` to dynamically scale the rank $r$ of the gradient manifold.
*   **Integration:** We propose **RAD-Routing**. The Adaptrix router receives a dual-signal: (1) Semantic Intent from the query and (2) Thermal Pressure from the OS.
    *   *Nominal State:* Load full-rank HLRA adapters for maximum accuracy.
    *   *Heavy Throttling:* Dynamically switch to a "Low-Rank Proxy" or prune the adapter heads to maintain real-time TPS (Tokens Per Second).

### C. Fused Modular Kernels (Solving the Stacking Bottleneck)
*   **Adaptrix Problem:** *Hypothesis 1 (Adapter Interference)* suggests stacking multiple adapters. However, Neural Gravity's *Speculative Cascade* experiment proved that Python-to-Metal dispatch overhead (-58% speedup) kills performance for multi-path logic.
*   **Neural Gravity Solution:** Custom Metal Kernels via `mx.fast.metal_kernel`.
*   **Integration:** To achieve the <200ms swap latency, Adaptrix must avoid sequential adapter loading. We propose a **Fused Modular Kernel** that loads multiple adapter weights into a single GPU address space and uses a "Masked-Gating" operation (implemented in Metal) to apply domain deltas in a single fused pass.

## 3. Revised Research Hypotheses (ICLR 2026 Target)

1.  **The Quantization-Modularity Paradox:** Modular domain adapters trained via standard LoRA on 2-bit models will diverge; however, **HLRA-based modularity** will exhibit statistical parity with full-precision adapters.
2.  **Thermal-Latency Equilibrium:** RAD-Routing can maintain a consistent inference latency of <200ms across varying thermal loads (0 to 3) by dynamically modulating adapter rank $r$ without dropping below 85% domain accuracy.
3.  **Manifold Orthogonality:** By using Neural Gravity’s **EGMP** during the training of domain adapters, we can enforce gradient orthogonality between the Medical and Legal subspaces, significantly reducing "adapter interference" during stacking.

## 4. Implementation Path
1.  **Base:** Use Qwen2-1.5B/Phi-2 as the substrate.
2.  **Training:** Employ the `EGMPOptimizer` from `neural_gravity/` to train Adaptrix adapters with 82% less memory.
3.  **Deployment:** Implement the `HybridLinear` layer for all domain-specific heads to ensure reasoning stability on Apple Silicon M3/M4.
