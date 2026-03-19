# Technical Correlation: Adaptrix × Neural Gravity

## 1. Executive Summary: The Driver and the Engine
This document maps the **Adaptrix v4 Research Architecture** (the high-level "Logic" and "Strategy") onto the **Neural Gravity Implementation** (the "Physical Substrate" and "Hardware Engine"). 

Neural Gravity has already implemented the critical low-level infrastructure required to test the core hypotheses of the Adaptrix plan.

---

## 2. Research Problem Mapping (Adaptrix v4 Sections)

### 3.1 Capability Injection (Substrate Ready)
*   **Adaptrix Goal:** Inject narrow cognitive abilities via modular updates.
*   **Neural Gravity Solution:** `neural_gravity/hybrid_adapter.py`. 
*   **Status:** **Implemented.** The `HybridLinear` layer uses **HLRA (DoRA + EoRA)** to ensure injected skills remain stable even on 2-bit quantized models. This provides a "hardened" version of the injection method proposed in Adaptrix.

### 3.3 Reversible Learning (Mechanism Ready)
*   **Adaptrix Goal:** Measure acquisition and removal of intelligence.
*   **Neural Gravity Solution:** `neural_gravity/persistence.py` and `validate_hlra.py`.
*   **Status:** **Ready for Testing.** The system can already save, load, and detach adapters. You can immediately run "Before/After/Unloaded" benchmarks to validate the Reversibility Hypothesis.

### 3.6 Temporal Dynamics of Intelligence (Quantified)
*   **Adaptrix Goal:** Measure acquisition speed and loading latency.
*   **Neural Gravity Solution:** `benchmark_tta.py` (Time-To-Adapter) and `benchmark_uma_cascade.py`.
*   **Status:** **Partially Characterized.** Neural Gravity has already benchmarked the latency of swapping modules in Unified Memory. The "Speculative Cascade" reports quantify the exact temporal costs Adaptrix seeks to study.

### 3.7 Cognitive Compute Economics (Core Breakthrough)
*   **Adaptrix Goal:** Measure compute cost, VRAM per skill, and latency overhead.
*   **Neural Gravity Solution:** **EGMP Optimizer** and **Thermal PID Controller**.
*   **Status:** **Advanced.** Neural Gravity goes beyond measurement; it actively *manages* economics. EGMP reduces training VRAM by ~80%, and the Thermal PID dynamically scales the rank $r$ of intelligence to fit within the device's thermal "budget."

### 3.8 Intelligence as Operating System (Kernel Level)
*   **Adaptrix Goal:** View LLMs as cognitive kernels and adapters as packages.
*   **Neural Gravity Solution:** Hardware-aware scheduling.
*   **Status:** **Implemented as "Drivers."** If Adaptrix is the OS, Neural Gravity provides the kernel drivers. `thermal_pid.py` acts as the resource scheduler, and `egmp_optimizer.py` acts as the low-level memory manager for the manifold update.

---

## 3. The Research Gap (Where Adaptrix Adds Value)
While Neural Gravity provides the **Engine**, the following "Logic" problems in the Adaptrix v4 plan are currently **unsolved** and represent the primary research opportunity:

*   **Section 3.2 (Isolation):** Neural Gravity does not yet measure "Domain Confusion" or reasoning chain collapse when the wrong adapter is triggered.
*   **Section 3.4 (Composition):** The behavioral effects of stacking (confusion collapse vs. cross-domain reasoning) have not been studied.
*   **Section 3.5 (Interference Mapping):** Creating the **Skill Interaction Graph** (e.g., does Math help Physics?) is a high-novelty task that remains entirely open.

## 4. Synthesis for ICLR 2026
We propose a unified research path: **"Adaptrix: A Resource-Aware Operating System for Modular Intelligence."**
1.  **Architecture:** Use Neural Gravity’s HLRA/EGMP for the physical layer.
2.  **Logic:** Implement Adaptrix’s Dynamic Routing and Interference Mapping for the cognitive layer.
3.  **Result:** A local LLM that can gain/shed skills on demand while dynamically scaling its reasoning depth based on hardware thermal pressure.
