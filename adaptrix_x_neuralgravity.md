# Technical Correlation: Antigravity × Neural Gravity

## 1. Project Identities: Logic vs. Physics
While both projects utilize LoRA-based adapters for local LLMs, they solve different layers of the modular intelligence problem:

*   **Antigravity (The Logic):** Focuses on **Dynamic Routing** and **Lifecycle Management**. It answers: *"Which adapter should I load for this specific query, and how do I swap it out cleanly?"*
*   **Neural Gravity (The Physics):** Focuses on **Architecture** and **Hardware Execution**. It answers: *"How do I ensure these adapters don't break when quantized to 2-bits, and how do I train/run them within the thermal limits of a MacBook M3?"*

## 2. How Neural Gravity Helps Antigravity

### A. Solving the "Quantization-Modularity Paradox"
Antigravity assumes that adapters can be hot-swapped onto small base models. However, at the 1.5B–3B scale, aggressive quantization (4-bit/2-bit) often destroys the "reasoning manifold."
*   **Neural Gravity's Contribution:** The **Hybrid Low-Rank Adaptation (HLRA)** in `hybrid_adapter.py` uses an Error-compensated path (EoRA) to recover signals lost during quantization. This provides Antigravity with a **stable substrate**, ensuring that a "Medical" or "Legal" adapter actually works on a compressed local model.

### B. Thermal-Aware Routing (RAD-Routing)
Antigravity aims for <200ms swap latency. On mobile devices, thermal throttling can spike this latency, making the system unusable.
*   **Neural Gravity's Contribution:** The **Thermal PID Controller** and **EGMP Optimizer** allow the system to sense hardware pressure. This enables a "Resource-Aware" version of Antigravity routing: if the device is overheating, the router can choose a lower-rank version of an adapter to maintain the 200ms target.

### C. Reducing Training Overhead
Antigravity requires training multiple domain adapters (Medical, Code, Legal). 
*   **Neural Gravity's Contribution:** The **Elastic Gradient Manifold Projection (EGMP)** optimizer reduces memory usage by ~80% during training by projecting gradients into a low-rank manifold. This makes the Antigravity vision of "user-trained local experts" feasible on a single consumer GPU.

## 3. Honest Distinctions
*   **Routing:** Neural Gravity does **not** implement semantic intent classification (the core of Antigravity). It assumes the "what to run" is decided elsewhere.
*   **Modularity:** Antigravity treats adapters as **black boxes** to be routed. Neural Gravity treats them as **mathematical manifolds** to be optimized for the M3/M4 GPU architecture.

## 4. Synthesis for ICLR 2026
We propose using **Neural Gravity's HLRA** as the standard module format for **Antigravity's Routing System**. This combination allows for a modular LLM that is both semantically intelligent (Antigravity) and physically resilient (Neural Gravity).
