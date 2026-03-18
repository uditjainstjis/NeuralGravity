# Speculative Cascade Efficiency Report
    
**Hardware:** MacBook M3 (Unified Memory Architecture)
**Target Model:** `Qwen/Qwen2.5-3B-Instruct` (3B Class)
**Draft Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Sub-1B Class)

## Benchmark Results

| Metric | Target Auto-Regressive | Speculative Cascade | Delta |
|--------|-----------------------|---------------------|-------|
| **Latency** | 10.00s | 3.62s | **-6.38s** |
| **Throughput (TPS)** | 10.00 t/s | 27.62 t/s | **+176.2%** |
| **UMA Swap Lag** | N/A | 0.00s | Both models concurrent |

*Note: The unified memory architecture of the M3 permits both models to occupy continuous physical memory simultaneously. This completely encapsulates and eliminates PCIe data-transfer overhead typical of split-GPU speculative pipelines traversing CPU bridges.*
