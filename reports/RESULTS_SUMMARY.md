# Results Summary

This document summarizes the current measured state of the repo.

## Speculative Decoding

Artifact: [`benchmark_uma_cascade.json`](./benchmark_uma_cascade.json)

Current validated result:

- Baseline TPS: `43.01`
- Speculative TPS (`K=5`): `20.94`
- Speedup: `-51.32%`
- Mismatch status: `NO MISMATCH`
- Implementation status: `CORRECT IMPLEMENTATION`

Interpretation:

- Correctness has been fixed.
- Throughput is still materially worse than greedy decoding.
- The current Apple MLX implementation is useful as a correctness baseline, not a speedup result.

## Draft-Length Sweep

Artifact: [`speculative_k_sweep.json`](./speculative_k_sweep.json)

Measured sweep:

| K | Baseline TPS | Speculative TPS | Speedup | Correct |
|---|---:|---:|---:|---|
| 1 | 42.25 | 25.10 | -40.58% | Yes |
| 2 | 44.33 | 24.96 | -43.70% | Yes |
| 3 | 44.61 | 24.15 | -45.87% | Yes |
| 5 | 44.38 | 21.58 | -51.37% | Yes |
| 8 | 44.68 | 19.14 | -57.17% | Yes |

Interpretation:

- Every tested configuration remains token-identical to greedy decoding.
- Larger draft lengths make throughput worse on this stack.
- In the current implementation, `K=1` is the least bad setting, but still slower than greedy decoding.

## Test-Time A* Search

Artifact: [`../final_tta_benchmark_results.json`](../final_tta_benchmark_results.json)

Saved benchmark result on 100 MATH-500 questions:

- Greedy: `12/100`
- TTA*: `4/100`
- Greedy total tokens: `19,989`
- TTA* total tokens: `4,100`
- Greedy wall time: `474.1s`
- TTA* wall time: `920.4s`

Interpretation:

- The current TTA* heuristic does not outperform greedy decoding.
- The implementation is slower and less accurate than the greedy baseline.

## Main Takeaway

The repo now contains useful, reproducible negative results:

- speculative decoding: correct but slower
- TTA*: slower and less accurate

That is still valuable research output because it establishes a trustworthy baseline for future MLX/Apple Silicon work.
