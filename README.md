# Neural Gravity

Neural Gravity is a research repo for edge-LLM experimentation on Apple Silicon using MLX. It currently contains three strands of work:

- `HLRA`: hybrid low-rank adaptation for aggressively quantized models
- `Speculative decoding`: correctness and throughput experiments on Apple M3
- `TTA*`: test-time A* search for reasoning on MATH-500

The repo no longer presents speculative decoding or TTA* as breakthrough wins. The current value is artifact-backed systems evidence: what worked, what failed, and why.

## Current Status

### Speculative decoding

The current speculative decoding implementation is now token-identical to greedy decoding on the benchmark prompts, but it is slower than baseline on this hardware/software stack.

Latest validated result from [`reports/benchmark_uma_cascade.json`](./reports/benchmark_uma_cascade.json):

- Baseline TPS: `43.01`
- Speculative TPS (`K=5`): `20.94`
- Speedup: `-51.32%`
- Correctness: `NO MISMATCH`

We also ran a draft-length sweep in [`reports/speculative_k_sweep.json`](./reports/speculative_k_sweep.json):

- `K=1`: `-40.58%`
- `K=2`: `-43.70%`
- `K=3`: `-45.87%`
- `K=5`: `-51.37%`
- `K=8`: `-57.17%`

All tested `K` values remained token-identical, and throughput degraded monotonically as `K` increased.

### Test-time A* search

Saved artifact: [`final_tta_benchmark_results.json`](./final_tta_benchmark_results.json)

- Greedy: `12/100` on MATH-500
- TTA*: `4/100`
- Greedy wall time: `474.1s`
- TTA* wall time: `920.4s`

Current conclusion: the self-reflection heuristic is not reliable enough on the tested 4-bit 3B model to outperform greedy decoding.

## What This Repo Is Useful For

- Reproducing a correctness-first speculative decoding implementation on Apple MLX
- Understanding why speculative decoding can fail to speed up on consumer Apple hardware
- Inspecting an artifact-backed negative result instead of relying on optimistic claims
- Using HLRA training components as a base for further quantization-resilience experiments

## Main Artifacts

- [`reports/iclr_2026_submission.tex`](./reports/iclr_2026_submission.tex): current paper draft focused on measured results
- [`reports/RESULTS_SUMMARY.md`](./reports/RESULTS_SUMMARY.md): concise summary of findings
- [`reports/REPRODUCIBILITY.md`](./reports/REPRODUCIBILITY.md): commands and artifact map
- [`docs/index.html`](./docs/index.html): static results page suitable for GitHub Pages
- [`benchmark_uma_cascade.py`](./benchmark_uma_cascade.py): speculative decoding benchmark
- [`benchmark_tta.py`](./benchmark_tta.py): MATH-500 TTA* benchmark
- [`neural_gravity/`](./neural_gravity): HLRA and optimizer code

## Reproducing The Main Results

Speculative decoding benchmark:

```bash
./venv/bin/python3 benchmark_uma_cascade.py
```

TTA* benchmark:

```bash
./venv/bin/python3 benchmark_tta.py
```

## GitHub Pages

The repo includes a static results page at [`docs/index.html`](./docs/index.html).

To publish it with GitHub Pages:

1. Open the GitHub repository settings.
2. Go to `Pages`.
3. Set the source to `Deploy from a branch`.
4. Choose branch `main` and folder `/docs`.
5. Save.

After GitHub finishes publishing, the page should be available at:

```text
https://uditjainstjis.github.io/NeuralGravity/
```

For a short operational note, see [`docs/PUBLISHING.md`](./docs/PUBLISHING.md).

## Paper Positioning

The current paper is not a breakthrough paper. It is a practical systems/negative-results paper:

- correctness fixed
- speedup not achieved
- TTA* underperforms greedy
- draft-length ablation shows larger speculative depth hurts on this stack

That makes the repo useful as a baseline and debugging reference for future Apple MLX inference work.

## Acknowledgements

This project is part of the SKIntern Team's research on hardware-constrained large language model optimization.
