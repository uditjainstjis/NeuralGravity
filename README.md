# Neural Gravity

Neural Gravity is an edge-LLM systems research project built and tested on an 8GB Apple MacBook M3 using Apple MLX. The project implements three main components: a Hybrid Low-Rank Adaptation (HLRA) training framework for aggressively quantized models, a correctness-first speculative decoding pipeline using a 0.5B drafter with a 3B target model, and a hardware-aware training daemon with thermal control and persistence mechanisms.

The strongest validated result in this repo is not a speed breakthrough, but a reproducible systems finding: speculative decoding was made token-identical to greedy decoding, yet on this Apple MLX stack it reduced throughput by 51.32% rather than improving it. The repo also includes executable HLRA and thermal-control infrastructure, but the currently saved ablation artifacts do not demonstrate a confirmed HLRA quality win.

Neural Gravity should therefore be understood as an artifact-backed exploration of what does and does not work for local LLM acceleration on consumer Apple Silicon. Its main value is trustworthy implementation detail, negative results, and reproducibility, rather than a verified frontier-performance breakthrough.

## What Neural Gravity Contains

- `HLRA`: a dual-path adapter implementation for quantized-model fine-tuning
- `Speculative decoding`: a correctness-first Apple MLX implementation using `Qwen2.5-0.5B-Instruct` and `Qwen2.5-3B-Instruct-4bit`
- `Thermal-aware training infrastructure`: `powermetrics`-driven PID control, persistence helpers, and a long-running training daemon
- `TTA*`: a test-time A* search experiment for reasoning on MATH-500

## Validated Results

### Speculative decoding

Validated artifact: [`reports/benchmark_uma_cascade.json`](./reports/benchmark_uma_cascade.json)

- Baseline TPS: `43.01`
- Speculative TPS (`K=5`): `20.94`
- Speedup: `-51.32%`
- Correctness: `NO MISMATCH`
- Status: `CORRECT IMPLEMENTATION`

Interpretation:

- speculative decoding correctness was fixed
- throughput did not improve on this hardware/software stack
- the implementation is useful as a correctness baseline, not as a speedup result

Draft-length sweep artifact: [`reports/speculative_k_sweep.json`](./reports/speculative_k_sweep.json)

- `K=1`: `-40.58%`
- `K=2`: `-43.70%`
- `K=3`: `-45.87%`
- `K=5`: `-51.37%`
- `K=8`: `-57.17%`

All tested `K` values remained token-identical, and throughput worsened as `K` increased.

### HLRA

Confirmed in the repo:

- HLRA adapter code exists in [`neural_gravity/hybrid_adapter.py`](./neural_gravity/hybrid_adapter.py)
- a real ablation runner exists in [`train_hlra_ablation.py`](./train_hlra_ablation.py)
- validation/training infrastructure exists in [`validate_hlra.py`](./validate_hlra.py) and [`train_daemon.py`](./train_daemon.py)

Current saved ablation artifact: [`reports/hlra_ablation_results.json`](./reports/hlra_ablation_results.json)

- DoRA loss: `1.7652`
- HLRA loss: `1.7905`
- Relative delta: `-1.43%`

Interpretation:

- the repo contains working HLRA infrastructure and ablation code
- the currently saved artifact does not show a confirmed HLRA win
- HLRA should be presented as an implemented research direction, not a proven quality breakthrough

### Test-time A* search

Saved artifact: [`final_tta_benchmark_results.json`](./final_tta_benchmark_results.json)

- Greedy: `12/100` on MATH-500
- TTA*: `4/100`
- Greedy wall time: `474.1s`
- TTA* wall time: `920.4s`

Interpretation:

- the current self-reflection heuristic does not outperform greedy decoding
- the implementation is slower and less accurate than the baseline

## What This Repo Is Useful For

- reproducing a correctness-first speculative decoding implementation on Apple MLX
- understanding why speculative decoding can fail to speed up on consumer Apple hardware
- studying edge-LLM engineering tradeoffs on Unified Memory systems
- using HLRA, EGMP, and thermal-control code as a base for further experiments
- inspecting an artifact-backed negative result instead of relying on optimistic claims

## Main Files

- [`reports/iclr_2026_submission.tex`](./reports/iclr_2026_submission.tex): current paper draft focused on measured results
- [`reports/RESULTS_SUMMARY.md`](./reports/RESULTS_SUMMARY.md): concise summary of findings
- [`reports/REPRODUCIBILITY.md`](./reports/REPRODUCIBILITY.md): commands and artifact map
- [`docs/index.html`](./docs/index.html): static results page suitable for GitHub Pages
- [`benchmark_uma_cascade.py`](./benchmark_uma_cascade.py): speculative decoding benchmark
- [`benchmark_tta.py`](./benchmark_tta.py): MATH-500 TTA* benchmark
- [`train_hlra_ablation.py`](./train_hlra_ablation.py): HLRA vs DoRA ablation runner
- [`train_daemon.py`](./train_daemon.py): thermal-aware training daemon
- [`neural_gravity/`](./neural_gravity): HLRA, optimizer, thermal, and persistence code

## Reproducing Main Results

Speculative decoding benchmark:

```bash
./venv/bin/python3 benchmark_uma_cascade.py
```

TTA* benchmark:

```bash
./venv/bin/python3 benchmark_tta.py
```

HLRA ablation:

```bash
./venv/bin/python3 train_hlra_ablation.py
```

## Project Positioning

This is not a breakthrough-performance repo. It is a practical edge-LLM systems repo built around careful measurement:

- speculative decoding: correct but slower
- TTA*: slower and less accurate
- HLRA: implemented, but not yet validated here as a clear quality win

That still makes Neural Gravity useful. The repo establishes a concrete baseline for future Apple MLX and Apple Silicon research, and it preserves the implementation details needed to continue from a technically honest starting point.

## GitHub Pages

The repo includes a static page at [`docs/index.html`](./docs/index.html).

To publish it with GitHub Pages:

1. Open the GitHub repository settings.
2. Go to `Pages`.
3. Set the source to `Deploy from a branch`.
4. Choose branch `main` and folder `/docs`.
5. Save.

Expected URL:

```text
https://uditjainstjis.github.io/NeuralGravity/
```

For a short operational note, see [`docs/PUBLISHING.md`](./docs/PUBLISHING.md).

## Acknowledgements

This project is part of the SKIntern Team's research on hardware-constrained large language model optimization.
