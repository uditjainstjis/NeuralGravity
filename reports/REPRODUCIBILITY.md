# Reproducibility

This file maps the main claims in the repo to commands and saved artifacts.

## Environment

The experiments were run from the repo-local virtual environment:

```bash
./venv/bin/python3
```

## Speculative Decoding Benchmark

Command:

```bash
./venv/bin/python3 benchmark_uma_cascade.py
```

Outputs:

- [`benchmark_uma_cascade.json`](./benchmark_uma_cascade.json)
- [`current_status.txt`](./current_status.txt)

What it does:

- runs greedy baseline decoding
- runs custom speculative decoding
- compares outputs token-by-token
- reports mismatch status
- computes TPS and speedup

## Draft-Length Sweep

This was run as an ad hoc research ablation using the same `speculative_decode(...)` and `baseline_decode(...)` functions from [`../benchmark_uma_cascade.py`](../benchmark_uma_cascade.py).

Saved output:

- [`speculative_k_sweep.json`](./speculative_k_sweep.json)

Sweep values:

- `K = 1, 2, 3, 5, 8`

## TTA* Benchmark

Command:

```bash
./venv/bin/python3 benchmark_tta.py
```

Output:

- [`../final_tta_benchmark_results.json`](../final_tta_benchmark_results.json)

What it does:

- evaluates greedy decoding on a 100-question MATH-500 subset
- evaluates TTA* search on the same subset
- saves exact question-level outputs and accuracy totals

## Paper Source

Current paper draft:

- [`iclr_2026_submission.tex`](./iclr_2026_submission.tex)

The paper is intended to reflect only artifact-backed numbers. If any artifact changes, the paper should be updated to match it exactly.
