# Announcement Draft

## Short Version

We wrapped up a full correctness-first evaluation of speculative decoding and test-time A* search on Apple MLX / Apple M3.

Main outcome:

- speculative decoding is now token-identical to greedy decoding
- but it is slower than baseline on this stack
- TTA* also underperformed greedy decoding on the saved MATH-500 benchmark

This is not a breakthrough result. It is a reproducible negative result with useful systems lessons for anyone trying to run edge LLM inference on Apple Silicon.

Repo:

```text
https://github.com/uditjainstjis/NeuralGravity
```

Results page:

```text
https://uditjainstjis.github.io/NeuralGravity/
```

## Longer Version

We cleaned up the Neural Gravity repo into a research artifact focused on measured edge-LLM results on Apple Silicon.

What we validated:

- Correct speculative decoding on Apple MLX with no token mismatches on the benchmark prompts
- Real throughput numbers from executed runs, not hardcoded claims
- A draft-length sweep showing that larger speculative depth made performance worse in the current implementation
- A saved MATH-500 TTA* benchmark showing the current self-reflection heuristic underperforms greedy decoding

Current headline numbers:

- Speculative decoding: `43.01 TPS` greedy baseline vs `20.94 TPS` speculative, `-51.32%` speedup, `NO MISMATCH`
- TTA*: `12/100` greedy vs `4/100` TTA* on the saved benchmark

Why this is useful:

- it gives a correctness-first reference implementation
- it documents where Apple MLX / Metal behavior breaks the usual speculative-decoding assumptions
- it provides artifact-backed negative results instead of inflated claims

The repo now includes:

- a cleaned-up paper draft
- reproducibility docs
- saved benchmark artifacts
- a GitHub Pages results page

## Suggested Post

We turned Neural Gravity into a reproducible edge-LLM research artifact on Apple Silicon.

Key result: speculative decoding is now correct on our Apple MLX benchmark, but still slower than greedy decoding on this stack. TTA* also underperformed greedy on the saved MATH-500 run.

So this is not a breakthrough paper. It is a useful negative-result and systems-debugging repo:

- token-identical speculative decoding
- real TPS artifacts
- `K`-sweep ablation
- TTA* benchmark artifacts
- paper + docs + results page

Repo: https://github.com/uditjainstjis/NeuralGravity

If you are trying speculative decoding or search-based inference on Apple Silicon, the failure modes here should save you time.
