import gc
import json
import logging
import os
import statistics
import time

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - BenchmarkUMA - %(message)s")

TARGET_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DRAFT_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 64
MAIN_K = 5
K_SWEEP = [1, 2, 3, 5, 8]
BLOCK_RELOAD_INTERVAL = 10
K_SWEEP_PROMPTS = 20

# 70 bounded prompts across 10 categories. The benchmark is about inference
# behavior across prompt styles rather than task accuracy in one domain.
PROMPT_SPECS = [
    {"category": "science", "prompt": "In 5 short sentences, explain general relativity to a senior undergraduate physics student."},
    {"category": "science", "prompt": "In 5 short sentences, summarize how TCP congestion control works in practice."},
    {"category": "science", "prompt": "In 6 short sentences, describe the process of cellular respiration step by step."},
    {"category": "science", "prompt": "In 4 short sentences, explain how cryptographic hash functions are used in blockchains."},
    {"category": "science", "prompt": "In 4 short sentences, describe the greenhouse effect without equations."},
    {"category": "science", "prompt": "In 5 short sentences, explain Bayes' theorem with a medical-testing example."},
    {"category": "science", "prompt": "In 5 short sentences, explain why the harmonic series diverges."},

    {"category": "programming", "prompt": "Write only Python code for breadth-first search on a graph."},
    {"category": "programming", "prompt": "Write only Python code for merging overlapping intervals."},
    {"category": "programming", "prompt": "Write only Python code for topological sort."},
    {"category": "programming", "prompt": "Write only Python code for detecting a cycle in a linked list."},
    {"category": "programming", "prompt": "Write only Python code for an LRU cache using standard library features."},
    {"category": "programming", "prompt": "Write only shell script to count file extensions recursively in a directory."},
    {"category": "programming", "prompt": "Write only pseudocode for Dijkstra's algorithm."},

    {"category": "systems_ml", "prompt": "In 5 short sentences, discuss the tradeoff between latency and throughput in serving LLMs."},
    {"category": "systems_ml", "prompt": "In 5 short sentences, explain the difference between RAM bandwidth limits and compute limits in GPU inference."},
    {"category": "systems_ml", "prompt": "In 5 short sentences, explain why overfitting happens in machine learning."},
    {"category": "systems_ml", "prompt": "In 5 short sentences, describe how gradient descent behaves near a saddle point."},
    {"category": "systems_ml", "prompt": "In 6 short sentences, compare supervised, unsupervised, and reinforcement learning."},
    {"category": "systems_ml", "prompt": "In 4 short sentences, explain why recursion works in divide-and-conquer algorithms."},
    {"category": "systems_ml", "prompt": "In 6 short sentences, describe how compilers move from source code to machine code."},

    {"category": "math", "prompt": "Solve step by step: if 3x + 7 = 22, what is x?"},
    {"category": "math", "prompt": "Solve step by step: a rectangle has perimeter 30 and length 8, find width."},
    {"category": "math", "prompt": "Find the derivative of x^3 * sin(x) and explain each rule used briefly."},
    {"category": "math", "prompt": "Evaluate the integral of 2x from 0 to 5 and explain the steps briefly."},
    {"category": "math", "prompt": "A fair coin is flipped 10 times. What is the probability of exactly 6 heads? Show the key steps briefly."},
    {"category": "math", "prompt": "In 4 short sentences, explain why floating-point equality is dangerous for money calculations."},
    {"category": "math", "prompt": "In 4 short sentences, describe a regex strategy for a simple email-like validator."},

    {"category": "data_sql", "prompt": "Write only SQL to find the second-highest salary from an employee table."},
    {"category": "data_sql", "prompt": "Write only SQL to compute a 7-day rolling average from a daily_sales table."},
    {"category": "data_sql", "prompt": "Write only SQL to find duplicate email addresses in a users table."},
    {"category": "data_sql", "prompt": "Write only SQL to rank products by revenue within each category."},
    {"category": "data_sql", "prompt": "Write only SQL to return customers with no orders in the last 90 days."},
    {"category": "data_sql", "prompt": "Write only SQL to compute month-over-month growth from a revenue table."},
    {"category": "data_sql", "prompt": "Write only SQL to join orders, customers, and payments into one report query."},

    {"category": "business_econ", "prompt": "In 5 short sentences, explain the difference between inflation and stagflation."},
    {"category": "business_econ", "prompt": "In 5 short sentences, explain why price elasticity matters for product strategy."},
    {"category": "business_econ", "prompt": "In 5 short sentences, explain how working capital affects a small business."},
    {"category": "business_econ", "prompt": "In 5 short sentences, compare gross margin and operating margin."},
    {"category": "business_econ", "prompt": "In 5 short sentences, explain the main risks in a two-sided marketplace business."},
    {"category": "business_econ", "prompt": "In 5 short sentences, describe when a startup should prioritize growth over profitability."},
    {"category": "business_econ", "prompt": "In 5 short sentences, explain the economic tradeoff between automation and labor costs."},

    {"category": "writing_marketing", "prompt": "Write a short product launch email in under 120 words for a note-taking app."},
    {"category": "writing_marketing", "prompt": "Write a press release in under 120 words for a local robotics competition."},
    {"category": "writing_marketing", "prompt": "Write one persuasive paragraph under 120 words arguing for public libraries."},
    {"category": "writing_marketing", "prompt": "Write a concise cold email in under 120 words for a B2B analytics tool."},
    {"category": "writing_marketing", "prompt": "Write a 6-bullet outline for a talk on edge AI deployment."},
    {"category": "writing_marketing", "prompt": "Write a short FAQ answer in under 100 words about resetting a password."},
    {"category": "writing_marketing", "prompt": "Write a 5-bullet onboarding checklist for a new software intern."},

    {"category": "creative", "prompt": "Write a science fiction scene in under 120 words set inside a Dyson sphere."},
    {"category": "creative", "prompt": "Write a short dialogue in under 120 words between a skeptical engineer and an optimistic founder."},
    {"category": "creative", "prompt": "Write a story opening in under 120 words about a city where gravity changes every hour."},
    {"category": "creative", "prompt": "In under 120 words, describe a calm morning at a railway station in literary prose."},
    {"category": "creative", "prompt": "Write a monologue in under 120 words from the perspective of a retired astronaut."},
    {"category": "creative", "prompt": "Write a fantasy tavern scene in under 120 words with a hidden political tension."},
    {"category": "creative", "prompt": "Write a noir-style city description in under 120 words during a power outage."},

    {"category": "education", "prompt": "Write a Git tutorial introduction in under 120 words for first-year students."},
    {"category": "education", "prompt": "In 5 short sentences, explain photosynthesis to a 12-year-old."},
    {"category": "education", "prompt": "In 5 short sentences, explain what an API is to a beginner programmer."},
    {"category": "education", "prompt": "In 5 short sentences, explain the concept of opportunity cost to a high-school student."},
    {"category": "education", "prompt": "In 5 short sentences, explain how the Internet finds a website from a domain name."},
    {"category": "education", "prompt": "In 5 short sentences, explain what a database index does."},
    {"category": "education", "prompt": "In 5 short sentences, explain why unit tests are useful in software projects."},

    {"category": "health_policy", "prompt": "In 5 short sentences, explain the difference between a virus and a bacterium."},
    {"category": "health_policy", "prompt": "In 5 short sentences, explain herd immunity in simple terms."},
    {"category": "health_policy", "prompt": "In 5 short sentences, explain why antibiotic resistance is a public-health problem."},
    {"category": "health_policy", "prompt": "In 5 short sentences, explain the main tradeoffs in public transit policy."},
    {"category": "health_policy", "prompt": "In 5 short sentences, explain what due process means in a civic context."},
    {"category": "health_policy", "prompt": "In 5 short sentences, explain why data privacy matters in consumer apps."},
    {"category": "health_policy", "prompt": "In 5 short sentences, explain the difference between a regulation and a law."},
]


def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def safe_speedup(new_value, baseline_value):
    if baseline_value == 0:
        return 0.0
    return ((new_value / baseline_value) - 1.0) * 100.0


def summarize_by_category(per_prompt_rows):
    grouped = {}
    for row in per_prompt_rows:
        category = row["category"]
        grouped.setdefault(category, {"baseline": [], "correctness_first": [], "native": []})
        grouped[category]["baseline"].append(row["baseline_tps"])
        grouped[category]["correctness_first"].append(row["correctness_first_tps"])
        grouped[category]["native"].append(row["native_speculative_tps"])

    summary = {}
    for category, values in grouped.items():
        base_mean, base_std = mean_std(values["baseline"])
        cf_mean, cf_std = mean_std(values["correctness_first"])
        native_mean, native_std = mean_std(values["native"])
        summary[category] = {
            "num_prompts": len(values["baseline"]),
            "baseline": {"mean_tps": base_mean, "std_tps": base_std},
            "correctness_first_speculative": {
                "mean_tps": cf_mean,
                "std_tps": cf_std,
                "speedup_percentage": safe_speedup(cf_mean, base_mean),
            },
            "native_mlx_speculative": {
                "mean_tps": native_mean,
                "std_tps": native_std,
                "speedup_percentage": safe_speedup(native_mean, base_mean),
            },
        }
    return summary


def load_models_sequentially(target_name, draft_name):
    logging.info(f"Loading target model: {target_name}")
    target_model, target_tokenizer = load(target_name)
    mx.eval(target_model.parameters())

    logging.info("Clearing caches before loading draft model...")
    clear_runtime_caches()

    logging.info(f"Loading draft model: {draft_name}")
    draft_model, _ = load(draft_name)
    mx.eval(draft_model.parameters())
    return target_model, draft_model, target_tokenizer


def clear_runtime_caches():
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    else:
        mx.metal.clear_cache()


def unload_models(*models):
    for model in models:
        del model
    clear_runtime_caches()


def baseline_decode(prompt_tokens, target_model, max_tokens=MAX_TOKENS):
    target_cache = make_prompt_cache(target_model)
    y = mx.array(prompt_tokens, dtype=mx.uint32)
    accepted_tokens = []
    per_token_times = []

    for _ in range(max_tokens):
        token_start = time.perf_counter()
        logits = target_model(y[None], cache=target_cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tok)
        per_token_times.append(time.perf_counter() - token_start)
        accepted_tokens.append(next_tok.item())
        y = next_tok

    return {
        "tokens": accepted_tokens,
        "per_token_times_sec": per_token_times,
    }


def speculative_decode_correctness_first(prompt_tokens, target, draft, max_tokens=MAX_TOKENS, k=MAIN_K):
    """
    Correctness-first implementation.
    - target prefix prefill once
    - draft cache rebuilt each round from the accepted prefix
    - target verifies draft tokens sequentially to preserve token identity
    """
    t_cache = make_prompt_cache(target)
    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    prefix_tokens = list(prompt_tokens)
    accepted_tokens = []
    ntoks = 0

    profiling = {
        "target_prefill_sec": 0.0,
        "draft_generation_sec": 0.0,
        "verification_sec": 0.0,
        "correction_sec": 0.0,
        "bookkeeping_sec": 0.0,
        "rounds": 0,
        "draft_tokens_proposed": 0,
        "draft_tokens_accepted": 0,
        "corrections": 0,
    }

    prefill_start = time.perf_counter()
    t_logits = target(prompt[None], cache=t_cache)
    t_pred = mx.argmax(t_logits[:, -1, :], axis=-1)
    mx.eval(t_pred)
    profiling["target_prefill_sec"] = time.perf_counter() - prefill_start

    while ntoks < max_tokens:
        profiling["rounds"] += 1
        num_draft = min(k, max_tokens - ntoks)

        draft_start = time.perf_counter()
        d_cache = make_prompt_cache(draft)
        drafts = []
        curr = mx.array(prefix_tokens, dtype=mx.uint32)
        for _ in range(num_draft):
            d_logits = draft(curr[None], cache=d_cache)
            curr = mx.argmax(d_logits[:, -1, :], axis=-1)
            mx.eval(curr)
            drafts.append(curr.item())
        profiling["draft_generation_sec"] += time.perf_counter() - draft_start
        profiling["draft_tokens_proposed"] += len(drafts)

        verify_start = time.perf_counter()
        match_len = 0
        for i in range(num_draft):
            if t_pred.item() != drafts[i]:
                break

            accepted_tokens.append(drafts[i])
            ntoks += 1
            match_len += 1
            profiling["draft_tokens_accepted"] += 1

            if ntoks >= max_tokens:
                break

            t_logits = target(mx.array([drafts[i]], dtype=mx.uint32)[None], cache=t_cache)
            t_pred = mx.argmax(t_logits[:, -1, :], axis=-1)
            mx.eval(t_pred)
        profiling["verification_sec"] += time.perf_counter() - verify_start

        if ntoks >= max_tokens:
            break

        correction_start = time.perf_counter()
        correction = t_pred.item()
        accepted_tokens.append(correction)
        ntoks += 1
        profiling["corrections"] += 1

        if ntoks < max_tokens:
            t_logits = target(mx.array([correction], dtype=mx.uint32)[None], cache=t_cache)
            t_pred = mx.argmax(t_logits[:, -1, :], axis=-1)
            mx.eval(t_pred)
        profiling["correction_sec"] += time.perf_counter() - correction_start

        bookkeeping_start = time.perf_counter()
        prefix_tokens.extend(drafts[:match_len])
        prefix_tokens.append(correction)
        profiling["bookkeeping_sec"] += time.perf_counter() - bookkeeping_start

    prof_total = (
        profiling["target_prefill_sec"]
        + profiling["draft_generation_sec"]
        + profiling["verification_sec"]
        + profiling["correction_sec"]
        + profiling["bookkeeping_sec"]
    )
    profiling["measured_component_total_sec"] = prof_total
    profiling["acceptance_rate"] = (
        profiling["draft_tokens_accepted"] / profiling["draft_tokens_proposed"]
        if profiling["draft_tokens_proposed"]
        else 0.0
    )

    return {
        "tokens": accepted_tokens,
        "profiling": profiling,
    }


def native_speculative_generate(prompt_text, tokenizer, target_model, draft_model, max_tokens=MAX_TOKENS):
    start = time.perf_counter()
    response = generate(
        target_model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        verbose=False,
        draft_model=draft_model,
    )
    elapsed = time.perf_counter() - start

    full_tokens = tokenizer.encode(response)
    prompt_tokens = tokenizer.encode(prompt_text)
    generated_tokens = full_tokens[len(prompt_tokens):]
    return {
        "tokens": generated_tokens[:max_tokens],
        "elapsed_sec": elapsed,
    }


def find_first_mismatch(reference_tokens, candidate_tokens):
    limit = min(len(reference_tokens), len(candidate_tokens))
    for idx in range(limit):
        if reference_tokens[idx] != candidate_tokens[idx]:
            return idx, reference_tokens[idx], candidate_tokens[idx]
    if len(reference_tokens) != len(candidate_tokens):
        idx = limit
        ref_tok = reference_tokens[idx] if idx < len(reference_tokens) else None
        cand_tok = candidate_tokens[idx] if idx < len(candidate_tokens) else None
        return idx, ref_tok, cand_tok
    return None


def run_k_sweep(tokenizer, target_model, draft_model, prompt_specs, max_tokens):
    sweep_results = []
    for k in K_SWEEP:
        logging.info("K-sweep: starting k=%d", k)
        base_tps_values = []
        spec_tps_values = []
        mismatch_found = False
        for prompt_spec in prompt_specs:
            prompt = prompt_spec["prompt"]
            prompt_tokens = tokenizer.encode(prompt)
            base = baseline_decode(prompt_tokens, target_model, max_tokens=max_tokens)
            spec_start = time.perf_counter()
            spec = speculative_decode_correctness_first(
                prompt_tokens,
                target_model,
                draft_model,
                max_tokens=max_tokens,
                k=k,
            )
            mx.eval(target_model.parameters(), draft_model.parameters())
            spec_elapsed = time.perf_counter() - spec_start

            base_elapsed = sum(base["per_token_times_sec"])
            base_tps_values.append(max_tokens / base_elapsed)
            spec_tps_values.append(max_tokens / spec_elapsed)
            mismatch_found = mismatch_found or (find_first_mismatch(base["tokens"], spec["tokens"]) is not None)

        avg_base, _ = mean_std(base_tps_values)
        avg_spec, _ = mean_std(spec_tps_values)
        sweep_results.append(
            {
                "k": k,
                "avg_baseline_tps": avg_base,
                "avg_speculative_tps": avg_spec,
                "speedup_percentage": safe_speedup(avg_spec, avg_base),
                "correct": not mismatch_found,
            }
        )
    return sweep_results


def summarize_profile(profile_samples):
    keys = [
        "target_prefill_sec",
        "draft_generation_sec",
        "verification_sec",
        "correction_sec",
        "bookkeeping_sec",
        "measured_component_total_sec",
        "acceptance_rate",
        "rounds",
        "draft_tokens_proposed",
        "draft_tokens_accepted",
        "corrections",
    ]
    summary = {}
    for key in keys:
        values = [sample[key] for sample in profile_samples]
        avg, std = mean_std(values)
        summary[key] = {"mean": avg, "std": std}

    total = summary["measured_component_total_sec"]["mean"] or 1e-9
    summary["share_of_time_percent"] = {
        "target_prefill": 100.0 * summary["target_prefill_sec"]["mean"] / total,
        "draft_generation": 100.0 * summary["draft_generation_sec"]["mean"] / total,
        "verification": 100.0 * summary["verification_sec"]["mean"] / total,
        "correction": 100.0 * summary["correction_sec"]["mean"] / total,
        "bookkeeping": 100.0 * summary["bookkeeping_sec"]["mean"] / total,
    }
    return summary


def ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def run_benchmark():
    ensure_reports_dir()
    logging.info("=== Speculative Decoding Benchmark: Apple MLX / M3 8GB ===")
    logging.info("Prompt count: %d", len(PROMPT_SPECS))
    target_model, draft_model, tokenizer = load_models_sequentially(TARGET_NAME, DRAFT_NAME)

    warmup_tokens = tokenizer.encode("Warmup prompt for speculative decoding.")
    speculative_decode_correctness_first(warmup_tokens, target_model, draft_model, max_tokens=10, k=MAIN_K)
    baseline_decode(warmup_tokens, target_model, max_tokens=10)
    native_speculative_generate("Warmup prompt for speculative decoding.", tokenizer, target_model, draft_model, max_tokens=10)

    baseline_tps_values = []
    correctness_first_tps_values = []
    native_spec_tps_values = []
    correctness_profiles = []
    per_prompt = []
    correctness_mismatch_found = False
    native_mismatch_found = False
    failed_prompts = []

    for idx, prompt_spec in enumerate(PROMPT_SPECS, start=1):
        if idx > 1 and (idx - 1) % BLOCK_RELOAD_INTERVAL == 0:
            logging.info("Reloading models after %d prompts to reduce long-run Metal instability.", idx - 1)
            unload_models(target_model, draft_model)
            target_model, draft_model, tokenizer = load_models_sequentially(TARGET_NAME, DRAFT_NAME)

        logging.info("Running prompt %d/%d", idx, len(PROMPT_SPECS))
        prompt = prompt_spec["prompt"]
        category = prompt_spec["category"]
        prompt_tokens = tokenizer.encode(prompt)

        try:
            clear_runtime_caches()
            base_start = time.perf_counter()
            baseline = baseline_decode(prompt_tokens, target_model, max_tokens=MAX_TOKENS)
            mx.eval(target_model.parameters())
            base_elapsed = time.perf_counter() - base_start
            baseline_tps = MAX_TOKENS / base_elapsed
            baseline_tps_values.append(baseline_tps)

            clear_runtime_caches()
            cf_start = time.perf_counter()
            correctness_first = speculative_decode_correctness_first(
                prompt_tokens,
                target_model,
                draft_model,
                max_tokens=MAX_TOKENS,
                k=MAIN_K,
            )
            mx.eval(target_model.parameters(), draft_model.parameters())
            cf_elapsed = time.perf_counter() - cf_start
            correctness_first_tps = MAX_TOKENS / cf_elapsed
            correctness_first_tps_values.append(correctness_first_tps)
            correctness_profiles.append(correctness_first["profiling"])

            cf_mismatch = find_first_mismatch(baseline["tokens"], correctness_first["tokens"])
            correctness_mismatch_found = correctness_mismatch_found or (cf_mismatch is not None)

            clear_runtime_caches()
            native_spec = native_speculative_generate(
                prompt,
                tokenizer,
                target_model,
                draft_model,
                max_tokens=MAX_TOKENS,
            )
            mx.eval(target_model.parameters(), draft_model.parameters())
            native_tps = MAX_TOKENS / native_spec["elapsed_sec"]
            native_spec_tps_values.append(native_tps)

            native_mismatch = find_first_mismatch(baseline["tokens"], native_spec["tokens"])
            native_mismatch_found = native_mismatch_found or (native_mismatch is not None)

            per_prompt.append(
                {
                    "prompt_index": idx,
                    "category": category,
                    "prompt": prompt,
                    "baseline_tps": baseline_tps,
                    "correctness_first_tps": correctness_first_tps,
                    "native_speculative_tps": native_tps,
                    "correctness_first_speedup_pct": safe_speedup(correctness_first_tps, baseline_tps),
                    "native_speculative_speedup_pct": safe_speedup(native_tps, baseline_tps),
                    "correctness_first_mismatch": None if cf_mismatch is None else {
                        "token_index": cf_mismatch[0],
                        "baseline_token": cf_mismatch[1],
                        "candidate_token": cf_mismatch[2],
                    },
                    "native_speculative_mismatch": None if native_mismatch is None else {
                        "token_index": native_mismatch[0],
                        "baseline_token": native_mismatch[1],
                        "candidate_token": native_mismatch[2],
                    },
                    "correctness_first_profile": correctness_first["profiling"],
                }
            )
        except Exception as exc:
            logging.exception("Prompt %d failed and will be skipped: %s", idx, exc)
            failed_prompts.append({"prompt_index": idx, "category": category, "prompt": prompt, "error": str(exc)})
            unload_models(target_model, draft_model)
            target_model, draft_model, tokenizer = load_models_sequentially(TARGET_NAME, DRAFT_NAME)
            continue

        with open("reports/current_status.txt", "w") as handle:
            handle.write(f"Benchmark progress: {idx}/{len(PROMPT_SPECS)}\n")
            handle.write(f"Latest category: {category}\n")
            handle.write(f"Latest baseline TPS: {baseline_tps:.2f}\n")
            handle.write(f"Latest correctness-first TPS: {correctness_first_tps:.2f}\n")
            handle.write(f"Latest native speculative TPS: {native_tps:.2f}\n")

        logging.info(
            "Prompt %d | baseline %.2f TPS | correctness-first %.2f TPS | native speculative %.2f TPS",
            idx,
            baseline_tps,
            correctness_first_tps,
            native_tps,
        )

    baseline_mean, baseline_std = mean_std(baseline_tps_values)
    cf_mean, cf_std = mean_std(correctness_first_tps_values)
    native_mean, native_std = mean_std(native_spec_tps_values)
    profile_summary = summarize_profile(correctness_profiles)
    clear_runtime_caches()
    k_sweep = run_k_sweep(tokenizer, target_model, draft_model, PROMPT_SPECS[:K_SWEEP_PROMPTS], MAX_TOKENS)
    category_summary = summarize_by_category(per_prompt)

    results = {
        "hardware": "Apple M3 MacBook 8GB",
        "framework": "Apple MLX / MLX-LM",
        "target_model": TARGET_NAME,
        "draft_model": DRAFT_NAME,
        "num_prompts": len(PROMPT_SPECS),
        "num_successful_prompts": len(per_prompt),
        "num_failed_prompts": len(failed_prompts),
        "max_tokens": MAX_TOKENS,
        "main_k": MAIN_K,
        "block_reload_interval": BLOCK_RELOAD_INTERVAL,
        "categories": sorted({spec["category"] for spec in PROMPT_SPECS}),
        "baseline": {
            "mean_tps": baseline_mean,
            "std_tps": baseline_std,
        },
        "correctness_first_speculative": {
            "mean_tps": cf_mean,
            "std_tps": cf_std,
            "speedup_percentage": safe_speedup(cf_mean, baseline_mean),
            "mismatch_found": correctness_mismatch_found,
            "status": "CORRECT IMPLEMENTATION" if not correctness_mismatch_found else "MISMATCH FOUND",
            "profile_summary": profile_summary,
        },
        "native_mlx_speculative": {
            "mean_tps": native_mean,
            "std_tps": native_std,
            "speedup_percentage": safe_speedup(native_mean, baseline_mean),
            "mismatch_found": native_mismatch_found,
            "status": "MATCHED BASELINE" if not native_mismatch_found else "DIVERGED FROM BASELINE",
        },
        "k_sweep": k_sweep,
        "k_sweep_prompt_count": K_SWEEP_PROMPTS,
        "category_summary": category_summary,
        "failed_prompts": failed_prompts,
        "per_prompt": per_prompt,
    }

    with open("reports/benchmark_uma_cascade.json", "w") as handle:
        json.dump(results, handle, indent=2)

    with open("reports/speculative_k_sweep.json", "w") as handle:
        json.dump(k_sweep, handle, indent=2)

    logging.info("=== FINAL RESULTS ===")
    logging.info("Baseline: %.2f +/- %.2f TPS", baseline_mean, baseline_std)
    logging.info(
        "Correctness-first speculative: %.2f +/- %.2f TPS | speedup %.2f%% | mismatch=%s",
        cf_mean,
        cf_std,
        safe_speedup(cf_mean, baseline_mean),
        correctness_mismatch_found,
    )
    logging.info(
        "Native MLX speculative: %.2f +/- %.2f TPS | speedup %.2f%% | mismatch=%s",
        native_mean,
        native_std,
        safe_speedup(native_mean, baseline_mean),
        native_mismatch_found,
    )


if __name__ == "__main__":
    run_benchmark()
