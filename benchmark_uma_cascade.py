import time
import json
import logging
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - BenchmarkUMA - %(message)s')

def load_models_sequentially(target_name, draft_name):
    logging.info(f"Loading Target Model: {target_name}")
    target_model, target_tokenizer = load(target_name)
    mx.eval(target_model.parameters())

    logging.info("Forcing Garbarge Collection and Metal Cache Clear...")
    import gc; gc.collect(); mx.metal.clear_cache()

    logging.info(f"Loading Drafter Model: {draft_name}")
    draft_model, _ = load(draft_name)
    mx.eval(draft_model.parameters())

    return target_model, draft_model, target_tokenizer

def speculative_decode(prompt_tokens, target, draft, max_tokens=64, k=5):
    """
    Sequential-verification speculative decode.
    - Draft generates K tokens in bulk (fast, small model)
    - Target verifies each draft token ONE AT A TIME (eliminates numerical drift)
    - On mismatch: take target's prediction as correction, skip remaining drafts
    """
    t_cache = make_prompt_cache(target)

    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    prefix_tokens = list(prompt_tokens)
    accepted_tokens = []
    ntoks = 0

    # Prefill: process prompt through target
    t_logits = target(prompt[None], cache=t_cache)
    t_pred = mx.argmax(t_logits[:, -1, :], axis=-1)
    mx.eval(t_pred)

    while ntoks < max_tokens:
        num_draft = min(k, max_tokens - ntoks)

        # DRAFT: generate K tokens from the draft model (fresh cache each time)
        d_cache = make_prompt_cache(draft)
        drafts = []
        curr = mx.array(prefix_tokens, dtype=mx.uint32)
        for _ in range(num_draft):
            d_logits = draft(curr[None], cache=d_cache)
            curr = mx.argmax(d_logits[:, -1, :], axis=-1)
            mx.eval(curr)
            drafts.append(curr.item())

        # VERIFY: check each draft token against target's prediction SEQUENTIALLY
        match_len = 0
        for i in range(num_draft):
            if t_pred.item() == drafts[i]:
                # Draft matches target prediction -> accept
                accepted_tokens.append(drafts[i])
                ntoks += 1
                match_len += 1

                if ntoks >= max_tokens:
                    break

                # Feed this token to target to get next prediction
                t_logits = target(mx.array([drafts[i]], dtype=mx.uint32)[None], cache=t_cache)
                t_pred = mx.argmax(t_logits[:, -1, :], axis=-1)
                mx.eval(t_pred)
            else:
                break

        if ntoks >= max_tokens:
            break

        # Accept correction (target's prediction)
        correction = t_pred.item()
        accepted_tokens.append(correction)
        ntoks += 1

        if ntoks >= max_tokens:
            break

        # Feed correction to target
        t_logits = target(mx.array([correction], dtype=mx.uint32)[None], cache=t_cache)
        t_pred = mx.argmax(t_logits[:, -1, :], axis=-1)
        mx.eval(t_pred)

        # Update prefix for next draft round
        prefix_tokens.extend(drafts[:match_len])
        prefix_tokens.append(correction)

    return accepted_tokens

def baseline_decode(prompt_tokens, target_model, max_tokens=128):
    target_cache = make_prompt_cache(target_model)
    y = mx.array(prompt_tokens, dtype=mx.uint32)
    
    accepted_tokens = []
    for _ in range(max_tokens):
        logits = target_model(y[None], cache=target_cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        accepted_tokens.append(next_tok.item())
        y = next_tok
        mx.eval(y)
        
    return accepted_tokens

def find_first_mismatch(baseline_tokens, speculative_tokens):
    limit = min(len(baseline_tokens), len(speculative_tokens))
    for idx in range(limit):
        if baseline_tokens[idx] != speculative_tokens[idx]:
            return idx, baseline_tokens[idx], speculative_tokens[idx]
    if len(baseline_tokens) != len(speculative_tokens):
        idx = limit
        base_tok = baseline_tokens[idx] if idx < len(baseline_tokens) else None
        spec_tok = speculative_tokens[idx] if idx < len(speculative_tokens) else None
        return idx, base_tok, spec_tok
    return None

def validate_speculative_correctness(tokenizer, target_model, draft_model, prompts, max_tokens, k):
    logging.info("Validating speculative decoding against greedy baseline...")
    mismatch_found = False
    for prompt_idx, prompt in enumerate(prompts, start=1):
        tokens = tokenizer.encode(prompt)
        baseline_tokens = baseline_decode(tokens, target_model, max_tokens=max_tokens)
        speculative_tokens = speculative_decode(tokens, target_model, draft_model, max_tokens=max_tokens, k=k)
        mismatch = find_first_mismatch(baseline_tokens, speculative_tokens)
        if mismatch is None:
            logging.info(f"Validation Prompt {prompt_idx}: MATCH")
            continue

        mismatch_found = True
        mismatch_idx, base_tok, spec_tok = mismatch
        logging.error(
            f"Validation Prompt {prompt_idx}: MISMATCH at token {mismatch_idx} "
            f"| baseline={base_tok} speculative={spec_tok}"
        )
    return not mismatch_found

def run_benchmark():
    target_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    draft_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    logging.info("=== TRUE SPECULATIVE DECODING BENCHMARK ===")
    target_model, draft_model, tokenizer = load_models_sequentially(target_name, draft_name)
    
    test_prompts = [
        "Explain the theory of general relativity in detail.",
        "Write a Python script to train a simple neural network from scratch.",
        "What are the economic implications of artificial intelligence?",
        "Describe the cellular respiration cycle step-by-step.",
        "Write a short science fiction story about a dyson sphere."
    ]
    
    num_prompts = len(test_prompts)
    max_tokens = 64
    k_drafts = 5
    
    logging.info("Warming up graphs...")
    warmup = tokenizer.encode("Warmup")
    speculative_decode(warmup, target_model, draft_model, max_tokens=10, k=k_drafts)
    baseline_decode(warmup, target_model, max_tokens=10)
    validation_ok = validate_speculative_correctness(
        tokenizer,
        target_model,
        draft_model,
        test_prompts,
        max_tokens=max_tokens,
        k=k_drafts,
    )
    
    logging.info(f"--- Starting Baseline (Target Only) vs Speculative (K={k_drafts}) ---")
    
    control_metrics = []
    cascade_metrics = []
    mismatch_found = not validation_ok
    
    for idx, prompt in enumerate(test_prompts):
        tokens = tokenizer.encode(prompt)
        
        # --- Baseline Run ---
        mx.metal.clear_cache()
        t0 = time.perf_counter()
        baseline_tokens = baseline_decode(tokens, target_model, max_tokens=max_tokens)
        mx.eval(target_model.parameters()) # ensure completion
        t1 = time.perf_counter()
        
        base_time = t1 - t0
        base_tps = max_tokens / base_time
        control_metrics.append(base_tps)
        
        # --- Speculative Run ---
        mx.metal.clear_cache()
        t2 = time.perf_counter()
        speculative_tokens = speculative_decode(tokens, target_model, draft_model, max_tokens=max_tokens, k=k_drafts)
        mx.eval(target_model.parameters(), draft_model.parameters()) # ensure completion
        t3 = time.perf_counter()
        
        spec_time = t3 - t2
        spec_tps = max_tokens / spec_time
        cascade_metrics.append(spec_tps)
        
        mismatch = find_first_mismatch(baseline_tokens, speculative_tokens)
        if mismatch is None:
            mismatch_status = "MATCH"
        else:
            mismatch_found = True
            mismatch_idx, base_tok, spec_tok = mismatch
            mismatch_status = (
                f"MISMATCH at token {mismatch_idx} "
                f"(baseline={base_tok}, speculative={spec_tok})"
            )

        logging.info(
            f"Prompt {idx+1}/{num_prompts} | Base: {base_tps:.2f} TPS | "
            f"Speculative: {spec_tps:.2f} TPS | Speedup: {((spec_tps/base_tps)-1)*100:.1f}% | "
            f"{mismatch_status}"
        )
        
        with open("reports/current_status.txt", "w") as f:
            f.write(f"TRUE SPECULATIVE CASCADE PROGRESS: {idx+1}/{num_prompts}\n")
            f.write(f"Latest Base: {base_tps:.2f} TPS\n")
            f.write(f"Latest Speculative: {spec_tps:.2f} TPS\n")
            
    avg_base = sum(control_metrics) / len(control_metrics)
    avg_spec = sum(cascade_metrics) / len(cascade_metrics)
    speedup = ((avg_spec / avg_base) - 1) * 100
    
    logging.info("=== FINAL RESULTS ===")
    logging.info(f"Avg Control TPS: {avg_base:.2f}")
    logging.info(f"Avg Cascade TPS: {avg_spec:.2f}")
    logging.info(f"Real Speedup: {speedup:.2f}%")
    logging.info(f"Mismatch Status: {'MISMATCH FOUND' if mismatch_found else 'NO MISMATCH'}")
    logging.info(f"{'INCORRECT IMPLEMENTATION' if mismatch_found else 'CORRECT IMPLEMENTATION'}")
    
    results = {
        "avg_control_tps": avg_base,
        "avg_cascade_tps": avg_spec,
        "speedup_percentage": speedup,
        "mismatch_found": mismatch_found,
        "status": "INCORRECT IMPLEMENTATION" if mismatch_found else "CORRECT IMPLEMENTATION",
    }
    
    with open("reports/benchmark_uma_cascade.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_benchmark()
