import time
import json
import logging
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

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

def speculative_decode(prompt_tokens, target_model, draft_model, max_tokens=128, k=5):
    # Initialize caches
    target_cache = make_prompt_cache(target_model)
    draft_cache = make_prompt_cache(draft_model)

    y = mx.array(prompt_tokens, dtype=mx.uint32)
    draft_y = y

    # Prefill
    target_model(y[None], cache=target_cache)
    draft_model(draft_y[None], cache=draft_cache)
    mx.eval([c.state for c in target_cache] + [c.state for c in draft_cache])

    ntoks = 0
    accepted_tokens = []
    
    while ntoks < max_tokens:
        num_draft = min(max_tokens - ntoks, k)
        
        # 1. Draft K candidate tokens autoregressively
        drafts = []
        curr_draft_y = draft_y
        for _ in range(num_draft):
            logits = draft_model(curr_draft_y[None], cache=draft_cache)
            next_tok = mx.argmax(logits[:, -1, :], axis=-1)
            drafts.append(next_tok)
            curr_draft_y = next_tok
        
        draft_array = mx.concatenate(drafts)
        
        # 2. Target model evaluates the draft string in parallel
        # We append the drafts to the current verified sequence to evaluate
        eval_seq = mx.concatenate([y, draft_array])
        
        # We need the last `num_draft + 1` logits (to check the drafts + predict the token after the prefix)
        target_logits = target_model(eval_seq[None], cache=target_cache)
        target_logits = target_logits[:, -(num_draft + 1):, :]
        target_preds = mx.argmax(target_logits, axis=-1).squeeze(0)
        
        # 3. Verification Loop
        draft_list = draft_array.tolist()
        target_list = target_preds.tolist()
        
        match_len = 0
        for i in range(num_draft):
            if target_list[i] == draft_list[i]:
                match_len += 1
            else:
                break
        
        # The accepted sequence is the matching prefix + the target's correction/continuation
        accepted = target_list[:match_len + 1]
        ntoks += len(accepted)
        accepted_tokens.extend(accepted)
        
        if ntoks >= max_tokens:
            break
            
        # 4. Rollback Caches for rejected tokens
        # Target evaluated `num_draft + 1` tokens but we only accepted `match_len + 1`. Rollback the difference.
        if num_draft > match_len:
            target_rollback = num_draft - match_len
            for c in target_cache:
                c.trim(target_rollback)
            
            # Drafter generated `num_draft` tokens, we accepted `match_len` of its drafts
            draft_rollback = num_draft - match_len
            for c in draft_cache:
                c.trim(draft_rollback)

        # Set up the next sequence base for the targeted next step
        y = mx.array([accepted[-1]], dtype=mx.uint32)
        draft_y = y

    return accepted_tokens

def baseline_decode(prompt_tokens, target_model, max_tokens=128):
    target_cache = make_prompt_cache(target_model)
    y = mx.array(prompt_tokens, dtype=mx.uint32)
    
    target_model(y[None], cache=target_cache)
    mx.eval([c.state for c in target_cache])
    
    accepted_tokens = []
    
    for _ in range(max_tokens):
        logits = target_model(y[None], cache=target_cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        accepted_tokens.append(next_tok.item())
        y = next_tok
        mx.eval(y)
        
    return accepted_tokens

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
    
    logging.info(f"--- Starting Baseline (Target Only) vs Speculative (K={k_drafts}) ---")
    
    control_metrics = []
    cascade_metrics = []
    
    for idx, prompt in enumerate(test_prompts):
        tokens = tokenizer.encode(prompt)
        
        # --- Baseline Run ---
        mx.metal.clear_cache()
        t0 = time.perf_counter()
        baseline_decode(tokens, target_model, max_tokens=max_tokens)
        mx.eval(target_model.parameters()) # ensure completion
        t1 = time.perf_counter()
        
        base_time = t1 - t0
        base_tps = max_tokens / base_time
        control_metrics.append(base_tps)
        
        # --- Speculative Run ---
        mx.metal.clear_cache()
        t2 = time.perf_counter()
        speculative_decode(tokens, target_model, draft_model, max_tokens=max_tokens, k=k_drafts)
        mx.eval(target_model.parameters(), draft_model.parameters()) # ensure completion
        t3 = time.perf_counter()
        
        spec_time = t3 - t2
        spec_tps = max_tokens / spec_time
        cascade_metrics.append(spec_tps)
        
        logging.info(f"Prompt {idx+1}/{num_prompts} | Base: {base_tps:.2f} TPS | Speculative: {spec_tps:.2f} TPS | Speedup: {((spec_tps/base_tps)-1)*100:.1f}%")
        
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
    
    results = {
        "avg_control_tps": avg_base,
        "avg_cascade_tps": avg_spec,
        "speedup_percentage": speedup
    }
    
    with open("reports/benchmark_uma_cascade.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_benchmark()
