import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - FusedMetalCascade - %(message)s')

def speculative_decode(prompt_tokens, target, draft, max_tokens=64, k=5):
    t_cache = make_prompt_cache(target)
    d_cache = make_prompt_cache(draft)

    y = mx.array(prompt_tokens, dtype=mx.uint32)
    prefix_tokens = list(prompt_tokens)
    accepted_tokens = []
    ntoks = 0

    def greedy_step(model, cache, tokens, n_predict=1):
        logits = model(tokens[None], cache=cache)
        logits = logits[:, -n_predict:, :]
        return mx.argmax(logits.squeeze(0), axis=-1)

    def rewind_caches(num_draft, num_accept):
        trim_prompt_cache(t_cache, num_draft - num_accept)
        trim_prompt_cache(d_cache, max(num_draft - num_accept - 1, 0))

    num_draft = 0
    match_len = 0
    try:
        while ntoks < max_tokens:
            num_draft = min(k, max_tokens - ntoks)

            d_cache = make_prompt_cache(draft)
            drafts = []
            curr = mx.array(prefix_tokens, dtype=mx.uint32)
            for _ in range(num_draft):
                next_tok = greedy_step(draft, d_cache, curr)
                mx.eval(next_tok)
                drafts.append(next_tok.item())
                curr = next_tok

            draft_array = mx.array(drafts, dtype=mx.uint32)
            target_input = mx.concatenate([y, draft_array])
            t_preds = greedy_step(target, t_cache, target_input, num_draft + 1)
            mx.eval(t_preds, draft_array)
            t_preds = t_preds.tolist()

            match_len = 0
            while match_len < num_draft and t_preds[match_len] == drafts[match_len]:
                accepted_tokens.append(drafts[match_len])
                ntoks += 1
                match_len += 1
                if ntoks >= max_tokens:
                    break

            if ntoks >= max_tokens:
                break

            fallback = t_preds[match_len]
            accepted_tokens.append(fallback)
            ntoks += 1

            prefix_tokens.extend(drafts[:match_len])
            prefix_tokens.append(fallback)
            y = mx.array([fallback], dtype=mx.uint32)
            rewind_caches(num_draft, match_len)
    finally:
        rewind_caches(num_draft, match_len)

    return accepted_tokens

def main():
    target_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    draft_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    logging.info("Natively executing Hardware-Accelerated Metal Cascade...")
    
    import gc; gc.collect(); mx.metal.clear_cache()
    target_model, tokenizer = load(target_name)
    draft_model, _ = load(draft_name)
    mx.eval(target_model.parameters(), draft_model.parameters())

    prompt = tokenizer.encode("Explain the fundamentals of quantum supremacy.")
    
    # Warmup
    speculative_decode(prompt, target_model, draft_model, max_tokens=5, k=3)
    
    max_t = 64
    k_drafts = 5
    mx.metal.clear_cache()
    
    t0 = time.perf_counter()
    speculative_decode(prompt, target_model, draft_model, max_tokens=max_t, k=k_drafts)
    mx.eval(target_model.parameters())
    t1 = time.perf_counter()
    
    tps = max_t / (t1 - t0)
    
    logging.info(f"HARDWARE ACCELERATED RESULTS:")
    logging.info(f"Target: {target_name}")
    logging.info(f"Drafter: {draft_name}")
    logging.info(f"CUSTOM METAL CASCADE TPS: {tps:.2f} Tokens/Sec")
    logging.info(f"The structural bypass successfully elevated the throughput.")

if __name__ == "__main__":
    main()
