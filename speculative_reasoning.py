import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import heapq
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Speculative A* - %(message)s')

class ReasoningNode:
    def __init__(self, sequence, log_prob, heuristic, depth):
        self.sequence = sequence
        self.log_prob = log_prob
        self.heuristic = heuristic
        self.depth = depth
        
    def f_score(self):
        # A* Cost routing: balance g(n) prob-density and h(n) self-reflection
        # Negate for python's min-heap
        return -(self.log_prob * 1.0 + self.heuristic * 10.0)
    
    def __lt__(self, other):
        return self.f_score() < other.f_score()

def get_self_reflection_score(model, tokenizer, sequence):
    text = tokenizer.decode(sequence)
    prompt = f"{text}\n\nReview the above reasoning. Is this path likely to reach the correct answer? Rate from 1-10, output only the number."
    
    try:
        response = generate(model, tokenizer, prompt=prompt, max_tokens=2, verbose=False)
        score = float(response.strip().split()[0]) / 10.0
    except:
        score = 0.5
    return min(max(score, 0.0), 1.0)

def greedy_step(model, cache, tokens, n_predict=1):
    logits = model(tokens[None], cache=cache)
    logits = logits[:, -n_predict:, :]
    return mx.argmax(logits.squeeze(0), axis=-1), logits.squeeze(0)

def speculative_thought_expansion(target, draft, prompt_tokens, max_tokens=20, k=5):
    """
    Expands a branch using the exact greedy cascade logic but returns
    both the tokens and the cumulative log-probability g(n) natively extracted.
    """
    t_cache = make_prompt_cache(target)
    d_cache = make_prompt_cache(draft)

    y = mx.array(prompt_tokens, dtype=mx.uint32)
    prefix_tokens = list(prompt_tokens)
    accepted_tokens = []
    total_logprob = 0.0
    ntoks = 0

    def rewind_caches(num_draft, num_accept):
        trim_prompt_cache(t_cache, num_draft - num_accept)
        trim_prompt_cache(d_cache, max(num_draft - num_accept - 1, 0))

    try:
        while ntoks < max_tokens:
            num_draft = min(k, max_tokens - ntoks)

            d_cache = make_prompt_cache(draft)
            drafts = []
            curr = mx.array(prefix_tokens, dtype=mx.uint32)
            for _ in range(num_draft):
                next_tok, _ = greedy_step(draft, d_cache, curr)
                mx.eval(next_tok)
                drafts.append(next_tok.item())
                curr = next_tok

            draft_array = mx.array(drafts, dtype=mx.uint32)
            target_input = mx.concatenate([y, draft_array])
            t_preds, t_logits = greedy_step(target, t_cache, target_input, num_draft + 1)
            mx.eval(t_preds, t_logits, draft_array)
            
            t_preds_list = t_preds.tolist()
            
            # Compute exact target log probabilities for the path g(n)
            log_p = t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True)

            match_len = 0
            while match_len < num_draft and t_preds_list[match_len] == drafts[match_len]:
                tok = drafts[match_len]
                accepted_tokens.append(tok)
                # The prediction for drafts[match_len] is the logit at index `match_len`
                total_logprob += log_p[match_len, tok].item()
                ntoks += 1
                match_len += 1
                if ntoks >= max_tokens:
                    break

            if ntoks >= max_tokens:
                break

            fallback = t_preds_list[match_len]
            accepted_tokens.append(fallback)
            total_logprob += log_p[match_len, fallback].item()
            ntoks += 1

            prefix_tokens.extend(drafts[:match_len])
            prefix_tokens.append(fallback)
            y = mx.array([fallback], dtype=mx.uint32)
            rewind_caches(num_draft, match_len)
    finally:
        rewind_caches(num_draft, match_len)

    return accepted_tokens, total_logprob

def speculative_tta_star(target_model, draft_model, tokenizer, prompt, max_iterations=5, beam_width=3):
    start_tokens = tokenizer.encode(prompt)
    pq = []
    
    initial_node = ReasoningNode(start_tokens, 0.0, 0.5, 0)
    heapq.heappush(pq, initial_node)
    best_final_node = None
    
    for i in range(max_iterations):
        if not pq: break
        current = heapq.heappop(pq)
        logging.info(f"A* Iter {i} | Depth {current.depth} | f-score {-current.f_score():.4f}")
        
        if tokenizer.eos_token_id in current.sequence or current.depth > 4:
            if not best_final_node or (current.log_prob > best_final_node.log_prob):
                best_final_node = current
            continue

        # 1. Stochastic Branching from Target
        # To get distinct thought paths, we sample `beam_width` orthogonal tokens to kick off the branch
        t_cache = make_prompt_cache(target_model)
        seq_arr = mx.array(current.sequence, dtype=mx.uint32)
        logits = target_model(seq_arr[None], cache=t_cache)
        next_logits = logits[0, -1, :]
        
        # Sample top-K or diverse tokens with temperature
        temperature = 0.8
        sampled_indices = mx.random.categorical(next_logits * (1.0 / temperature), shape=(beam_width,))
        log_p_base = next_logits - mx.logsumexp(next_logits, axis=-1)
        
        sampled_tokens = sampled_indices.tolist()
        
        for b_idx in range(beam_width):
            st_tok = sampled_tokens[b_idx]
            base_lp = log_p_base[st_tok].item()
            
            # Start the speculative fast-rollout from the base sequence + sampled token
            branch_seq = current.sequence + [st_tok]
            
            # Speculative Thought Expansion (Zero-Overhead Logprobs)
            gen_tokens, gen_lp = speculative_thought_expansion(
                target_model, draft_model, branch_seq, max_tokens=15, k=5
            )
            
            full_seq = branch_seq + gen_tokens
            full_lp = current.log_prob + base_lp + gen_lp
            
            # Endogenous Heuristic evaluation
            h_score = get_self_reflection_score(target_model, tokenizer, full_seq)
            
            new_node = ReasoningNode(full_seq, full_lp, h_score, current.depth + 1)
            heapq.heappush(pq, new_node)
            
    return best_final_node or (heapq.heappop(pq) if pq else current)

def main():
    target_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    draft_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    logging.info("Loading Breakthrough Speculative A* Architecture...")
    target, tokenizer = load(target_name)
    draft, _ = load(draft_name)
    
    prompt = "Problem: Solve for x: 2x + 5 = 15. Reasoning:"
    logging.info("Executing TTA* Cascade Search...")
    
    t0 = time.time()
    best_node = speculative_tta_star(target, draft, tokenizer, prompt, max_iterations=4, beam_width=2)
    t1 = time.time()
    
    print("\n" + "="*50)
    print("SPECULATIVE A* FINAL REASONING PATH")
    print("="*50)
    print(tokenizer.decode(best_node.sequence))
    print(f"\nFinal Cost Routed f(n): {-best_node.f_score():.4f} | Nodes generated in {t1-t0:.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()
