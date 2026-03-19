import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import heapq
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - TTA* - %(message)s')

class ReasoningNode:
    def __init__(self, sequence, log_prob, heuristic, depth):
        self.sequence = sequence  # List of tokens
        self.log_prob = log_prob  # g(n)
        self.heuristic = heuristic # h(n)
        self.depth = depth
        
    def f_score(self):
        # f(n) = g(n) + h(n)
        # We negate for max-priority queue (heapq is min-priority)
        return -(self.log_prob + self.heuristic)
    
    def __lt__(self, other):
        return self.f_score() < other.f_score()

def get_self_reflection_score(model, tokenizer, sequence):
    """
    Asks the model to evaluate the partial sequence.
    Returns a heuristic score h(n).
    """
    text = tokenizer.decode(sequence)
    reflection_prompt = f"{text}\n\nReview the above reasoning. Is this path likely to reach the correct answer? Rate from 1-10, output only the number."
    
    input_ids = tokenizer.encode(reflection_prompt)
    # Simple generation for speed
    response = generate(model, tokenizer, prompt=reflection_prompt, max_tokens=1, verbose=False)
    
    try:
        score = float(response.strip()) / 10.0
    except:
        score = 0.5 # fallback
        
    return score

def tta_star_search(model, tokenizer, prompt, max_iterations=10, beam_width=3):
    start_tokens = tokenizer.encode(prompt)
    
    # Priority Queue of Nodes
    # (negative f_score, node)
    pq = []
    
    initial_node = ReasoningNode(start_tokens, 0.0, 0.5, 0)
    heapq.heappush(pq, initial_node)
    
    best_final_node = None
    
    for i in range(max_iterations):
        if not pq:
            break
            
        current = heapq.heappop(pq)
        logging.info(f"Iter {i} | Expanding node depth {current.depth} | f-score {-current.f_score():.4f}")
        
        # Check if terminal (e.g., contains end token or reached a certain length)
        if tokenizer.eos_token_id in current.sequence or current.depth > 5:
            if not best_final_node or (current.log_prob > best_final_node.log_prob):
                best_final_node = current
            continue

        # Generate candidates (Thought Branches)
        # For simplicity, we sample beam_width independent continuations
        for _ in range(beam_width):
            # Generate a "thought" (e.g., 20 tokens)
            tokens = []
            current_log_prob = current.log_prob
            
            # Generate step by step to track logprobs
            temp_seq = mx.array(current.sequence) # Must be 1D
            sampler = make_sampler(0.7)
            
            # We just take a chunk of generation
            for step, (token, logprobs) in enumerate(generate_step(temp_seq, model, sampler=sampler)):
                if step >= 20: break
                tokens.append(token)
                
                # generate_step yields logits of shape (vocab_size,) or (1, vocab_size,)
                # we must handle potential 2D shapes identically.
                if logprobs.ndim > 1:
                    log_p = logprobs[0] - mx.logsumexp(logprobs[0], axis=-1, keepdims=True)
                else:
                    log_p = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
                
                token_logprob = log_p[token].item()
                current_log_prob += token_logprob
                if token == tokenizer.eos_token_id: break
            
            new_sequence = current.sequence + tokens
            h = get_self_reflection_score(model, tokenizer, new_sequence)
            
            new_node = ReasoningNode(new_sequence, current_log_prob, h, current.depth + 1)
            heapq.heappush(pq, new_node)

    if not best_final_node and pq:
        best_final_node = heapq.heappop(pq)
    elif not best_final_node:
        best_final_node = current
        
    return best_final_node

def main():
    model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    logging.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    prompt = "Problem: Solve for x: 2x + 5 = 15. Reasoning:"
    logging.info(f"Starting TTA* for prompt: {prompt}")
    
    result_node = tta_star_search(model, tokenizer, prompt, max_iterations=5, beam_width=2)
    
    if result_node:
        print("\n--- TTA* Best Reasoning Path ---")
        print(tokenizer.decode(result_node.sequence))
        print(f"Final Log Prob: {result_node.log_prob:.4f}")
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
