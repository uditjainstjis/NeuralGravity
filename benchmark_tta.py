import json
import logging
import re
from datasets import load_dataset
from mlx_lm import load, generate
from reasoning_search import tta_star_search
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Benchmark - %(message)s')

def extract_answer(text):
    # Extract boxed content or the last number
    matches = re.findall(r'\\boxed{([^}]*)}', text)
    if matches:
        return matches[-1].strip()
    
    # Fallback to last number if no box
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return ""

def main():
    model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    logging.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    # Load MATH-500 subset
    logging.info("Loading MATH-500 dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as e:
        # Fallback to generic MATH if required
        dataset = load_dataset("competition_math", split="test", trust_remote_code=True)
    
    # Take first 100
    dataset = dataset.select(range(100))
    
    metrics = {
        "total_questions": 100,
        "greedy_correct": 0,
        "tta_correct": 0,
        "greedy_total_tokens": 0,
        "tta_total_tokens": 0,
        "greedy_time": 0.0,
        "tta_time": 0.0,
        "details": []
    }
    
    for i, item in enumerate(dataset):
        problem = item["problem"]
        ground_truth = extract_answer(item["solution"])
        logging.info(f"--- Question {i+1} ---")
        
        prompt = f"Problem: {problem}\nSolve the problem step by step and put the final answer in \\boxed{{}}.\nReasoning:"
        
        # 1. Greedy Run
        logging.info("Running Greedy Decoding...")
        start_greedy = time.time()
        greedy_resp = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
        greedy_time = time.time() - start_greedy
        
        greedy_ans = extract_answer(greedy_resp)
        tokens_greedy = len(tokenizer.encode(greedy_resp))
        metrics["greedy_total_tokens"] += tokens_greedy
        metrics["greedy_time"] += greedy_time
        
        greedy_is_correct = (greedy_ans == ground_truth)
        if greedy_is_correct: metrics["greedy_correct"] += 1
            
        # 2. TTA* Run
        logging.info("Running TTA* Decoding...")
        start_tta = time.time()
        best_node = tta_star_search(model, tokenizer, prompt, max_iterations=4, beam_width=2)
        tta_time = time.time() - start_tta
        
        if best_node:
            # The sequence includes the prompt, so decode only the generated portion
            tta_full = tokenizer.decode(best_node.sequence)
            tta_resp = tta_full.replace(prompt, "")
            tta_ans = extract_answer(tta_resp)
            tokens_tta = len(best_node.sequence) - len(tokenizer.encode(prompt))
        else:
            tta_resp = ""
            tta_ans = ""
            tokens_tta = 0
            
        metrics["tta_total_tokens"] += tokens_tta
        metrics["tta_time"] += tta_time
        
        tta_is_correct = (tta_ans == ground_truth)
        if tta_is_correct: metrics["tta_correct"] += 1
            
        logging.info(f"Target: {ground_truth} | Greedy: {greedy_ans} ({greedy_is_correct}) | TTA*: {tta_ans} ({tta_is_correct})")
        
        metrics["details"].append({
            "problem": problem,
            "target": ground_truth,
            "greedy_answer": greedy_ans,
            "greedy_correct": greedy_is_correct,
            "tta_answer": tta_ans,
            "tta_correct": tta_is_correct
        })
        
        # Incremental save
        with open("final_tta_benchmark_results.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
    # Final Summary
    logging.info("=== BENCHMARK COMPLETE ===")
    logging.info(f"Greedy Accuracy: {metrics['greedy_correct'] / metrics['total_questions'] * 100}%")
    logging.info(f"TTA* Accuracy:   {metrics['tta_correct'] / metrics['total_questions'] * 100}%")
    
if __name__ == "__main__":
    main()
