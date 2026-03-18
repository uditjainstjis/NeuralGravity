import time
import json
import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from mlx_lm import load
from datasets import load_dataset
import sys
import os

# Link local library
sys.path.append(os.getcwd())
from neural_gravity.hybrid_adapter import HybridLinear
from neural_gravity.egmp_optimizer import EGMPOptimizer
from neural_gravity.persistence import ImmortalTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HLRA-Ablation")

def causal_lm_loss(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))

def inject_adapters(model, enable_eora=True):
    count = 0
    core = model.model if hasattr(model, "model") else model
    if hasattr(core, "layers"):
        for layer in core.layers:
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for mod in ["q_proj", "v_proj"]:
                    if hasattr(attn, mod):
                        base = getattr(attn, mod)
                        adapter = HybridLinear(base, rank=16, eora_rank=16)
                        if not enable_eora:
                            # Isolate mathematical ablation by totally zeroing the 
                            # SVD-initialized components to evaluate base representation
                            adapter.A_eora = mx.zeros_like(adapter.A_eora)
                            adapter.B_eora = mx.zeros_like(adapter.B_eora)
                        setattr(attn, mod, adapter)
                        count += 1
    return model

def run_training(model_repo, steps, enable_eora, run_name):
    logger.info(f"[{run_name}] Loading Model: {model_repo}")
    model, tokenizer = load(model_repo)
    model.freeze()
    
    model = inject_adapters(model, enable_eora=enable_eora)
    
    base_opt = optimizers.AdamW(learning_rate=1e-4)
    opt = EGMPOptimizer(base_optimizer=base_opt, initial_rank=16)
    opt.init(model.trainable_parameters())
    
    loss_fn = nn.value_and_grad(model, causal_lm_loss)
    
    logger.info(f"[{run_name}] Loading Alpaca Cleaned Dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    def data_gen():
        tokens = []
        for ex in dataset:
            text = f"Instruction: {ex.get('instruction','')}\nOutput: {ex.get('output','')}"
            tokens.extend(tokenizer.encode(text))
        idx = 0
        while idx + (2 * 128) + 1 <= len(tokens):
            x, y = [], []
            for _ in range(2): # batch sizes
                chunk = tokens[idx : idx + 128 + 1]
                x.append(chunk[:-1]); y.append(chunk[1:])
                idx += 128
            yield mx.array(x), mx.array(y)

    iterator = data_gen()
    
    final_loss = 0.0
    logger.info(f"[{run_name}] Starting 200-step training matrix...")
    for s in range(steps):
        x, y = next(iterator)
        loss, grads = loss_fn(model, x, y)
        opt.update(model, grads)
        
        # Enforce step-by-step evaluation to prevent MLX lazy-graph memory leaks
        mx.eval(model.trainable_parameters(), opt.state, loss)
        
        if s % 20 == 0:
            logger.info(f"[{run_name}] Step {s} | Loss: {loss.item():.4f}")
        final_loss = loss.item()
        
    return final_loss

def main():
    immortal = ImmortalTrainer(save_callback=lambda: logger.info("Immortal hook pinged."))
    immortal.go_immortal()
    
    try:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info("============== ABLATION: DoRA ONLY ==============")
        dora_loss = run_training(model_name, steps=200, enable_eora=False, run_name="DoRA-Only")
        
        logger.info("============== ABLATION: HLRA (DoRA+EoRA) ==============")
        hlra_loss = run_training(model_name, steps=200, enable_eora=True, run_name="HLRA-Dual")
        
        results = {
            "baseline_dora_loss": dora_loss,
            "hlra_loss": hlra_loss,
            "improvement_delta": dora_loss - hlra_loss,
            "relative_percent": ((dora_loss - hlra_loss) / dora_loss) * 100
        }
        
        with open("reports/hlra_ablation_results.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved exact structural verification to reports/hlra_ablation_results.json: {results}")

    finally:
        immortal.exit_immortal()

if __name__ == "__main__":
    main()
