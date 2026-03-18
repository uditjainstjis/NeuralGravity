import time
import os
import csv
import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from mlx_lm import load
from datasets import load_dataset
import sys

# Link local library
sys.path.append(os.getcwd())
from neural_gravity.hybrid_adapter import HybridLinear
from neural_gravity.egmp_optimizer import EGMPOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HLRA-Validator")

# --- Standard LoRA Implementation ---
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=16.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.weight.shape[1]
        self.out_features = base_layer.weight.shape[0]
        self.scale = alpha / rank
        
        # Standard LoRA initialization
        import math
        self.A = mx.random.normal((rank, self.in_features)) * math.sqrt(1 / self.in_features)
        self.B = mx.zeros((self.out_features, rank))

    def _get_dequantized_weight(self):
        if hasattr(self.base_layer, "scales"):
            w_shape = (self.out_features, self.in_features)
            return mx.dequantize(
                self.base_layer.weight,
                self.base_layer.scales,
                self.base_layer.biases,
                self.base_layer.group_size,
                self.base_layer.bits
            ).reshape(w_shape)
        return self.base_layer.weight

    def __call__(self, x):
        W_0 = self._get_dequantized_weight()
        lora_update = self.B @ self.A * self.scale
        W_lora = W_0 + lora_update
        
        out = x @ W_lora.T
        if "bias" in self.base_layer and self.base_layer.bias is not None:
            out += self.base_layer.bias
        return out

def causal_lm_loss(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))

def inject_adapters(model, adapter_type="lora"):
    count = 0
    core = model.model if hasattr(model, "model") else model
    if hasattr(core, "layers"):
        for layer in core.layers:
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for mod in ["q_proj", "v_proj"]:
                    if hasattr(attn, mod):
                        base = getattr(attn, mod)
                        if adapter_type == "lora":
                            adapter = LoRALinear(base, rank=16)
                        elif adapter_type == "dora":
                            adapter = HybridLinear(base, rank=16, eora_rank=16)
                            # Zero out the SVD path for strict DoRA
                            adapter.A_eora = mx.zeros_like(adapter.A_eora)
                            adapter.B_eora = mx.zeros_like(adapter.B_eora)
                        elif adapter_type == "hlra":
                            # Full Hybrid Adapter
                            adapter = HybridLinear(base, rank=16, eora_rank=16)
                        setattr(attn, mod, adapter)
                        count += 1
    logger.info(f"Injected {count} {adapter_type.upper()} adapters into attention modules.")
    return model

def run_training(model_repo, steps, adapter_type):
    logger.info(f"--- Starting {adapter_type.upper()} Validation ---")
    mx.metal.clear_cache()
    
    model, tokenizer = load(model_repo)
    model.freeze()
    model = inject_adapters(model, adapter_type=adapter_type)
    
    # We use AdamW uniformly to isolate adapter impact
    opt = optimizers.AdamW(learning_rate=1e-4)
    # If it's HLRA, we use the EGMP projection to save memory organically
    if adapter_type == "hlra":
        opt = EGMPOptimizer(base_optimizer=opt, initial_rank=16)
        opt.init(model.trainable_parameters())
    else:
        opt.init(model.trainable_parameters())

    loss_fn = nn.value_and_grad(model, causal_lm_loss)
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    def data_gen():
        tokens = []
        for ex in dataset:
            text = f"Instruction: {ex.get('instruction','')}\\nOutput: {ex.get('output','')}"
            tokens.extend(tokenizer.encode(text))
        idx = 0
        while idx + (2 * 128) + 1 <= len(tokens):
            x, y = [], []
            for _ in range(2): # batch size 2
                chunk = tokens[idx : idx + 128 + 1]
                x.append(chunk[:-1]); y.append(chunk[1:])
                idx += 128
            yield mx.array(x), mx.array(y)

    iterator = data_gen()
    
    metrics = []
    
    for s in range(steps):
        x, y = next(iterator)
        loss, grads = loss_fn(model, x, y)
        opt.update(model, grads)
        
        # Eval and monitor memory natively
        mx.eval(model.trainable_parameters(), loss)
        if adapter_type == "hlra":
            mx.eval(opt.state)
            
        mem_mb = mx.metal.get_active_memory() / (1024 * 1024)
        loss_val = loss.item()
        
        if s % 20 == 0:
            logger.info(f"[{adapter_type.upper()}] Step {s} | Loss: {loss_val:.4f} | Mem: {mem_mb:.1f}MB")
            
        metrics.append((loss_val, mem_mb))
        
    # Free memory forcefully after run
    del model
    del opt
    import gc
    gc.collect()
    mx.metal.clear_cache()
    
    return metrics

def main():
    steps = 200
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    csv_file = "reports/hlra_validation.csv"
    
    # We test on the 0.5B to guarantee it clears memory boundaries repeatedly 
    # and provides a strong baseline comparison for adapter architectures.
    logger.info(f"Loaded target validation model: {model_name} for 3 consecutive runs.")
    
    lora_metrics = run_training(model_name, steps=steps, adapter_type="lora")
    dora_metrics = run_training(model_name, steps=steps, adapter_type="dora")
    hlra_metrics = run_training(model_name, steps=steps, adapter_type="hlra")
    
    # Write aggregated metrics to CSV
    logger.info(f"Writing aggregated validation data to {csv_file}")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "LoRA_Loss", "LoRA_Mem_MB", "DoRA_Loss", "DoRA_Mem_MB", "HLRA_Loss", "HLRA_Mem_MB"])
        
        for i in range(steps):
            writer.writerow([
                i, 
                lora_metrics[i][0], lora_metrics[i][1],
                dora_metrics[i][0], dora_metrics[i][1],
                hlra_metrics[i][0], hlra_metrics[i][1]
            ])
            
    logger.info("Validation sequence complete. Data exported successfully.")

if __name__ == "__main__":
    main()
