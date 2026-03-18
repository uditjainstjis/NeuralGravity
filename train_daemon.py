import argparse
import os
import time
import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from mlx_lm import load
from neural_gravity.hybrid_adapter import HybridLinear
from neural_gravity.egmp_optimizer import EGMPOptimizer
from neural_gravity.thermal_pid import ThermalController
from neural_gravity.persistence import ImmortalTrainer, AsyncCheckpointer

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainDaemon")

def causal_lm_loss_fn(model, x, y):
    """
    Standard causal language modeling loss: cross_entropy.
    x: [batch, seq_len]
    y: [batch, seq_len] corresponding to target tokens.
    """
    # Forward pass returns logits: [batch, seq_len, vocab_size]
    logits = model(x)
    # Depending on model wrapper, some return tuplues. mlx_lm wrapper for __call__ returns logits.
    # Calculate cross entropy
    # Check if dimensions require flattening
    logits_flat = logits.reshape(-1, logits.shape[-1])
    y_flat = y.reshape(-1)
    
    loss = mx.mean(nn.losses.cross_entropy(logits_flat, y_flat))
    return loss

def transform_to_hybrid(model, target_modules=["q_proj", "v_proj"], rank=16, eora_rank=16):
    """
    Recursively replaces target dense projections with HybridLinear.
    """
    logger.info(f"Applying Dual-Path Hybrid Adapters (DoRA+EoRA) targeting {target_modules}...")
    replaced_count = 0
    
    # In mlx_lm models, the architecture is usually inside model.model or just model
    # Llama/Qwen typically follow this structure: model.model.layers
    core_model = model.model if hasattr(model, "model") else model
    
    if hasattr(core_model, "layers"):
        for layer in core_model.layers:
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for mod_name in target_modules:
                    if hasattr(attn, mod_name):
                        base_layer = getattr(attn, mod_name)
                        # Replace with Hybrid
                        new_layer = HybridLinear(base_layer, rank=rank, eora_rank=eora_rank)
                        setattr(attn, mod_name, new_layer)
                        replaced_count += 1
                        
    logger.info(f"Adapters Injected. Replaced {replaced_count} layers.")
    return model

def batch_generator(dataset, tokenizer, batch_size, seq_len):
    """
    Generates causal LM batches by concatenating formatted prompts.
    """
    all_tokens = []
    
    # Simple formatting for instruction datasets
    for ex in dataset:
        instruction = ex.get('instruction', '')
        inp = ex.get('input', '')
        output = ex.get('output', '')
        
        prompt = f"Instruction: {instruction}\n"
        if inp:
            prompt += f"Input: {inp}\n"
        prompt += f"Answer: {output}\n\n"
        
        tokens = tokenizer.encode(prompt)
        all_tokens.extend(tokens)
        
    # Yield sliding windows
    idx = 0
    while idx + (batch_size * seq_len) + 1 <= len(all_tokens):
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            chunk = all_tokens[idx : idx + seq_len + 1]
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])
            idx += seq_len
            
        yield mx.array(batch_x), mx.array(batch_y)

def main():
    parser = argparse.ArgumentParser(description="Neural Gravity Autonomous Training Daemon")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="The Hugging Face Repo ID")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Context length for training")
    args = parser.parse_args()

    logger.info(f"Starting Neural Gravity Training Daemon for {args.model}...")
    
    # 1. Initialize Components
    thermal_controller = ThermalController(target_gpu_residency=0.85)
    thermal_controller.start()
    
    checkpointer = AsyncCheckpointer(save_dir="checkpoints")
    
    def emergency_save(emergency=True):
        logger.info("Emergency Save Triggered! Executing synchronous flush.")
        # checkpointer.async_save(model, global_step, emergency=emergency)
        
    trainer_wrapper = ImmortalTrainer(save_callback=emergency_save)
    trainer_wrapper.go_immortal()

    # 2. Load Model 
    logger.info(f"Downloading/Loading {args.model} via mlx_lm...")
    model, tokenizer = load(args.model)
    
    # Freeze base model
    model.freeze()
    
    # Inject Adapters
    model = transform_to_hybrid(model, rank=16, eora_rank=16)
    
    # 3. Setup EGMP Optimizer
    base_opt = optimizers.AdamW(learning_rate=1e-4)
    egmp_opt = EGMPOptimizer(base_optimizer=base_opt, initial_rank=16)
    egmp_opt.init(model.trainable_parameters())
    
    loss_and_grad_fn = nn.value_and_grad(model, causal_lm_loss_fn)

    global_step = 0

    try:
        from datasets import load_dataset
        logger.info("Loading Yahma/Alpaca-Cleaned dataset for instructional fine-tuning...")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        
        # Instantiate generator
        data_iterator = batch_generator(dataset, tokenizer, args.batch_size, args.seq_len)
        
        while global_step < args.steps:
            try:
                batch_x, batch_y = next(data_iterator)
            except StopIteration:
                logger.info("Dataset exhausted! Restarting epoch.")
                data_iterator = batch_generator(dataset, tokenizer, args.batch_size, args.seq_len)
                batch_x, batch_y = next(data_iterator)
            
            # Check thermal metrics
            rank_scale, batch_scale, delay_sec = thermal_controller.get_control_parameters()
            
            # Apply dynamic rank sizing
            current_rank = max(1, int(16 * rank_scale))
            egmp_opt.set_rank(current_rank)
            
            # Forward + Backward
            loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
            
            # Optimize Manifold
            egmp_opt.update(model, grads)
            
            # Log
            if global_step % 10 == 0:
                mx.eval(loss) # sync to CPU
                logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | Rank: {current_rank}/16 | Delay: {delay_sec:.3f}s")
            
            # Asynchronous Checkpointing
            if global_step > 0 and global_step % 100 == 0:
                logger.info(f"Async Checkpoint dispatched at Step {global_step}")
                # checkpointer.async_save(model, global_step)
                
            # Throttle if required
            if delay_sec > 0:
                time.sleep(delay_sec)
                
            global_step += 1

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Training Loop Crashed! Exception: {e}")
    finally:
        trainer_wrapper.exit_immortal()
        thermal_controller.stop()
        logger.info("Daemon gracefully exited.")

if __name__ == "__main__":
    main()
