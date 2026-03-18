import os
import signal
import subprocess
import threading
import logging
import mlx.core as mx
from safetensors.numpy import save_file

logger = logging.getLogger("NeuralGravity-Persistence")

class ImmortalTrainer:
    """
    Wraps the training process with power management overrides to survive lid closure.
    Handles SIGTERM gracefully by triggering an immediate emergency save.
    """
    def __init__(self, save_callback=None):
        self.save_callback = save_callback
        self.caffeinate_process = None
        self.pmset_applied = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGHUP, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def go_immortal(self):
        """Invoke pmset array and caffeinate."""
        logger.info("Engaging pmset (requires sudo) and caffeinate...")
        try:
            # Requires passwordless sudo or user to provide password upfront
            # using -n to prevent hanging if password is required
            subprocess.run(["sudo", "-n", "pmset", "-a", "disablesleep", "1"], check=True)
            self.pmset_applied = True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to run pmset (sudo lacking?): {e}")

        try:
            # -is protects idle sleep and system sleep
            self.caffeinate_process = subprocess.Popen(["caffeinate", "-is"])
        except Exception as e:
            logger.warning(f"Failed to run caffeinate: {e}")

    def exit_immortal(self):
        """Revert pmset and kill caffeinate."""
        logger.info("Exiting immortal state...")
        if self.pmset_applied:
            try:
                subprocess.run(["sudo", "pmset", "-a", "disablesleep", "0"], check=False)
            except Exception as e:
                logger.error(f"Failed to reset pmset: {e}")
        
        if self.caffeinate_process:
            self.caffeinate_process.terminate()

    def _handle_sigterm(self, signum, frame):
        logger.error(f"Received signal {signum}. Triggering emergency checkpoint...")
        if self.save_callback:
            self.save_callback(emergency=True)
        self.exit_immortal()
        exit(0)

    def _handle_sigint(self, signum, frame):
        logger.info("Keyboard interrupt received.")
        if self.save_callback:
            self.save_callback(emergency=True)
        self.exit_immortal()
        exit(0)


class AsyncCheckpointer:
    """
    Saves model weights to .safetensors using a background thread.
    Dual-slot rotation to prevent corruption.
    """
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_thread = None
        self.slot = 0 # Rotates between 0 and 1

    def is_saving(self):
        return self.save_thread is not None and self.save_thread.is_alive()

    def async_save(self, model, step, emergency=False):
        """
        model: mlx_lm model or nn.Module
        Extracts weights into numpy arrays to save via safetensors.
        """
        if self.is_saving():
            logger.warning("Previous save operation still running. Skipping.")
            return

        # Double buffer: we need to evaluate MLX arrays so the main thread
        # can continue changing them without affecting the save.
        
        state_dict_numpy = {}
        for k, v in model.parameters().items():
            # mx.eval(v) implicitly happens when converting to numpy
            # We copy to prevent modification during train loop
            import numpy as np
            state_dict_numpy[k] = np.array(v)

        def save_task(state_dict, current_slot, step_num):
            filename = os.path.join(self.save_dir, f"checkpoint_slot_{current_slot}.safetensors")
            # Save step info to filename or another file
            temp_name = filename + ".tmp"
            logger.info(f"Background thread starting save to {filename}")
            try:
                save_file(state_dict, temp_name)
                os.rename(temp_name, filename)
                logger.info(f"Saved checkpoint for step {step_num}")
            except Exception as e:
                logger.error(f"Failed to save safetensors: {e}")

        # Choose slot
        current_slot = self.slot
        self.slot = 1 - self.slot # Rotate 0 -> 1 -> 0

        if emergency:
            # Run synchronously if process is shutting down
            save_task(state_dict_numpy, current_slot, step)
        else:
            self.save_thread = threading.Thread(
                target=save_task, 
                args=(state_dict_numpy, current_slot, step),
                daemon=True
            )
            self.save_thread.start()
