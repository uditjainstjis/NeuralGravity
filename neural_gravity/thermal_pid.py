import subprocess
import threading
import time
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThermalPID")

class ThermalController:
    """
    PID Controller for Apple Silicon M3 to maintain 85% GPU residency.
    Monitors `powermetrics` and adjusts the target rank scale and micro-batch delays.
    """
    def __init__(self, target_gpu_residency=0.85):
        self.target_residency = target_gpu_residency
        
        # PID constants
        self.Kp = 0.5
        self.Ki = 0.1
        self.Kd = 0.05
        
        self.integral = 0
        self.previous_error = 0
        
        # Output control values
        self.current_rank_scale = 1.0  # 1.0 = Max Rank, 0.0 = Minimum Rank
        self.current_delay_ms = 0      # Delay introduced between batches
        self.current_batch_scale = 1.0 # 1.0 = Max Batch
        
        # State
        self.current_gpu_residency = 0.0
        self.thermal_pressure = "Nominal"
        
        self.running = False
        self.monitor_thread = None

    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Thermal PID Controller started.")

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Thermal PID Controller stopped.")

    def update_pid(self, current_residency):
        error = self.target_residency - current_residency
        
        # Prevent integral windup
        self.integral = max(-1.0, min(1.0, self.integral + error))
        
        derivative = error - self.previous_error
        self.previous_error = error
        
        # Calculate control variable (adjustment)
        adjustment = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        # Apply adjustment to rank scale
        # If error > 0 (residency < target), we can increase rank scale (max 1.0)
        # If error < 0 (residency > target), we need to reduce rank scale (min 0.1)
        self.current_rank_scale = max(0.1, min(1.0, self.current_rank_scale + adjustment))
        
        # Map thermal pressure states to deterministic overrides
        if self.thermal_pressure == "Nominal":
            self.current_batch_scale = 1.0
            self.current_delay_ms = max(0.0, -adjustment * 100) if error < 0 else 0.0
        elif self.thermal_pressure == "Fair":
            self.current_rank_scale = min(self.current_rank_scale, 0.8)
            self.current_delay_ms = max(10.0, self.current_delay_ms)
        elif self.thermal_pressure == "Serious":
            self.current_batch_scale = 0.5
            self.current_rank_scale = min(self.current_rank_scale, 0.5)
            self.current_delay_ms = max(50.0, self.current_delay_ms)
        elif self.thermal_pressure == "Critical":
            self.current_batch_scale = 0.1
            self.current_rank_scale = 0.1
            self.current_delay_ms = 1000.0

    def _monitor_loop(self):
        # We use a simulated sub-process or actual powermetrics
        # Note: `sudo powermetrics` requires passwordless sudo or running the main script with sudo
        cmd = ["sudo", "-n", "powermetrics", "-i", "1000", "-s", "gpu_power,thermal"]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Regex for GPU idle/active residency
            # Typical line: "GPU idle residency:   15.22%" -> active is 100 - 15.22
            # Or "GPU active residency:  84.78%" depending on macOS version
            gpu_regex = re.compile(r"GPU (?:idle|active) residency:\s+([0-9.]+)%")
            thermal_regex = re.compile(r"SMC output thermal level:\s+(.*)")
            
            while self.running:
                line = process.stdout.readline()
                if not line:
                    break
                    
                match_gpu = gpu_regex.search(line)
                if match_gpu:
                    val = float(match_gpu.group(1))
                    if "idle" in line.lower():
                        self.current_gpu_residency = (100.0 - val) / 100.0
                    else:
                        self.current_gpu_residency = val / 100.0
                        
                    self.update_pid(self.current_gpu_residency)
                    logger.debug(f"PID Update: Residency={self.current_gpu_residency:.2f}, RankScale={self.current_rank_scale:.2f}, Delay={self.current_delay_ms:.0f}ms")
                    
                match_therm = thermal_regex.search(line)
                if match_therm:
                    # SMC pressure levels usually nominal, fair, serious, critical
                    self_thermal = match_therm.group(1).strip()
                    if "nominal" in self_thermal.lower():
                        self.thermal_pressure = "Nominal"
                    elif "fair" in self_thermal.lower() or "moderate" in self_thermal.lower():
                        self.thermal_pressure = "Fair"
                    elif "serious" in self_thermal.lower() or "heavy" in self_thermal.lower():
                        self.thermal_pressure = "Serious"
                    elif "critical" in self_thermal.lower() or "trapping" in self_thermal.lower():
                        self.thermal_pressure = "Critical"

        except Exception as e:
            logger.error(f"Failed to start powermetrics monitoring. Proceeding with dummy loop. {e}")
            self._dummy_loop()
        finally:
            if 'process' in locals() and process:
                process.terminate()

    def _dummy_loop(self):
        """Fallback loop if sudo powermetrics is not available."""
        while self.running:
            # Simulate a 90% load
            self.current_gpu_residency = 0.90
            self.update_pid(self.current_gpu_residency)
            time.sleep(1)

    def get_control_parameters(self):
        """Returns (rank_scale_factor, micro_batch_scale_factor, delay_seconds)"""
        return self.current_rank_scale, self.current_batch_scale, self.current_delay_ms / 1000.0
