import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_academic_plot():
    csv_path = "reports/hlra_validation.csv"
    output_path = "reports/hlra_loss_curve.png"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)

    # Set up the academic plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # We apply a small rolling average to smooth out the micro-batch noise, typical in academic loss curves
    window_size = 10
    
    # LoRA (Baseline) - Blue
    ax.plot(df['Step'], df['LoRA_Loss'].rolling(window=window_size, min_periods=1).mean(), 
            label='Standard LoRA (Baseline)', color='#1f77b4', linewidth=2, linestyle='--')
            
    # DoRA - Orange
    ax.plot(df['Step'], df['DoRA_Loss'].rolling(window=window_size, min_periods=1).mean(), 
            label='Standard DoRA', color='#ff7f0e', linewidth=2, linestyle='-.')
            
    # HLRA (Ours) - Red (Solid to stand out)
    ax.plot(df['Step'], df['HLRA_Loss'].rolling(window=window_size, min_periods=1).mean(), 
            label='Dual-Path HLRA (Ours)', color='#d62728', linewidth=2.5, linestyle='-')

    # Formatting the axes and titles
    ax.set_title("Cross-Entropy Loss vs. Training Steps (0.5B Drafter on M3 8GB)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Training Step", fontsize=14, labelpad=10)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=14, labelpad=10)
    
    # Force axis limits starting slightly below the minimum smoothed loss
    ax.set_xlim(0, 200)
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, borderpad=1)
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Tight layout ensures labels aren't cut off
    plt.tight_layout()

    # Save artifact
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Successfully generated high-res academic plot: {output_path}")

if __name__ == "__main__":
    generate_academic_plot()
