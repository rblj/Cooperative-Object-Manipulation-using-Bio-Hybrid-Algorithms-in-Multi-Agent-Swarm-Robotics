import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_thesis_dashboard(file_path='final_thesis_results.csv'):
    # 1. DATA LOADING
    if not pd.io.common.file_exists(file_path):
        print("Waiting for data...")
        return
    
    data = pd.read_csv(file_path).ffill().dropna()
    
    # 2. GLOBAL STYLING
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    
    # Initialize figure with constrained_layout to prevent any overlaps
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    
    # Create a GridSpec with better vertical ratios
    # Top row: 25%, Middle row: 25%, Bottom row: 35%
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])

    # --- GRAPH A: SUCCESS RATE ---
    ax1 = fig.add_subplot(grid[0, 0:2])
    sns.lineplot(data=data, x='step', y='success', ax=ax1, color='teal', alpha=0.3, label='Raw')
    ax1.plot(data['step'], data['success'].rolling(50).mean(), color='darkred', linewidth=2, label='SMA 50')
    ax1.set_title("A. Learning Convergence (Success)", fontweight='bold')
    ax1.set_ylabel("Probability")
    ax1.legend(loc='upper right', fontsize='small')

    # --- GRAPH B: BEE RECRUITMENT ---
    ax2 = fig.add_subplot(grid[0, 2])
    sns.kdeplot(data=data['avg_active'], fill=True, color="orange", ax=ax2)
    ax2.set_title("B. Bee Recruitment Dist.", fontweight='bold')
    ax2.set_xlabel("Avg Active Agents")

    # --- GRAPH C: PSO COHESION ---
    ax3 = fig.add_subplot(grid[1, 0:2])
    sns.lineplot(data=data, x='step', y='cohesion', ax=ax3, color='navy', alpha=0.2)
    ax3.plot(data['step'], data['cohesion'].rolling(50).mean(), color='navy', linewidth=2)
    ax3.set_title("C. Swarm Cohesion Stability (IADV)", fontweight='bold')
    ax3.set_ylabel("Variance (m²)")

    # --- GRAPH D: ACO EFFICIENCY ---
    ax4 = fig.add_subplot(grid[1, 2])
    # hue assigned to avg_active to fix the warning and show cooperation depth
    sns.scatterplot(data=data, x='avg_active', y='min_dist', hue='avg_active', 
                    palette='viridis', alpha=0.6, ax=ax4, legend=False)
    ax4.set_title("D. Cooperation vs. Efficiency", fontweight='bold')
    ax4.set_xlabel("Active Agents")
    ax4.set_ylabel("Dist to Goal")

    # --- GRAPH E: TCT BOXPLOT ---
    ax5 = fig.add_subplot(grid[2, :])
    # Binning logic
    data['bin'] = pd.cut(data['step'], bins=5).apply(lambda x: f"{int(x.left/1000)}k-{int(x.right/1000)}k")
    # Fixed FutureWarning by assigning hue='bin'
    sns.boxplot(data=data, x='bin', y='time', hue='bin', palette="coolwarm", ax=ax5, legend=False)
    ax5.set_title("E. Task Completion Time (TCT) over Training Phases", fontweight='bold')
    ax5.set_xlabel("Training Progress (Steps)")
    ax5.set_ylabel("Seconds")

    # 3. TITLING & POLISH
    # We use a secondary title layout to ensure it sits ABOVE the constrained layout
    fig.suptitle("Bio-Hybrid MARL Evaluation: Multi-Agent Cooperative Transport\n(ACO Pathing + PSO Cohesion + Bee Recruitment Integration)", 
                 fontsize=18, fontweight='bold', fontfamily='serif')

    # Save and show
    plt.savefig('Cleaned_Thesis_Dashboard.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_thesis_dashboard()