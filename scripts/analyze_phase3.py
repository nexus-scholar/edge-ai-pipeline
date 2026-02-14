import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import json
import numpy as np
from pathlib import Path

# Configuration
RUNS_DIR = "runs/phase3_wgisd"
OUTPUT_DIR = "runs/reports/phase3_analysis"
METRICS_OF_INTEREST = ["map50_proxy_test", "precision50_test", "recall50_test", "loss"]

def load_data():
    all_data = []
    # Find all metrics.csv files in phase3 runs
    files = glob.glob(os.path.join(RUNS_DIR, "**", "metrics.csv"), recursive=True)
    
    if not files:
        print(f"No metrics.csv files found in {RUNS_DIR}")
        return pd.DataFrame()

    print(f"Found {len(files)} metrics files.")

    for f in files:
        try:
            path = Path(f)
            seed_dir = path.parent
            exp_dir = seed_dir.parent
            
            exp_name = exp_dir.name
            seed = int(seed_dir.name) if seed_dir.name.isdigit() else 0
            
            df = pd.read_csv(f)
            
            if 'round_index' in df.columns and 'metric' in df.columns and 'value' in df.columns:
                df_filtered = df[df['metric'].isin(METRICS_OF_INTEREST)]
                df_dedup = df_filtered.drop_duplicates(subset=['round_index', 'metric'], keep='last')
                
                # Pivot only numeric values
                df_pivot = df_dedup.pivot(index='round_index', columns='metric', values='value').reset_index()
                df_pivot['experiment'] = exp_name
                df_pivot['seed'] = seed
                all_data.append(df_pivot)
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

def plot_metric(df, metric_name, output_path):
    plt.figure(figsize=(12, 7))
    
    experiments = sorted(df['experiment'].unique())
    # Define colors and markers for visual distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    markers = ['o', 's', '^', 'D', 'x', 'v', '*', 'p']
    
    for i, exp in enumerate(experiments):
        subset = df[df['experiment'] == exp]
        
        # Group only numeric columns for mean/std
        numeric_cols = [metric_name]
        grouped = subset.groupby('round_index')[numeric_cols].agg(['mean', 'std'])
        
        rounds = grouped.index
        means = grouped[(metric_name, 'mean')]
        stds = grouped[(metric_name, 'std')].fillna(0)
        
        # Clean label for plot
        label = exp.replace('phase3_wgisd_', '').replace('_adapted', '')
        
        # Plot mean
        plt.plot(rounds, means, label=label, marker=markers[i % len(markers)], color=colors[i], linewidth=2)
        
        # Fill standard deviation
        plt.fill_between(rounds, means - stds, means + stds, alpha=0.1, color=colors[i])

    plt.title(f"Benchmark: {metric_name} over AL Rounds", fontsize=14)
    plt.xlabel("AL Round (20 samples per round)", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Phase 3 data...")
    df = load_data()
    
    if df.empty:
        print("No data found to analyze.")
        return

    print(f"Loaded data for experiments: {df['experiment'].unique()}")
    
    for metric in METRICS_OF_INTEREST:
        if metric in df.columns:
            output_file = os.path.join(OUTPUT_DIR, f"{metric}.png")
            print(f"Plotting {metric} to {output_file}...")
            plot_metric(df, metric, output_file)
    
    # Save a summary CSV (grouping experiment and round)
    # Ensure only numeric columns are averaged
    summary = df.groupby(['experiment', 'round_index'])[METRICS_OF_INTEREST].agg(['mean', 'std'])
    summary_path = os.path.join(OUTPUT_DIR, "phase3_summary.csv")
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
