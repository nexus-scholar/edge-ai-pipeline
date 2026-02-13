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
    # Structure: runs/phase3_wgisd/EXP_NAME/SEED/metrics.csv
    files = glob.glob(os.path.join(RUNS_DIR, "**", "metrics.csv"), recursive=True)
    
    if not files:
        print(f"No metrics.csv files found in {RUNS_DIR}")
        return pd.DataFrame()

    print(f"Found {len(files)} metrics files.")

    for f in files:
        try:
            # Parse path to get experiment info
            path = Path(f)
            # Example: runs/phase3_wgisd/phase3_wgisd_domain_guided_v1/42/metrics.csv
            # We assume the parent folder of the seed folder is the experiment name
            seed_dir = path.parent
            exp_dir = seed_dir.parent
            
            exp_name = exp_dir.name
            seed = int(seed_dir.name) if seed_dir.name.isdigit() else 0
            
            df = pd.read_csv(f)
            
            # Filter for rows that have the metrics we care about
            # In our runner, metrics are logged with round_index
            # We want to track these metrics over rounds.
            
            # The metrics.csv format from Edge AL pipeline usually has:
            # round_index, epoch, metric, value, ...
            
            # We want to group by round_index and extract the final value for each metric per round
            # Or if it's logged per epoch, we might want the last epoch of the round.
            
            # Let's inspect the columns. Usually: round_index, metric, value, phase/split
            
            # Pivot the table
            if 'round_index' in df.columns and 'metric' in df.columns and 'value' in df.columns:
                 # Filter for the metrics of interest
                df_filtered = df[df['metric'].isin(METRICS_OF_INTEREST)]
                
                # We might have multiple values per round (e.g. from multiple epochs). 
                # We typically want the last one (final epoch) or the one explicitly logged as round summary.
                # Assuming the logger logs the round summary at the end of the round.
                
                # Let's drop duplicates keeping the last one for each round/metric
                df_dedup = df_filtered.drop_duplicates(subset=['round_index', 'metric'], keep='last')
                
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
    plt.figure(figsize=(10, 6))
    
    experiments = df['experiment'].unique()
    
    for exp in experiments:
        subset = df[df['experiment'] == exp]
        
        # Group by round_index to calculate mean and std across seeds
        grouped = subset.groupby('round_index')[metric_name].agg(['mean', 'count', 'std'])
        
        rounds = grouped.index
        means = grouped['mean']
        stds = grouped['std'].fillna(0)
        
        # Plot mean
        plt.plot(rounds, means, label=f"{exp} (n={grouped['count'].max()})", marker='o')
        
        # Fill standard deviation (or SE)
        plt.fill_between(rounds, means - stds, means + stds, alpha=0.2)

    plt.title(f"Phase 3: {metric_name} over Active Learning Rounds")
    plt.xlabel("AL Round")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
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
    
    # Save a summary CSV
    summary = df.groupby(['experiment', 'round_index'])[METRICS_OF_INTEREST].agg(['mean', 'std'])
    summary_path = os.path.join(OUTPUT_DIR, "phase3_summary.csv")
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
