import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from pathlib import Path

# Configuration
RUNS_DIR = "runs/kaggle_bundle_extracted_20260213_131207/runs"
OUTPUT_DIR = "runs/reports/week1_analysis"
METRICS_OF_INTEREST = ["test_accuracy", "test_accuracy_blur", "test_accuracy_low_contrast"]
STRATEGY_MAP = {
    "cdgp_week1_batch_cifar10_random": "Random",
    "cdgp_week1_batch_cifar10_entropy": "Entropy",
    "cdgp_week1_batch_cifar10_domain_guided_calibrated_w02": "Domain (w=0.2)",
    "cdgp_week1_batch_cifar10_domain_guided_calibrated_w05": "Domain (w=0.5)",
    "cdgp_week1_batch_cifar10_domain_guided_calibrated_w08": "Domain (w=0.8)"
}

def load_data(runs_dir):
    all_data = []
    # Find all metrics.csv files
    files = glob.glob(os.path.join(runs_dir, "**", "metrics.csv"), recursive=True)
    
    for f in files:
        try:
            # Extract experiment name (parent of parent of file, usually)
            # Structure: runs/EXP_NAME/TIMESTAMP_SEED/metrics.csv
            path_parts = Path(f).parts
            # exp_name is the folder under 'runs'
            # We need to find which part corresponds to the experiment name
            # based on the STRATEGY_MAP keys
            
            exp_name = None
            for part in path_parts:
                if part in STRATEGY_MAP:
                    exp_name = part
                    break
            
            if not exp_name:
                continue

            df = pd.read_csv(f)
            # Filter for train split (where AL metrics are usually logged per round)
            # Actually, the metrics are logged with split='train' but represent test performance at that round
            df = df[df['split'] == 'train'] 
            
            # Pivot to get metrics as columns
            df_pivot = df.pivot(index='round_index', columns='metric', values='value').reset_index()
            df_pivot['experiment'] = exp_name
            df_pivot['strategy'] = STRATEGY_MAP[exp_name]
            
            # Extract seed from path or file (it's in the file)
            seed = df['seed'].iloc[0] if 'seed' in df.columns else 0
            df_pivot['seed'] = seed
            
            all_data.append(df_pivot)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

def plot_metric(df, metric_name, output_path):
    plt.figure(figsize=(10, 6))
    
    strategies = df['strategy'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for i, strategy in enumerate(sorted(strategies)):
        subset = df[df['strategy'] == strategy]
        
        # Group by round_index to get mean and std
        grouped = subset.groupby('round_index')[metric_name].agg(['mean', 'count', 'std'])
        
        # Calculate 95% CI
        ci95 = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
        
        plt.plot(grouped.index, grouped['mean'], label=strategy, marker='o', color=colors[i])
        plt.fill_between(grouped.index, grouped['mean'] - ci95, grouped['mean'] + ci95, alpha=0.2, color=colors[i])

    plt.title(f"Active Learning Performance: {metric_name}")
    plt.xlabel("AL Round")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_report_table(df):
    # Get final round performance
    max_round = df['round_index'].max()
    final_df = df[df['round_index'] == max_round]
    
    summary = final_df.groupby('strategy')[METRICS_OF_INTEREST].agg(['mean', 'std'])
    return summary

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    df = load_data(RUNS_DIR)
    
    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} rows of data.")
    print("Strategies found:", df['strategy'].unique())
    print("Seeds per strategy:", df.groupby('strategy')['seed'].nunique())

    print("Generating plots...")
    for metric in METRICS_OF_INTEREST:
        if metric in df.columns:
            output_file = os.path.join(OUTPUT_DIR, f"{metric}.png")
            plot_metric(df, metric, output_file)
            print(f"Saved {output_file}")
        else:
            print(f"Metric {metric} not found in data.")

    print("--- Final Round Summary (Mean ± Std) ---")
    summary = generate_report_table(df)
    print(summary.to_markdown())
    
    # Save summary to CSV
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"))
    print(f"Summary saved to {os.path.join(OUTPUT_DIR, 'summary_metrics.csv')}")

if __name__ == "__main__":
    main()
