import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def read_history_csv(file_path, output_path):
    # Load data from CSV file
    data = pd.read_csv(file_path)

    # Filter the metrics to exclude 'mean_io_u'
    metrics = [col for col in data.columns if 'mean_io_u' not in col and not col.startswith('val_')]

    # Creating the ploting environement 
    plt.figure(figsize=(16, 12))
    num_metrics = len(metrics)

    # Plot metrics and their matching validation 
    for i, metric in enumerate(metrics):
        plt.subplot(num_metrics, 1, i + 1)
        
        plt.plot(data[metric], label=f'Train {metric}')
        plt.plot(data[f'val_{metric}'], label=f'Val {metric}')
        
        plt.title(f'{metric.capitalize()} vs Validation {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()

    # Save png image
    plt.savefig(output_path)
    plt.close() 

def read_all_csv_folds(folder_path):
    # Gather files beging by fold
    fold_paths = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.startswith('fold')
    ]
    
    for i, fold_path in enumerate(fold_paths):
        # Define output paths
        output_path = os.path.join(folder_path, f'fold_{i+1}_metrics.png')
        read_history_csv(fold_path, output_path)

if __name__ == '__main__':
    # Creating arguments parser
    parser = argparse.ArgumentParser(description="Process CSV files in a folder and save plots.")

    # Add arguments needed
    parser.add_argument('folder_path', help="Path to the folder containing CSV files for each fold.")

    # Parse the arguments
    args = parser.parse_args()

    # Call read function 
    read_all_csv_folds(args.folder_path)
