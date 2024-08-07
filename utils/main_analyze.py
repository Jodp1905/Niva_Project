import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import logging
import re
from pathlib import Path

# Configure logging
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

# Paths parameters
NIVA_PROJECT_DATA_ROOT = os.getenv('NIVA_PROJECT_DATA_ROOT')
MODEL_FOLDER = Path(f'{NIVA_PROJECT_DATA_ROOT}/model/')

def extract_fold_number(fold_path):
    # Extract the fold number from the folder name
    match = re.search(r'fold-(\d+)', os.path.basename(fold_path))
    return match.group(1) if match else 'unknown'

def plot_epoch_data(epoch_data_path, output_path, fold_number):
    # Load data from epoch_data.csv
    data = pd.read_csv(epoch_data_path)
    
    # Selecting metrics to plot
    metrics = [col for col in data.columns if col not in ['epoch', 'epoch_duration', 'rss_memory_gb', 'vms_memory_gb'] and not col.startswith('val_')]

    # Creating the plotting environment
    plt.figure(figsize=(16, 12))
    num_metrics = len(metrics)

    # Plot metrics and their matching validation 
    for i, metric in enumerate(metrics):
        plt.subplot(num_metrics, 1, i + 1)
        
        plt.plot(data['epoch'], data[metric], label=f'Train {metric}')
        plt.plot(data['epoch'], data[f'val_{metric}'], label=f'Val {metric}')
        
        plt.title(f'Fold {fold_number}: {metric.capitalize()} vs Validation {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(output_path)
    plt.close()

def plot_batch_data(batch_data_path, output_path, fold_number):
    # Load data from batch_data.csv
    data = pd.read_csv(batch_data_path)
    
    # Selecting metrics to plot
    metrics = [col for col in data.columns if col not in ['batch', 'batch_duration']]

    # Creating the plotting environment
    plt.figure(figsize=(16, 12))
    num_metrics = len(metrics)

    # Plot metrics
    for i, metric in enumerate(metrics):
        plt.subplot(num_metrics, 1, i + 1)
        
        plt.plot(data['batch'], data[metric], label=f'{metric}')
        
        plt.title(f'Fold {fold_number}: {metric.capitalize()}')
        plt.xlabel('Batch')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(output_path)
    plt.close()

def save_ploted_training_data(folder_path):
    # Define the path for images analysis
    images_folder = os.path.join(folder_path, 'images_analysis')
    os.makedirs(images_folder, exist_ok=True)

    # Gather all fold paths
    fold_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('fold')]
    
    for fold_path in fold_paths:
        fold_number = extract_fold_number(fold_path)
        
        # Define paths for epoch and batch data
        epoch_data_path = os.path.join(fold_path, 'epoch_data.csv')
        batch_data_path = os.path.join(fold_path, 'batch_data.csv')
        
        # Define output paths for the plots
        epoch_output_path = os.path.join(images_folder, f'fold_{fold_number}_epoch_metrics.png')
        batch_output_path = os.path.join(images_folder, f'fold_{fold_number}_batch_metrics.png')
        
        # Plot and save epoch data
        plot_epoch_data(epoch_data_path, epoch_output_path, fold_number)
        
        # Plot and save batch data
        plot_batch_data(batch_data_path, batch_output_path, fold_number)

def get_training_duration(file_path):
    # Read the duration from file
    with open(file_path, 'r') as file:
        duration_seconds = float(file.read().strip())
    
    duration_seconds = int(duration_seconds)
    
    # Convert to hours, minutes, seconds
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{hours}h {minutes}m {seconds}s"

def get_hyperparameters(file_path):
    # Read hyperparameters from JSON file
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters

def get_model_config(file_path):
    # Read model configuration from JSON file
    with open(file_path, 'r') as file:
        model_config = json.load(file)
    return model_config

def save_all_information_to_txt(folder_path, output_txt_path):
    with open(output_txt_path, 'w') as txt_file:
        # Training duration
        duration_file = os.path.join(folder_path, 'duration_seconds.txt')
        duration = get_training_duration(duration_file)
        txt_file.write(f"Training Duration:\n{duration}\n\n")
        
        # Hyperparameters
        hyperparameters_file = os.path.join(folder_path, 'hyperparameters.json')
        hyperparameters = get_hyperparameters(hyperparameters_file)
        txt_file.write(f"Hyperparameters:\n{json.dumps(hyperparameters, indent=4)}\n\n")
        
        # Model configuration
        model_cfg_file = os.path.join(folder_path, 'model_cfg.json')
        model_config = get_model_config(model_cfg_file)
        txt_file.write(f"Model Configuration:\n{json.dumps(model_config, indent=4)}\n\n")
        
        # Information from the "average" model
        avg_model_folder = os.path.join(folder_path, 'resunet-a_avg_2024-08-06-05-11-34+02-00')
        if os.path.exists(avg_model_folder):
            evaluation_avg_file = os.path.join(avg_model_folder, 'evaluation_avg.json')
            if os.path.exists(evaluation_avg_file):
                with open(evaluation_avg_file, 'r') as file:
                    avg_evaluation_data = json.load(file)
                txt_file.write(f"Average Model Evaluation Data:\n{json.dumps(avg_evaluation_data, indent=4)}\n\n")

if __name__ == '__main__':
    # Creating arguments parser
    parser = argparse.ArgumentParser(description="Process CSV, txt and json files in a folder, save plots, and generate a summary report.")

    # Add arguments needed
    parser.add_argument('folder_name', help="Path to the folder containing CSV files for each fold.")

    # Parse the arguments
    args = parser.parse_args()

    # Generate the output text file name based on the last folder name
    folder_name = args.folder_name
    folder_path = os.path.join(MODEL_FOLDER, folder_name)
    output_txt_path = os.path.join(folder_path, f"{folder_name}_experience_analyze.txt")

    # Call read function 
    save_ploted_training_data(folder_path)

    # Save all information to a text file
    save_all_information_to_txt(folder_path, output_txt_path)

    logging.info(f"Summary report saved to {output_txt_path}")
