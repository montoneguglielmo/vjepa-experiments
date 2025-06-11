#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss(experiment_folder):
    # Convert to Path object for easier path handling
    exp_path = Path(experiment_folder).resolve()
    
    if not exp_path.exists():
        print(f"Error: Experiment folder {experiment_folder} does not exist")
        return
    
    # Find all CSV files in the experiment folder
    csv_files = list(exp_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {exp_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {exp_path}")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Group by epoch and calculate mean loss
            epoch_loss = df.groupby('epoch')['loss'].mean()
            
            # Plot the data
            plt.plot(epoch_loss.index, epoch_loss.values, 
                    label=f'Run {csv_file.stem.split("_r")[-1]}',
                    marker='o', markersize=4)
            print(f"Processed {csv_file.name}")
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue
    
    # Customize plot
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot in the experiment folder
    output_path = exp_path / 'loss_plot.png'
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
    
    # Close the plot
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot average loss per epoch from experiment CSV files')
    parser.add_argument('experiment_folder', type=str, help='Path to the experiment folder containing CSV files')
    
    args = parser.parse_args()
    plot_loss(args.experiment_folder)

if __name__ == '__main__':
    main() 