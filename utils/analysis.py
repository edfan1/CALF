import numpy as np
import os
import glob

def display_metrics(file_path):
    """Display metrics from a .npy file with proper formatting"""
    try:
        # Load the metrics data
        metrics = np.load(file_path, allow_pickle=True)
        
        print(f"\n{'='*50}")
        print(f"Metrics from: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        
        # If metrics is a dictionary or has named fields
        if isinstance(metrics, np.ndarray) and metrics.dtype.names is not None:
            for name in metrics.dtype.names:
                print(f"{name}: {metrics[name]}")

        # If metrics is a simple array
        elif isinstance(metrics, np.ndarray):
            if metrics.ndim == 0:  # Single value
                print(f"Value: {metrics}")
            else:
                # Assumes standard CALF metrics order: MAE, MSE, RMSE, MAPE, MSPE
                metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'SMAPE']
                if len(metrics) == len(metric_names):
                    for name, value in zip(metric_names, metrics):
                        print(f"{name}: {value}")
                else:
                    print(f"Shape: {metrics.shape}")
                    print(f"Content: {metrics}")
        
        # If metrics is a dict
        elif isinstance(metrics, dict):
            for key, value in metrics.items():
                print(f"{key}: {value}")
        
        else:
            print(f"Content: {metrics}")
            
    except Exception as e:
        print(f"Error loading metrics file: {e}")

# Find and display all metrics files
metrics_files = glob.glob("./metrics*.npy")
if metrics_files:
    for file_path in metrics_files:
        display_metrics(file_path)
else:
    # Try looking in the results directory
    metrics_files = glob.glob("./results/metrics*.npy")
    if metrics_files:
        for file_path in metrics_files:
            display_metrics(file_path)
    else:
        # Allow user to input path
        print("No metrics files found automatically.")
        file_path = input("Enter path to metrics.npy file: ")
        if file_path:
            display_metrics(file_path)