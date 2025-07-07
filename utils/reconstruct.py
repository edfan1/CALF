import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # Import from sklearn

# Add the parent directory to path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.metrics import MAE, MSE, MAPE, SMAPE, ND

def unscale_and_calculate_metrics(result_folder, data_folder, dataset_file):
    """
    Load .npy files, unscale them using the original dataset, and calculate metrics
    
    Args:
        result_folder: Path to folder containing pred.npy and true.npy
        data_folder: Path to folder containing original dataset
        dataset_file: CSV filename of the original dataset
    """
    # Load prediction and ground truth data
    pred_path = os.path.join(result_folder, 'pred.npy')
    true_path = os.path.join(result_folder, 'true.npy')
    
    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        print(f"Error: Prediction files not found in {result_folder}")
        return
    
    pred = np.load(pred_path)
    true = np.load(true_path)
    
    print(f"Loaded data - Pred shape: {pred.shape}, True shape: {true.shape}")
    
    # Load the original dataset to get scaling parameters
    dataset_path = os.path.join(data_folder, dataset_file)
    if not os.path.exists(dataset_path):
        print(f"Error: Original dataset not found at {dataset_path}")
        return
    
    # Load and prepare original data for scaling
    df_raw = pd.read_csv(dataset_path)
    
    # Extract numeric columns (skip date column if present)
    if not np.issubdtype(df_raw.iloc[:, 0].dtype, np.number):
        numeric_data = df_raw.iloc[:, 1:].values
        column_names = df_raw.columns[1:]
    else:
        numeric_data = df_raw.values
        column_names = df_raw.columns
    
    print(f"Original dataset features: {numeric_data.shape[1]}")
    print(f"Prediction features: {pred.shape[-1]}")
    
    # Handle feature dimension mismatch
    if pred.shape[-1] != numeric_data.shape[1]:
        print(f"WARNING: Feature dimension mismatch. Dataset has {numeric_data.shape[1]} features but predictions have {pred.shape[-1]} features.")
        
        # Take the appropriate number of features from the dataset
        if pred.shape[-1] < numeric_data.shape[1]:
            print(f"Using first {pred.shape[-1]} features from dataset")
            numeric_data = numeric_data[:, :pred.shape[-1]]
        else:
            print(f"ERROR: Predictions have more features than dataset. Using available features and padding.")
            # This is a more complex case - let's handle it by padding
            padding_needed = pred.shape[-1] - numeric_data.shape[1]
            # Pad the numeric_data with zeros - this is just for fitting the scaler
            numeric_data_padded = np.pad(numeric_data, ((0, 0), (0, padding_needed)))
            numeric_data = numeric_data_padded
    
    # Create and fit scaler
    print("\nTo correctly rescale predictions, we need information about the original training split:")
    train_start_idx = input("Enter starting index of training data (default=0): ")
    train_start_idx = int(train_start_idx) if train_start_idx.strip() else 0

    train_end_idx = input("Enter ending index of training data: ")
    if not train_end_idx.strip():
        print("Error: Need the ending index of training data for accurate scaling")
        train_end_idx = len(df_raw) - 1  # Default to using all data
    else:
        train_end_idx = int(train_end_idx)

    print(f"Using data from index {train_start_idx} to {train_end_idx} for scaler fitting")

    # Extract training portion for correct scaling
    if not np.issubdtype(df_raw.iloc[:, 0].dtype, np.number):
        train_data = df_raw.iloc[train_start_idx:train_end_idx+1, 1:].values
        column_names = df_raw.columns[1:]
    else:
        train_data = df_raw.iloc[train_start_idx:train_end_idx+1].values
        column_names = df_raw.columns

    print(f"Training data shape for scaler: {train_data.shape}")

    # Handle feature dimension mismatch if needed
    if pred.shape[-1] != train_data.shape[1]:
        print(f"WARNING: Feature dimension mismatch. Training data has {train_data.shape[1]} features but predictions have {pred.shape[-1]} features.")
        if pred.shape[-1] < train_data.shape[1]:
            print(f"Using first {pred.shape[-1]} features from training data")
            train_data = train_data[:, :pred.shape[-1]]
        else:
            print(f"ERROR: Predictions have more features than training data.")
            # Handle appropriately

    # Create and fit scaler ONLY on the training portion
    scaler = StandardScaler()
    scaler.fit(train_data)
    print("StandardScaler fit to training data only - this matches the original scaling")
    
    # Reshape data if needed to match dimensions for inverse_transform
    original_pred_shape = pred.shape
    original_true_shape = true.shape
    
    # Handle different dimension structures
    if len(pred.shape) > 2:
        # Multi-dimensional case: reshape to 2D for inverse transform
        pred_reshaped = pred.reshape(-1, pred.shape[-1])
        true_reshaped = true.reshape(-1, true.shape[-1])
    else:
        # Already 2D
        pred_reshaped = pred
        true_reshaped = true
    
    # Apply inverse transform
    print("Unscaling data with sklearn's StandardScaler...")
    pred_unscaled = scaler.inverse_transform(pred_reshaped)
    true_unscaled = scaler.inverse_transform(true_reshaped)
    
    # Reshape back to original dimensions if needed
    if len(original_pred_shape) > 2:
        pred_unscaled = pred_unscaled.reshape(original_pred_shape)
        true_unscaled = true_unscaled.reshape(original_true_shape)
    
    # Ask if user wants to analyze a specific feature
    analyze_specific = input("\nDo you want to analyze a specific feature? (y/n): ").lower() == 'y'
    
    if analyze_specific and len(original_pred_shape) > 2:
        # Show available features
        print("\nAvailable features:")
        for i, name in enumerate(column_names[:pred_unscaled.shape[2]]):
            print(f"{i}: {name}")
        
        # Get feature selection
        while True:
            try:
                feature_idx = int(input("\nSelect feature to analyze (0-" + str(pred_unscaled.shape[2]-1) + "): "))
                if 0 <= feature_idx < pred_unscaled.shape[2]:
                    break
                else:
                    print(f"Error: Please enter a number between 0 and {pred_unscaled.shape[2]-1}")
            except ValueError:
                print("Please enter a valid number")
        
        # Get feature name
        feature_name = column_names[feature_idx] if feature_idx < len(column_names) else f"Feature {feature_idx}"
        print(f"\nAnalyzing feature: {feature_name}")
        
        # Extract single feature data (shape becomes [samples, time_steps])
        pred_feature = pred_unscaled[:, :, feature_idx]
        true_feature = true_unscaled[:, :, feature_idx]
        
        # Calculate metrics for this specific feature
        print(f"\n===== Metrics for {feature_name} =====")
        mae = MAE(pred_feature, true_feature)
        mse = MSE(pred_feature, true_feature)
        mape = MAPE(pred_feature, true_feature)
        smape = SMAPE(pred_feature, true_feature)
        nd = ND(pred_feature, true_feature)
        
        print(f"MAE:   {mae:.6f}")
        print(f"MSE:   {mse:.6f}")
        print(f"MAPE:  {mape:.6f}%")
        print(f"SMAPE: {smape:.6f}%")
        print(f"ND:    {nd:.6f}")
        
        # Save feature-specific metrics
        feature_metrics = {
            'feature': feature_idx,
            'feature_name': feature_name,
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape,
            'SMAPE': smape,
            'ND': nd
        }
        # np.save(os.path.join(result_folder, f'metrics_feature_{feature_idx}.npy'), feature_metrics)
        
        # Generate feature-specific visualizations
        # generate_feature_visualizations(pred_feature, true_feature, feature_name, result_folder)
        
        return feature_metrics
    else:
        # Calculate metrics on all features (existing code)
        print("\n===== Metrics on All Features (Unscaled Data) =====")
        mae = MAE(pred_unscaled, true_unscaled)
        mse = MSE(pred_unscaled, true_unscaled)
        mape = MAPE(pred_unscaled, true_unscaled)
        smape = SMAPE(pred_unscaled, true_unscaled)
        nd = ND(pred_unscaled, true_unscaled)
        
        print(f"MAE:   {mae:.6f}")
        print(f"MSE:   {mse:.6f}")
        print(f"MAPE:  {mape:.6f}%")
        print(f"SMAPE: {smape:.6f}%")
        print(f"ND:    {nd:.6f}")
        
        # Save metrics
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape,
            'SMAPE': smape,
            'ND': nd
        }
        # np.save(os.path.join(result_folder, 'unscaled_metrics.npy'), metrics)
        
        # Save unscaled data
        # np.save(os.path.join(result_folder, 'pred_unscaled.npy'), pred_unscaled)
        # np.save(os.path.join(result_folder, 'true_unscaled.npy'), true_unscaled)
        
        # Generate sample visualization
        # generate_visualizations(pred_unscaled, true_unscaled, result_folder)
        
        return metrics

def generate_visualizations(pred, true, save_dir, num_samples=3):
    """Generate visualizations comparing predictions and ground truth"""
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    
    # Determine the number of samples and features
    if len(pred.shape) == 3:  # [samples, time_steps, features]
        num_series = min(pred.shape[0], num_samples)
        for i in range(num_series):
            for feat_idx in range(min(3, pred.shape[2])):  # Plot up to 3 features
                plt.figure(figsize=(12, 6))
                plt.plot(true[i, :, feat_idx], label='Ground Truth', linewidth=2)
                plt.plot(pred[i, :, feat_idx], label='Prediction', linewidth=2, linestyle='--')
                plt.title(f'Series {i}, Feature {feat_idx} (Unscaled)', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'plots', f'sample_{i}_feature_{feat_idx}.png'))
                plt.close()
    else:  # [time_steps, features]
        for feat_idx in range(min(3, pred.shape[1])):  # Plot up to 3 features
            plt.figure(figsize=(12, 6))
            plt.plot(true[:, feat_idx], label='Ground Truth', linewidth=2)
            plt.plot(pred[:, feat_idx], label='Prediction', linewidth=2, linestyle='--')
            plt.title(f'Feature {feat_idx} (Unscaled)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'plots', f'feature_{feat_idx}.png'))
            plt.close()

# Add this new function for feature-specific visualization
def generate_feature_visualizations(pred_feature, true_feature, feature_name, save_dir, num_samples=5):
    """Generate visualizations for a specific feature"""
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    
    # Plot average prediction across all samples
    plt.figure(figsize=(14, 7))
    
    # Calculate mean and std across all samples
    true_mean = np.mean(pred_feature, axis=0)
    pred_mean = np.mean(true_feature, axis=0)
    true_std = np.std(pred_feature, axis=0)
    pred_std = np.std(true_feature, axis=0)
    
    # Plot mean with shaded std dev
    x = np.arange(len(true_mean))
    plt.plot(x, true_mean, 'b-', linewidth=2, label='True (mean)')
    plt.fill_between(x, true_mean - true_std, true_mean + true_std, color='b', alpha=0.2, label='True (±1σ)')
    
    plt.plot(x, pred_mean, 'r--', linewidth=2, label='Prediction (mean)')
    plt.fill_between(x, pred_mean - pred_std, pred_mean + pred_std, color='r', alpha=0.2, label='Prediction (±1σ)')
    
    plt.title(f"Average {feature_name} Prediction vs Ground Truth", fontsize=14)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'plots', f'{feature_name.replace(" ", "_")}_average.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved average plot to {save_path}")
    
    # Individual sample plots
    indices = np.random.choice(pred_feature.shape[0], min(num_samples, pred_feature.shape[0]), replace=False)
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(12, 6))
        plt.plot(true_feature[idx], 'b-', linewidth=2, label='Ground Truth')
        plt.plot(pred_feature[idx], 'r--', linewidth=2, label='Prediction')
        plt.title(f"{feature_name} - Sample #{idx}", fontsize=14)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'plots', f'{feature_name.replace(" ", "_")}_sample_{idx}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved sample plot to {save_path}")

if __name__ == "__main__":
    # Determine base project directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils folder
    project_root = os.path.dirname(current_dir)  # CALF root folder
    
    # List of possible locations for results directories
    potential_result_dirs = [
        os.path.join(project_root, 'results'),
        os.path.join(project_root, 'checkpoints'),
        project_root,  # Also check the root directory itself
    ]
    
    result_dirs = []
    
    # Search for result directories in each potential location
    for base_dir in potential_result_dirs:
        if os.path.exists(base_dir):
            print(f"Searching in {base_dir}...")
            for root, dirs, files in os.walk(base_dir):
                if 'pred.npy' in files and 'true.npy' in files:
                    result_dirs.append(root)
    
    if result_dirs:
        print(f"\nFound {len(result_dirs)} result directories:")
        for i, d in enumerate(result_dirs):
            # Get relative path for cleaner display
            rel_path = os.path.relpath(d, project_root)
            print(f"{i+1}. {rel_path}")
            
        choice = input("\nEnter number to analyze (or press Enter for manual input): ")
        
        try:
            idx = int(choice) - 1
            result_folder = result_dirs[idx]
        except (ValueError, IndexError):
            result_folder = input("Enter path to result folder containing pred.npy and true.npy: ")
            # Convert relative path to absolute if needed
            if not os.path.isabs(result_folder):
                result_folder = os.path.join(project_root, result_folder)
    else:
        print("No result directories found automatically.")
        print("Tip: Result directories should contain both 'pred.npy' and 'true.npy' files.")
        result_folder = input("Enter path to result folder containing pred.npy and true.npy: ")
        # Convert relative path to absolute if needed
        if not os.path.isabs(result_folder):
            result_folder = os.path.join(project_root, result_folder)
    
    # Similar improvements for dataset paths
    data_folder = input("Enter path to folder containing original dataset: ")
    if not os.path.isabs(data_folder):
        data_folder = os.path.join(project_root, data_folder)
        
    dataset_file = input("Enter original dataset filename: ")
    
    # Run the analysis
    unscale_and_calculate_metrics(result_folder, data_folder, dataset_file)