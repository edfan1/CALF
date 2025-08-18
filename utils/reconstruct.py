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
    # --- Part 1: Load and unscale the initial result ---
    print("--- Loading initial result ---")
    pred_path = os.path.join(result_folder, 'pred.npy')
    true_path = os.path.join(result_folder, 'true.npy')
    
    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        print(f"Error: Prediction files not found in {result_folder}")
        return
    
    pred = np.load(pred_path)
    true = np.load(true_path)
    
    print(f"Loaded data - Pred shape: {pred.shape}, True shape: {true.shape}")
    
    dataset_path = os.path.join(data_folder, dataset_file)
    if not os.path.exists(dataset_path):
        print(f"Error: Original dataset not found at {dataset_path}")
        return
    
    df_raw = pd.read_csv(dataset_path)
    
    print("\nTo correctly rescale predictions, we need the original training split indices.")
    train_start_idx = int(input("Enter starting index of training data (e.g., 0): "))
    train_end_idx = int(input("Enter ending index of training data: "))

    if not np.issubdtype(df_raw.iloc[:, 0].dtype, np.number):
        train_data = df_raw.iloc[train_start_idx:train_end_idx+1, 1:].values
    else:
        train_data = df_raw.iloc[train_start_idx:train_end_idx+1].values

    if pred.shape[-1] != train_data.shape[1]:
        print(f"Warning: Feature mismatch in {dataset_file}. Adjusting data columns to match prediction.")
        train_data = train_data[:, :pred.shape[-1]]

    scaler = StandardScaler().fit(train_data)
    
    pred_reshaped = pred.reshape(-1, pred.shape[-1]) if len(pred.shape) > 2 else pred
    pred_unscaled = scaler.inverse_transform(pred_reshaped).reshape(pred.shape)
    
    true_reshaped = true.reshape(-1, true.shape[-1]) if len(true.shape) > 2 else true
    true_unscaled = scaler.inverse_transform(true_reshaped).reshape(true.shape)

    # Store results in lists
    all_preds_unscaled = [pred_unscaled]
    all_trues_unscaled = [true_unscaled]
    
    # --- Part 2: Loop to append additional results ---
    while True:
        add_another = input("\nAppend another result for combined analysis? (y/n): ").lower()
        if add_another != 'y':
            break
        
        print("\n--- Appending new result ---")
        next_result_folder = input("Enter path to the NEXT result folder: ")
        next_data_folder = input("Enter path to its corresponding data folder: ")
        next_dataset_file = input("Enter its corresponding dataset filename: ")

        next_pred_path = os.path.join(next_result_folder, 'pred.npy')
        next_true_path = os.path.join(next_result_folder, 'true.npy')

        if not (os.path.exists(next_pred_path) and os.path.exists(next_true_path)):
            print(f"Error: Prediction files not found in {next_result_folder}. Skipping.")
            continue

        next_pred = np.load(next_pred_path)
        next_true = np.load(next_true_path)

        if next_pred.shape[:-1] != pred.shape[:-1]:
            print("Error: Incompatible shapes. Number of samples and time steps must match. Skipping.")
            continue

        next_dataset_path = os.path.join(next_data_folder, next_dataset_file)
        if not os.path.exists(next_dataset_path):
            print(f"Error: Original dataset not found at {next_dataset_path}. Skipping.")
            continue
        
        next_df_raw = pd.read_csv(next_dataset_path)
        
        if not np.issubdtype(next_df_raw.iloc[:, 0].dtype, np.number):
            next_train_data = next_df_raw.iloc[train_start_idx:train_end_idx+1, 1:].values
        else:
            next_train_data = next_df_raw.iloc[train_start_idx:train_end_idx+1].values

        if next_pred.shape[-1] != next_train_data.shape[1]:
            print(f"Warning: Feature mismatch in {next_dataset_file}. Adjusting data columns.")
            next_train_data = next_train_data[:, :next_pred.shape[-1]]

        next_scaler = StandardScaler().fit(next_train_data)

        next_pred_reshaped = next_pred.reshape(-1, next_pred.shape[-1]) if len(next_pred.shape) > 2 else next_pred
        next_pred_unscaled = next_scaler.inverse_transform(next_pred_reshaped).reshape(next_pred.shape)

        next_true_reshaped = next_true.reshape(-1, next_true.shape[-1]) if len(next_true.shape) > 2 else next_true
        next_true_unscaled = next_scaler.inverse_transform(next_true_reshaped).reshape(next_true.shape)

        all_preds_unscaled.append(next_pred_unscaled)
        all_trues_unscaled.append(next_true_unscaled)
        print(f"Successfully appended result from {next_result_folder}")

    # --- Part 3: Combine and Analyze ---
    print("\n" + "="*25 + " FINAL COMBINED ANALYSIS " + "="*25)
    
    # Concatenate along the feature axis
    combined_pred = np.concatenate(all_preds_unscaled, axis=-1)
    combined_true = np.concatenate(all_trues_unscaled, axis=-1)

    print(f"Total combined features: {combined_pred.shape[-1]}")
    print("\n----- Overall Metrics (All Columns Combined) -----")
    
    mae = MAE(combined_pred, combined_true)
    mse = MSE(combined_pred, combined_true)
    mape = MAPE(combined_pred, combined_true)
    smape = SMAPE(combined_pred, combined_true)
    nd = ND(combined_pred, combined_true)
    
    print(f"MAE:   {mae:.6f}")
    print(f"MSE:   {mse:.6f}")
    print(f"MAPE:  {mape:.6f}%")
    print(f"SMAPE: {smape:.6f}%")
    print(f"ND:    {nd:.6f}")

    metrics = {'MAE': mae, 'MSE': mse, 'MAPE': mape, 'SMAPE': smape, 'ND': nd}
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