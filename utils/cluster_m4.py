import os
import sys
import pandas as pd
import numpy as np
sys.path.append('.')  # Add the project root to the path
from utils.group import load_and_prepare_data, correlation_based_clustering, feature_based_clustering, time_series_clustering, dtw_clustering
from sklearn.preprocessing import StandardScaler

def load_m4_data_directly(seasonal_pattern, root_path='./datasets/m4', flag='test', max_series=500):
    """Load M4 data directly from CSV files, bypassing the Dataset_M4 class"""
    print(f"Loading {seasonal_pattern} {flag} data directly...")
    
    # Try different potential file locations
    potential_paths = [
        os.path.join(root_path, f"{seasonal_pattern}-{flag}.csv"),
    ]
    
    file_path = None
    for path in potential_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        print(f"Error: Could not find {flag} data file for {seasonal_pattern}")
        return None
    
    print(f"Found data file: {file_path}")
    
    # Load the CSV file - M4 files have 'V1' as the ID column and the rest are values
    try:
        df_raw = pd.read_csv(file_path)
        print(f"Loaded {len(df_raw)} series from {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    print(f"Final dataframe shape: {df_raw.shape}")
    return df_raw, df_raw.columns[1:]  # Return the series IDs (excluding the first column which is 'V1')

def cluster_m4_data(pattern, max_series=500):
    """Apply clustering to M4 data loaded directly from CSV files"""
    print(f"\n=== Clustering {pattern} Data ===")
    
    # Load data directly from CSV
    result = load_m4_data_directly(pattern, max_series=max_series)
    if result is None:
        print(f"Skipping {pattern} due to loading issues")
        return
    
    df, series_ids = result
    
    # Setup directories
    output_dir = f'./datasets/m4_for_clustering/clustering_{pattern}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save extracted data to CSV for inspection
    extracted_csv = os.path.join(output_dir, f"{pattern}_extracted.csv")
    df.to_csv(extracted_csv, index=False)
    print(f"Saved extracted data to {extracted_csv}")
    
    # Prepare data for clustering using the original function from group.py
    df, numeric_cols, timestamp_col = load_and_prepare_data(extracted_csv)
    print(f"Loaded {len(numeric_cols)} numeric columns for clustering")
    
    # Apply clustering methods from the original group.py
    groups_dict = {}
    
    # Correlation-based clustering (Pearson)
    print("\nPerforming Pearson correlation clustering...")
    try:
        pearson_groups, _, _ = correlation_based_clustering(df, numeric_cols, output_dir, method='pearson')
        groups_dict['Pearson'] = pearson_groups
    except Exception as e:
        print(f"Error in Pearson correlation clustering: {e}")
    
    # Correlation-based clustering (Spearman)
    print("\nPerforming Spearman correlation clustering...")
    try:
        spearman_groups, _, _ = correlation_based_clustering(df, numeric_cols, output_dir, method='spearman')
        groups_dict['Spearman'] = spearman_groups
    except Exception as e:
        print(f"Error in Spearman correlation clustering: {e}")
    
    # Feature-based clustering
    print("\nPerforming feature-based clustering...")
    try:
        feature_groups = feature_based_clustering(df, numeric_cols, output_dir)
        groups_dict['Feature'] = feature_groups
    except Exception as e:
        print(f"Error in feature-based clustering: {e}")
    
    # Time series clustering if we have a timestamp column
    if timestamp_col:
        print("\nPerforming time series clustering...")
        try:
            ts_groups = time_series_clustering(df, numeric_cols, timestamp_col, output_dir)
            if ts_groups:
                groups_dict['TimeSeries'] = ts_groups
        except Exception as e:
            print(f"Error in time series clustering: {e}")
        
        # DTW clustering - only do this for smaller datasets due to computational complexity
        if len(numeric_cols) <= 100 and len(df) <= 10 ** 4:
            print("\nPerforming DTW clustering...")
            try:
                dtw_groups = dtw_clustering(df, numeric_cols, timestamp_col, output_dir, max_warp_ratio=0.05)
                if dtw_groups:
                    groups_dict['DTW'] = dtw_groups
            except Exception as e:
                print(f"Error in DTW clustering: {e}")
        else:
            print(f"Skipping DTW clustering (too many series: {len(numeric_cols)})")
    
    # Create directory for M4-compatible files
    m4_clusters_dir = f'./datasets/m4_clustered/{pattern}'
    os.makedirs(m4_clusters_dir, exist_ok=True)
    
    # Save cluster information
    for method, groups in groups_dict.items():
        method_dir = os.path.join(m4_clusters_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        for cluster_idx, (label, columns) in enumerate(groups.items()):
            cluster_id = cluster_idx + 1
            
            # Save IDs to a text file
            with open(f"{method_dir}/cluster_{cluster_id}_ids.txt", 'w') as f:
                for col in columns:
                    f.write(f"{col}\n")
            
            print(f"Saved {method} cluster {cluster_id} with {len(columns)} series")

if __name__ == '__main__':
    os.makedirs('./datasets/m4_clustered', exist_ok=True)
    
    # Get seasonal pattern from command line or process all
    if len(sys.argv) > 1 and sys.argv[1] in ['Monthly', 'Yearly', 'Quarterly', 'Weekly', 'Daily', 'Hourly']:
        patterns = [sys.argv[1]]
    else:
        patterns = ['Monthly', 'Yearly', 'Quarterly', 'Weekly', 'Daily', 'Hourly']
    
    # Get max series count if provided
    max_series = 500  # Default to a smaller number for memory management
    if len(sys.argv) > 2:
        try:
            max_series = int(sys.argv[2])
        except ValueError:
            pass
    
    # Process each pattern
    for pattern in patterns:
        cluster_m4_data(pattern, max_series=max_series)