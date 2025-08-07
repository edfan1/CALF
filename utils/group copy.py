import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # Install with: pip install fastdtw
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """Load dataset and prepare for column clustering"""
    print(f"Loading dataset: {filepath}")
    df = pd.read_csv(filepath)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Check for date/timestamp column - now checks ALL columns, not just numeric ones
    timestamp_col = None
    
    # First check for datetime columns
    for col in df.columns:
        # Try to parse as datetime
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                timestamp_col = col
                if col in numeric_cols:
                    numeric_cols.remove(col)
                print(f"Identified datetime column: {timestamp_col}")
                break
        except:
            pass
    
    # If no datetime column found, check for column names containing time-related terms
    if timestamp_col is None:
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
                timestamp_col = col
                if col in numeric_cols:
                    numeric_cols.remove(col)
                print(f"Identified potential date column by name: {timestamp_col}")
                break
    
    # If still no column found, try to parse string columns as dates
    if timestamp_col is None:
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Attempt to parse the first few non-null values
                sample = df[col].dropna().head(5)
                if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notna().all():
                    timestamp_col = col
                    print(f"Identified string column parseable as dates: {timestamp_col}")
                    break
            except:
                continue
    
    if timestamp_col is None:
        print("No date/timestamp column detected")
    
    print(f"Found {len(numeric_cols)} numeric columns for clustering")
    return df, numeric_cols, timestamp_col

def correlation_based_clustering(df, numeric_cols, output_dir, method='pearson'):
    """Group columns based on correlation matrix"""
    print(f"\n=== Correlation-based Clustering ({method}) ===")
    
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = df[numeric_cols].corr(method='pearson')
    else:
        corr_matrix = df[numeric_cols].corr(method='spearman')
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title(f'{method.capitalize()} Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{method}.png'))
    plt.close()
    
    # Convert correlation to distance (1 - |corr|)
    dist_matrix = 1 - corr_matrix.abs()
    
    # Perform hierarchical clustering on the distance matrix
    linkage_matrix = linkage(dist_matrix, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(14, 8))
    dendrogram(linkage_matrix, labels=numeric_cols, leaf_rotation=90)
    plt.title(f'Hierarchical Clustering based on {method.capitalize()} Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_dendrogram_{method}.png'))
    plt.close()
    
    # Determine optimal number of clusters
    cluster_counts = range(2, min(11, len(numeric_cols)))
    silhouette_scores = []
    
    for n_clusters in cluster_counts:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(dist_matrix)
        silhouette_avg = silhouette_score(dist_matrix, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Optimal Number of Clusters ({method})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'correlation_silhouette_{method}.png'))
    plt.close()
    
    # Find optimal number of clusters
    optimal_n_clusters = cluster_counts[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    
    # Create clusters with optimal number
    clustering = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(dist_matrix)
    
    # Create groups
    groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(numeric_cols[i])
    
    # Print and save groups
    with open(os.path.join(output_dir, f'correlation_clusters_{method}.txt'), 'w') as f:
        f.write(f"Column Groups based on {method.capitalize()} Correlation\n")
        f.write("="*50 + "\n\n")
        
        for i, (label, columns) in enumerate(groups.items()):
            group_info = f"Group {i+1}: {len(columns)} columns"
            print(group_info)
            print("  " + ", ".join(columns))
            
            f.write(f"Group {i+1}: {len(columns)} columns\n")
            for col in columns:
                f.write(f"  - {col}\n")
            f.write("\n")
    
    return groups, dist_matrix, corr_matrix

def feature_based_clustering(df, numeric_cols, output_dir):
    """Cluster columns based on feature behavior (transpose data for column clustering)"""
    print("\n=== Feature-based K-means Clustering ===")
    
    # Transpose data to make columns as samples
    # First, standardize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols
    )
    
    # Calculate column statistics as features for clustering
    col_stats = pd.DataFrame(index=numeric_cols)
    col_stats['mean'] = df[numeric_cols].mean()
    col_stats['std'] = df[numeric_cols].std()
    col_stats['min'] = df[numeric_cols].min()
    col_stats['max'] = df[numeric_cols].max()
    col_stats['median'] = df[numeric_cols].median()
    col_stats['skew'] = df[numeric_cols].skew()
    col_stats['kurtosis'] = df[numeric_cols].kurtosis()
    
    # Scale the statistics
    stats_scaled = StandardScaler().fit_transform(col_stats)
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    cluster_counts = range(1, min(11, len(numeric_cols)))
    
    for n_clusters in cluster_counts:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(stats_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'feature_elbow.png'))
    plt.close()
    
    # Calculate silhouette scores
    silhouette_scores = []
    for n_clusters in cluster_counts[1:]:  # Skip 1 cluster as silhouette requires >= 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(stats_scaled)
        silhouette_avg = silhouette_score(stats_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts[1:], silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'feature_silhouette.png'))
    plt.close()
    
    # Find optimal number of clusters
    optimal_n_clusters = cluster_counts[1:][np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    
    # Apply K-means with optimal clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(stats_scaled)
    
    # Visualize clusters with PCA
    pca = PCA(n_components=2)
    stats_2d = pca.fit_transform(stats_scaled)
    
    plt.figure(figsize=(12, 10))
    for cluster_num in range(optimal_n_clusters):
        plt.scatter(
            stats_2d[cluster_labels == cluster_num, 0],
            stats_2d[cluster_labels == cluster_num, 1],
            label=f'Cluster {cluster_num+1}',
            alpha=0.7,
            s=80
        )
    
    # Add column labels to the plot
    for i, col in enumerate(numeric_cols):
        plt.annotate(col, (stats_2d[i, 0], stats_2d[i, 1]), fontsize=8)
    
    plt.title('PCA Visualization of Column Clusters')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'feature_pca.png'))
    plt.close()
    
    # Create groups
    groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(numeric_cols[i])
    
    # Print and save groups
    with open(os.path.join(output_dir, 'feature_clusters.txt'), 'w') as f:
        f.write("Column Groups based on Feature Behavior\n")
        f.write("="*50 + "\n\n")
        
        for i, (label, columns) in enumerate(groups.items()):
            group_info = f"Group {i+1}: {len(columns)} columns"
            print(group_info)
            print("  " + ", ".join(columns))
            
            f.write(f"Group {i+1}: {len(columns)} columns\n")
            for col in columns:
                f.write(f"  - {col}\n")
            f.write("\n")
    
    return groups

def time_series_clustering(df, numeric_cols, timestamp_col, output_dir):
    """Cluster columns based on time series behavior"""
    if timestamp_col is None:
        print("No timestamp column detected. Skipping time series clustering.")
        return None
    
    print("\n=== Time Series Behavior Clustering ===")
    
    # Calculate derivatives (rate of change)
    df_derivative = df[numeric_cols].diff().dropna()
    
    # Calculate cross-correlation matrix
    cross_corr = {}
    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                # Calculate cross-correlation with different lags
                series1 = df[col1].values
                series2 = df[col2].values
                
                # Get max correlation at different lags
                max_corr = 0
                max_lag = 0
                for lag in range(-5, 6):  # -5 to +5 lags
                    if lag < 0:
                        s1 = series1[abs(lag):]
                        s2 = series2[:len(s1)]
                    elif lag > 0:
                        s2 = series2[lag:]
                        s1 = series1[:len(s2)]
                    else:
                        s1 = series1
                        s2 = series2
                    
                    # Ensure same length
                    min_len = min(len(s1), len(s2))
                    s1 = s1[:min_len]
                    s2 = s2[:min_len]
                    
                    corr = np.abs(np.corrcoef(s1, s2)[0, 1]) if len(s1) > 1 else 0
                    if np.isnan(corr):
                        corr = 0
                    
                    if corr > max_corr:
                        max_corr = corr
                        max_lag = lag
                
                cross_corr[(col1, col2)] = (max_corr, max_lag)
    
    # Create a cross-correlation matrix
    cross_corr_matrix = pd.DataFrame(0, index=numeric_cols, columns=numeric_cols)
    for (col1, col2), (corr, _) in cross_corr.items():
        cross_corr_matrix.loc[col1, col2] = corr
    
    # Plot cross-correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cross_corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Time Series Cross-Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series_correlation.png'))
    plt.close()
    
    # Convert to distance matrix for clustering
    dist_matrix = 1 - cross_corr_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(14, 8))
    dendrogram(linkage_matrix, labels=numeric_cols, leaf_rotation=90)
    plt.title('Hierarchical Clustering based on Time Series Behavior')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series_dendrogram.png'))
    plt.close()
    
    # Determine optimal number of clusters
    cluster_counts = range(2, min(11, len(numeric_cols)))
    silhouette_scores = []
    
    for n_clusters in cluster_counts:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(dist_matrix)
        silhouette_avg = silhouette_score(dist_matrix, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find optimal number of clusters
    optimal_n_clusters = cluster_counts[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    
    # Create clusters with optimal number
    clustering = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(dist_matrix)
    
    # Create groups
    groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(numeric_cols[i])
    
    # Print and save groups
    with open(os.path.join(output_dir, 'time_series_clusters.txt'), 'w') as f:
        f.write("Column Groups based on Time Series Behavior\n")
        f.write("="*50 + "\n\n")
        
        for i, (label, columns) in enumerate(groups.items()):
            group_info = f"Group {i+1}: {len(columns)} columns"
            print(group_info)
            print("  " + ", ".join(columns))
            
            f.write(f"Group {i+1}: {len(columns)} columns\n")
            for col in columns:
                f.write(f"  - {col}\n")
            f.write("\n")
    
    return groups

def dtw_clustering(df, numeric_cols, timestamp_col, output_dir, max_warp_ratio=0.1):
    """Cluster columns based on Dynamic Time Warping distance"""
    if timestamp_col is None:
        print("No timestamp column detected. Skipping DTW clustering.")
        return None
    
    print("\n=== Dynamic Time Warping Clustering ===")
    
    # Standardize the data for better comparison
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols
    )
    
    # Calculate DTW distance matrix
    n_cols = len(numeric_cols)
    dtw_matrix = np.zeros((n_cols, n_cols))
    max_warping_window = int(max_warp_ratio * len(df))  # Limit warping window for efficiency
    
    print(f"Computing DTW distances between {n_cols} columns (this may take time)...")
    
    # Create progress tracking
    total_comparisons = (n_cols * (n_cols - 1)) // 2
    completed = 0
    
    for i in range(n_cols):
        for j in range(i+1, n_cols):
            # Get time series
            ts1 = df_scaled[numeric_cols[i]].values
            ts2 = df_scaled[numeric_cols[j]].values
            
            # Calculate DTW distance with constrained window for efficiency
            distance, _ = fastdtw(ts1, ts2, dist=euclidean, radius=max_warping_window)
            
            # Store in matrix (symmetric)
            dtw_matrix[i, j] = distance
            dtw_matrix[j, i] = distance
            
            # Update progress
            completed += 1
            if completed % max(1, total_comparisons // 10) == 0:
                print(f"Progress: {completed}/{total_comparisons} comparisons ({completed/total_comparisons:.1%})")
    
    # Convert to DataFrame for better visualization
    dtw_df = pd.DataFrame(dtw_matrix, index=numeric_cols, columns=numeric_cols)
    
    # Plot DTW distance heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(dtw_df, annot=False, cmap='viridis')
    plt.title('Dynamic Time Warping Distances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dtw_distance_heatmap.png'))
    plt.close()
    
    # Perform hierarchical clustering on the DTW distances
    linkage_matrix = linkage(dtw_matrix, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(14, 8))
    dendrogram(linkage_matrix, labels=numeric_cols, leaf_rotation=90)
    plt.title('Hierarchical Clustering based on DTW Distances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dtw_dendrogram.png'))
    plt.close()
    
    # Determine optimal number of clusters
    cluster_counts = range(2, min(11, len(numeric_cols)))
    silhouette_scores = []
    
    for n_clusters in cluster_counts:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(dtw_matrix)
        silhouette_avg = silhouette_score(dtw_matrix, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find optimal number of clusters
    optimal_n_clusters = cluster_counts[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters from DTW: {optimal_n_clusters}")
    
    # Create clusters with optimal number
    clustering = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(dtw_matrix)
    
    # Create groups
    groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(numeric_cols[i])
    
    # Print and save groups
    with open(os.path.join(output_dir, 'dtw_clusters.txt'), 'w') as f:
        f.write("Column Groups based on Dynamic Time Warping\n")
        f.write("="*50 + "\n\n")
        
        for i, (label, columns) in enumerate(groups.items()):
            group_info = f"Group {i+1}: {len(columns)} columns"
            print(group_info)
            print("  " + ", ".join(columns))
            
            f.write(f"Group {i+1}: {len(columns)} columns\n")
            for col in columns:
                f.write(f"  - {col}\n")
            f.write("\n")
            
    # Visualize example DTW alignments for first cluster
    if len(groups) > 0:
        # Get first group
        first_group = list(groups.values())[0]
        if len(first_group) >= 2:
            # Take first two columns from first group
            col1 = first_group[0]
            col2 = first_group[1]
            
            # Get time series
            ts1 = df_scaled[col1].values[:200]  # Limit to first 200 points for visualization
            ts2 = df_scaled[col2].values[:200]
            
            # Calculate DTW with path
            distance, path = fastdtw(ts1, ts2, dist=euclidean, radius=max_warping_window)
            
            # Visualize alignment
            plt.figure(figsize=(12, 8))
            
            # Plot both time series
            plt.subplot(2, 1, 1)
            plt.plot(ts1, 'b-', label=col1)
            plt.plot(ts2, 'r-', label=col2)
            plt.title(f'Time Series Comparison - DTW Distance: {distance:.2f}')
            plt.legend()
            
            # Plot DTW alignment
            plt.subplot(2, 1, 2)
            plt.plot(ts1, 'b-', label=col1)
            plt.plot(ts2, 'r-', label=col2)
            
            # Draw alignment lines
            path_array = np.array(path)
            for i, j in path:
                if i % 5 == 0:  # Plot every 5th connection to avoid clutter
                    plt.plot([i, j], [ts1[i], ts2[j]], 'k-', alpha=0.2)
            
            plt.title('DTW Alignment Between Time Series')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'dtw_alignment_example.png'))
            plt.close()
    
    return groups

def save_column_groups(groups_dict, dataset_name, output_dir, df, timestamp_col=None):
    """
    Save columns from each group to separate CSV files with actual data
    
    Args:
        groups_dict: Dictionary of clustering results
        dataset_name: Name of the dataset
        output_dir: Directory to save results
        df: Original dataframe with all data
        timestamp_col: Name of timestamp column (if exists)
    """
    grouped_dir = os.path.join(output_dir, "grouped_columns")
    os.makedirs(grouped_dir, exist_ok=True)
    
    print(f"\nSaving grouped columns to {grouped_dir}")
    if timestamp_col:
        print(f"Including date/timestamp column '{timestamp_col}' in all group files")
    else:
        print("No date/timestamp column detected for inclusion")
    
    # Create a summary file
    with open(os.path.join(output_dir, "00_clustering_summary.txt"), "w") as f:
        f.write("Column Grouping Summary\n")
        f.write("=====================\n\n")
        
        for method, groups in groups_dict.items():
            f.write(f"\n{method} Clustering Results:\n")
            f.write("-"*50 + "\n")
            
            for i, (label, columns) in enumerate(groups.items()):
                group_num = i+1
                f.write(f"Group {group_num}: {len(columns)} columns\n")
                for col in columns:
                    f.write(f"  - {col}\n")
                f.write("\n")
                
                # Prepare columns for export, including timestamp if available
                export_columns = []
                if timestamp_col is not None:
                    export_columns.append(timestamp_col)
                export_columns.extend([c for c in columns if c != timestamp_col])  # Avoid duplicating timestamp
                
                # Export actual data to CSV
                group_file = os.path.join(grouped_dir, f"{dataset_name}_{method}_group_{group_num}.csv")
                df[export_columns].to_csv(group_file, index=False)
                
                num_rows = len(df)
                num_cols = len(export_columns)
                print(f"Saved {method} group {group_num} data ({num_rows} rows × {num_cols} columns) to {group_file}")
                
                # Also create a columns-only file for easy reference
                cols_file = os.path.join(grouped_dir, f"{dataset_name}_{method}_group_{group_num}_columns.txt")
                with open(cols_file, "w") as cols_f:
                    cols_f.write(",".join(export_columns))

if __name__ == "__main__":
    # Ask for dataset path
    default_path = "./datasets/network/processed/snmp_first10.csv"
    
    filepath = input(f"Enter dataset path (default: {default_path}): ")
    if not filepath.strip():
        filepath = default_path
    
    if not os.path.exists(filepath):
        # Try with project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        alternative_path = os.path.join(project_root, filepath)
        if os.path.exists(alternative_path):
            filepath = alternative_path
        else:
            print(f"Error: File not found at {filepath} or {alternative_path}")
            exit(1)
    
    # Create output directory
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    output_dir = os.path.join(os.path.dirname(filepath), f"clustering_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    
    # Load data
    df, numeric_cols, timestamp_col = load_and_prepare_data(filepath)
    
    # Skip if not enough numeric columns
    if len(numeric_cols) < 3:
        print("Not enough numeric columns for meaningful clustering. Need at least 3 columns.")
        exit(1)
        
    # Apply different clustering methods
    groups_dict = {}
    
    # Correlation-based clustering
    pearson_groups, _, _ = correlation_based_clustering(df, numeric_cols, output_dir, method='pearson')
    groups_dict['Pearson'] = pearson_groups
    
    spearman_groups, _, _ = correlation_based_clustering(df, numeric_cols, output_dir, method='spearman')
    groups_dict['Spearman'] = spearman_groups
    
    # Feature-based clustering
    feature_groups = feature_based_clustering(df, numeric_cols, output_dir)
    groups_dict['Feature'] = feature_groups
    
    # Time series clustering if timestamp column exists
    if timestamp_col is not None:
        ts_groups = time_series_clustering(df, numeric_cols, timestamp_col, output_dir)
        if ts_groups:
            groups_dict['TimeSeries'] = ts_groups
    
    # Save results
    save_column_groups(groups_dict, dataset_name, output_dir, df, timestamp_col)
    
    print(f"\nClustering analysis complete. Results saved to {output_dir}")