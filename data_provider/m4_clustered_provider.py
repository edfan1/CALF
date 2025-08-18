import os
import numpy as np
from data_provider.data_loader import Dataset_M4
from torch.utils.data import Dataset

class Dataset_M4_Clustered(Dataset_M4):
    """
    M4 dataset filtered by clusters from clustering analysis
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='m4_clustered',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Monthly', cluster_method='Pearson', 
                 cluster_id=1, percent=100):
        
        # Initialize with parent class but save cluster parameters
        self.cluster_method = cluster_method
        self.cluster_id = cluster_id
        
        # Call parent initialization
        super().__init__(root_path, flag, size, features, 
                         target, scale, inverse, timeenc, freq,
                         seasonal_patterns, percent)
    
    def __read_data__(self):
        # First, load all data using the parent method
        super().__read_data__()
        
        # Then filter based on cluster membership
        cluster_file = os.path.join(self.root_path, 'datasets/m4_clustered', 
                                   self.seasonal_patterns, self.cluster_method, 
                                   f'cluster_{self.cluster_id}_ids.txt')
        
        if not os.path.exists(cluster_file):
            raise FileNotFoundError(f"Cluster file not found: {cluster_file}")
        
        # Read series IDs in this cluster
        with open(cluster_file, 'r') as f:
            cluster_ids = [line.strip() for line in f]
        
        print(f"Loaded {len(cluster_ids)} series IDs from {self.cluster_method} cluster {self.cluster_id}")
        
        # Filter timeseries and IDs to only include those in the cluster
        cluster_indices = np.array([i for i, id_ in enumerate(self.ids) if id_ in cluster_ids])
        
        if len(cluster_indices) == 0:
            raise ValueError(f"No matching series found in cluster {self.cluster_id}")
        
        self.ids = self.ids[cluster_indices]
        self.timeseries = [self.timeseries[i] for i in cluster_indices]
        
        print(f"Filtered to {len(self.ids)} series for {self.seasonal_patterns} {self.cluster_method} cluster {self.cluster_id}")