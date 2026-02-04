"""
Dataset loader for 3-Body Problem trajectories.

Handles:
- Loading CSV data
- Normalization to (-1, 1)
- Creating sequence windows (10 timesteps input -> 1 timestep output)
- Train/val/test splits
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle


class ThreeBodyDataset(Dataset):
    """
    PyTorch Dataset for 3-body trajectory prediction.
    
    Each sample is:
    - Input: 10 consecutive timesteps of body states (10, 3, 6)
    - Output: Next timestep body states (3, 6)
    """
    
    def __init__(
        self,
        data_path='data/three_body_trajectories.csv',
        seq_len=10,
        scaler=None,
        fit_scaler=False,
        trajectory_ids=None
    ):
        self.seq_len = seq_len
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Filter by trajectory IDs if specified
        if trajectory_ids is not None:
            df = df[df['trajectory_id'].isin(trajectory_ids)]
        
        # Extract state columns
        self.state_columns = []
        for body in range(1, 4):
            for coord in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
                self.state_columns.append(f'body{body}_{coord}')
        
        # Group by trajectory
        self.trajectories = []
        self.trajectory_types = []
        
        for traj_id, group in df.groupby('trajectory_id'):
            states = group[self.state_columns].values  # (n_timesteps, 18)
            states = states.reshape(-1, 3, 6)  # (n_timesteps, 3 bodies, 6 state values)
            self.trajectories.append(states)
            self.trajectory_types.append(group['trajectory_type'].iloc[0])
        
        # Combine all data for scaling
        all_states = np.concatenate([t.reshape(-1, 18) for t in self.trajectories])
        
        # Fit or use provided scaler
        if fit_scaler:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(all_states)
        elif scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = None
        
        # Create windows
        self.windows = []
        self.window_types = []
        self.window_traj_ids = []
        
        for traj_idx, trajectory in enumerate(self.trajectories):
            n_timesteps = len(trajectory)
            # Need seq_len + 1 timesteps to create one sample
            for start_idx in range(n_timesteps - seq_len):
                window = trajectory[start_idx:start_idx + seq_len + 1]  # +1 for target
                self.windows.append(window)
                self.window_types.append(self.trajectory_types[traj_idx])
                self.window_traj_ids.append(traj_idx)
        
        self.windows = np.array(self.windows)  # (n_samples, seq_len+1, 3, 6)
        print(f"Created {len(self.windows)} samples from {len(self.trajectories)} trajectories")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx].copy()  # (seq_len+1, 3, 6)
        
        # Apply scaling
        if self.scaler is not None:
            original_shape = window.shape
            window_flat = window.reshape(-1, 18)
            window_flat = self.scaler.transform(window_flat)
            window = window_flat.reshape(original_shape)
        
        # Split into input and target
        x = window[:-1]  # (seq_len, 3, 6)
        y = window[-1]   # (3, 6)
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            self.window_types[idx]
        )
    
    def get_scaler(self):
        return self.scaler


def create_data_loaders(
    data_path='data/three_body_trajectories.csv',
    metadata_path='data/trajectory_metadata.csv',
    batch_size=64,
    seq_len=10,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    """
    Create train, validation, and test data loaders.
    
    Splits by trajectory to avoid data leakage.
    """
    np.random.seed(seed)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    trajectory_ids = metadata['id'].values
    
    # Shuffle and split
    np.random.shuffle(trajectory_ids)
    
    n_train = int(len(trajectory_ids) * train_ratio)
    n_val = int(len(trajectory_ids) * val_ratio)
    
    train_ids = trajectory_ids[:n_train]
    val_ids = trajectory_ids[n_train:n_train + n_val]
    test_ids = trajectory_ids[n_train + n_val:]
    
    print(f"Train trajectories: {len(train_ids)}")
    print(f"Val trajectories: {len(val_ids)}")
    print(f"Test trajectories: {len(test_ids)}")
    
    # Create datasets
    train_dataset = ThreeBodyDataset(
        data_path, seq_len=seq_len, fit_scaler=True, trajectory_ids=train_ids
    )
    
    scaler = train_dataset.get_scaler()
    
    val_dataset = ThreeBodyDataset(
        data_path, seq_len=seq_len, scaler=scaler, trajectory_ids=val_ids
    )
    
    test_dataset = ThreeBodyDataset(
        data_path, seq_len=seq_len, scaler=scaler, trajectory_ids=test_ids
    )
    
    # Save scaler for inference
    scaler_path = Path(data_path).parent / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader, scaler


def get_chaotic_stable_splits(
    data_path='data/three_body_trajectories.csv',
    metadata_path='data/trajectory_metadata.csv',
    batch_size=64,
    seq_len=10,
    scaler=None
):
    """
    Create separate loaders for chaotic and stable trajectories.
    Useful for hypothesis evaluation.
    """
    metadata = pd.read_csv(metadata_path)
    
    stable_ids = metadata[metadata['type'] == 'stable']['id'].values
    chaotic_ids = metadata[metadata['type'] == 'chaotic']['id'].values
    
    stable_dataset = ThreeBodyDataset(
        data_path, seq_len=seq_len, scaler=scaler, trajectory_ids=stable_ids
    )
    chaotic_dataset = ThreeBodyDataset(
        data_path, seq_len=seq_len, scaler=scaler, trajectory_ids=chaotic_ids
    )
    
    stable_loader = DataLoader(stable_dataset, batch_size=batch_size, shuffle=False)
    chaotic_loader = DataLoader(chaotic_dataset, batch_size=batch_size, shuffle=False)
    
    return stable_loader, chaotic_loader


if __name__ == '__main__':
    # Test dataset creation
    train_loader, val_loader, test_loader, scaler = create_data_loaders()
    
    # Check a batch
    for x, y, types in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Input (x): {x.shape}")  # (batch, seq_len, 3, 6)
        print(f"  Target (y): {y.shape}")  # (batch, 3, 6)
        print(f"  Types: {types[:5]}")
        
        print(f"\nValue ranges (should be ~[-1, 1]):")
        print(f"  x min: {x.min():.3f}, max: {x.max():.3f}")
        print(f"  y min: {y.min():.3f}, max: {y.max():.3f}")
        break
