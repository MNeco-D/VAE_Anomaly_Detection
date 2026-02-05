import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_data(n_samples=5000, n_features=20, anomaly_ratio=0.02, random_state=42):
    """
    Generates a synthetic dataset with normal multi-modal data and anomalies.
    
    Args:
        n_samples (int): Total number of samples.
        n_features (int): Number of dimensions.
        anomaly_ratio (float): Fraction of samples that are anomalous.
        random_state (int): Random seed.
        
    Returns:
        X_train (torch.Tensor): Training features (normal only for semi-supervised, or mixed for unsupervised).
                                Note: Typically for Anomaly Detection, we train on "mostly normal" data.
                                Here we will return a split where training data is contaminated but we might want to clean it or just treat it as unsupervised.
                                The prompt asks for "unsupervised anomaly detection", so we assume the training set contains anomalies but we don't know labels.
        X_test (torch.Tensor): Test features.
        y_test (torch.Tensor): Test labels (0 for normal, 1 for anomaly).
    """
    np.random.seed(random_state)
    
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    # Generate Normal Data (Multi-modal)
    # Using 3 centers to simulate multi-modal normal behavior
    X_normal, _ = make_blobs(n_samples=n_normal, n_features=n_features, centers=3, cluster_std=1.0, random_state=random_state)
    
    # Generate Anomalies
    # Anomalies are generated from a uniform distribution or a wider/shifted Gaussian to ensure they are distinct
    # Here we simulate sensor glitches or out-of-bounds values
    X_anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, n_features))
    
    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Normalize features to [0, 1] or [-1, 1] range which is better for Neural Networks
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train/test
    # Stratify to ensure anomalies are present in both (or we could keep test set specifically for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True, stratify=y, random_state=random_state)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    y_train_tensor = torch.FloatTensor(y_train) # Helpful for debugging, though VAE is unsupervised

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def get_dataloaders(n_samples=5000, n_features=20, anomaly_ratio=0.02, batch_size=64):
    X_train, X_test, y_train, y_test = generate_synthetic_data(n_samples, n_features, anomaly_ratio)
    
    # Unsupervised VAE: Input is X, Target is X (reconstruction)
    # We include y just for tracking purposes, but the model won't see it during training
    train_dataset = TensorDataset(X_train, X_train) 
    test_dataset = TensorDataset(X_test, y_test) # For evaluation, we need labels
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, n_features

if __name__ == "__main__":
    train_loader, test_loader, n_feat = get_dataloaders()
    print(f"Data Loaded. Features: {n_feat}")
    for batch in train_loader:
        print("Train Batch Shape:", batch[0].shape)
        break
