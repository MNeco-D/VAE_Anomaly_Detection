import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from data_loader import get_dataloaders
from vae import VAE, loss_function

def train(model, train_loader, optimizer, beta=1.0, epochs=20, device='cpu'):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kld = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += bce.item()
            epoch_kld += kld.item()
            
        print(f'Epoch {epoch+1}: Total Loss: {epoch_loss/len(train_loader.dataset):.4f} | Recon: {epoch_recon/len(train_loader.dataset):.4f} | KLD: {epoch_kld/len(train_loader.dataset):.4f}')
        train_losses.append(epoch_loss/len(train_loader.dataset))
        
    return train_losses

def evaluate_anomalies(model, test_loader, device='cpu'):
    model.eval()
    mse_scores = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            recon, _, _ = model(data)
            
            # Calculate Reconstruction Error per sample (MSE)
            # Sum over features
            loss_per_sample = torch.sum((recon - data)**2, dim=1)
            
            mse_scores.extend(loss_per_sample.cpu().numpy())
            labels.extend(target.cpu().numpy())
            
    return np.array(mse_scores), np.array(labels)

def plot_results(mse_scores, labels, beta):
    # Split scores
    normal_scores = mse_scores[labels == 0]
    anomaly_scores = mse_scores[labels == 1]
    
    plt.figure(figsize=(12, 5))
    
    # Histogram of Reconstruction Errors
    plt.subplot(1, 2, 1)
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
    plt.title(f'Reconstruction Error Distribution (beta={beta})')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.legend()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(labels, mse_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f'results_beta_{beta}.png')
    print(f"Results saved to results_beta_{beta}.png with AUC: {roc_auc:.4f}")
    return roc_auc

def main():
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 30
    LATENT_DIM = 5
    HIDDEN_DIM = 32
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Experiment with different Beta values
    betas = [0.1, 1.0, 5.0]
    
    train_loader, test_loader, n_features = get_dataloaders(batch_size=BATCH_SIZE)
    
    for beta in betas:
        print(f"\n--- Training with Beta = {beta} ---")
        model = VAE(input_dim=n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        train(model, train_loader, optimizer, beta=beta, epochs=EPOCHS, device=DEVICE)
        
        mse_scores, labels = evaluate_anomalies(model, test_loader, device=DEVICE)
        plot_results(mse_scores, labels, beta)

if __name__ == "__main__":
    main()
