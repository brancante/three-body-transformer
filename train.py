"""
Training script for the 3-Body Transformer.

Handles:
- Training loop with validation
- Learning rate scheduling
- Checkpointing
- Metrics logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from model import ThreeBodyTransformer, ThreeBodyTransformerV2, count_parameters
from dataset import create_data_loaders, get_chaotic_stable_splits


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(x)
        loss = criterion(predictions, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    # Track loss by trajectory type
    stable_losses = []
    chaotic_losses = []
    
    with torch.no_grad():
        for x, y, types in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            predictions = model(x)
            loss = criterion(predictions, y)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Per-sample losses for analysis
            per_sample_loss = ((predictions - y) ** 2).mean(dim=(1, 2))
            for i, t in enumerate(types):
                if t == 'stable':
                    stable_losses.append(per_sample_loss[i].item())
                else:
                    chaotic_losses.append(per_sample_loss[i].item())
    
    avg_loss = total_loss / n_batches
    stable_loss = np.mean(stable_losses) if stable_losses else 0
    chaotic_loss = np.mean(chaotic_losses) if chaotic_losses else 0
    
    return avg_loss, stable_loss, chaotic_loss


def train(
    model_type='v1',
    embed_dim=128,
    n_heads=8,
    n_layers=4,
    dim_feedforward=256,
    dropout=0.1,
    batch_size=64,
    seq_len=10,
    lr=0.001,
    epochs=100,
    patience=15,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='checkpoints',
    data_dir='data'
):
    """
    Main training function.
    """
    print(f"Using device: {device}")
    
    # Create directories
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        data_path=f'{data_dir}/three_body_trajectories.csv',
        metadata_path=f'{data_dir}/trajectory_metadata.csv',
        batch_size=batch_size,
        seq_len=seq_len
    )
    
    # Create model
    print("\nCreating model...")
    if model_type == 'v1':
        model = ThreeBodyTransformer(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_len=seq_len
        )
    else:
        model = ThreeBodyTransformerV2(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_len=seq_len
        )
    
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'stable_loss': [],
        'chaotic_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, stable_loss, chaotic_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['stable_loss'].append(stable_loss)
        history['chaotic_loss'].append(chaotic_loss)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"Stable: {stable_loss:.6f} | Chaotic: {chaotic_loss:.6f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'model_type': model_type,
                    'embed_dim': embed_dim,
                    'n_heads': n_heads,
                    'n_layers': n_layers,
                    'dim_feedforward': dim_feedforward,
                    'dropout': dropout,
                    'seq_len': seq_len
                }
            }, save_path / 'best_model.pt')
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("-" * 60)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    checkpoint = torch.load(save_path / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_stable, test_chaotic = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"  - Stable trajectories: {test_stable:.6f}")
    print(f"  - Chaotic trajectories: {test_chaotic:.6f}")
    
    # Save final results
    results = {
        'final_test_loss': test_loss,
        'test_stable_loss': test_stable,
        'test_chaotic_loss': test_chaotic,
        'best_val_loss': best_val_loss,
        'history': history,
        'config': {
            'model_type': model_type,
            'embed_dim': embed_dim,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'lr': lr,
            'epochs_trained': len(history['train_loss'])
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}/")
    
    return model, history, results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 3-Body Transformer')
    parser.add_argument('--model', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--dim-ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    train(
        model_type=args.model,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device
    )
