"""
Visualization utilities for 3-Body Transformer.

Creates plots for:
- Training curves
- Trajectory predictions vs ground truth
- Attention weights analysis
- Chaotic vs stable performance comparison
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import json
import pickle
from mpl_toolkits.mplot3d import Axes3D

from model import ThreeBodyTransformer, ThreeBodyTransformerV2
from dataset import ThreeBodyDataset


def plot_training_curves(results_path='checkpoints/training_results.json', save_path='results/training_curves.png'):
    """Plot training and validation loss curves."""
    with open(results_path) as f:
        results = json.load(f)
    
    history = results['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], label='Train', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Stable vs Chaotic loss
    ax2 = axes[0, 1]
    ax2.plot(history['stable_loss'], label='Stable', linewidth=2, color='green')
    ax2.plot(history['chaotic_loss'], label='Chaotic', linewidth=2, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Stable vs Chaotic Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Learning rate
    ax3 = axes[1, 0]
    ax3.plot(history['lr'], linewidth=2, color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Final comparison bar chart
    ax4 = axes[1, 1]
    categories = ['Overall', 'Stable', 'Chaotic']
    values = [
        results['final_test_loss'],
        results['test_stable_loss'],
        results['test_chaotic_loss']
    ]
    colors = ['blue', 'green', 'red']
    bars = ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Test MSE Loss')
    ax4.set_title('Final Test Performance')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_trajectory_comparison(
    model_path='checkpoints/best_model.pt',
    data_path='data/three_body_trajectories.csv',
    scaler_path='data/scaler.pkl',
    save_path='results/trajectory_comparison.png',
    n_steps=50,
    trajectory_id=0
):
    """
    Compare predicted trajectory with ground truth.
    Uses autoregressive prediction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    if config['model_type'] == 'v1':
        model = ThreeBodyTransformer(**{k: v for k, v in config.items() if k != 'model_type'})
    else:
        model = ThreeBodyTransformerV2(**{k: v for k, v in config.items() if k != 'model_type'})
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load data
    df = pd.read_csv(data_path)
    traj = df[df['trajectory_id'] == trajectory_id]
    traj_type = traj['trajectory_type'].iloc[0]
    
    state_columns = []
    for body in range(1, 4):
        for coord in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
            state_columns.append(f'body{body}_{coord}')
    
    states = traj[state_columns].values.reshape(-1, 3, 6)
    
    # Scale states
    states_flat = states.reshape(-1, 18)
    states_scaled = scaler.transform(states_flat).reshape(-1, 3, 6)
    
    seq_len = config['seq_len']
    
    # Get initial sequence
    initial_seq = torch.tensor(states_scaled[:seq_len], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Autoregressive prediction
    predictions = [states_scaled[:seq_len].copy()]
    current_seq = initial_seq.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current_seq)  # (1, 3, 6)
            predictions.append(pred.cpu().numpy()[0])
            
            # Update sequence
            current_seq = torch.cat([current_seq[:, 1:, :, :], pred.unsqueeze(1)], dim=1)
    
    predictions = np.array(predictions[1:])  # Remove initial sequence
    
    # Inverse transform predictions
    predictions_flat = predictions.reshape(-1, 18)
    predictions_original = scaler.inverse_transform(predictions_flat).reshape(-1, 3, 6)
    
    # Ground truth for comparison
    ground_truth = states[seq_len:seq_len + n_steps]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['red', 'blue', 'green']
    body_names = ['Body 1', 'Body 2', 'Body 3']
    
    for body_idx in range(3):
        ax = axes[body_idx]
        
        # Ground truth trajectory
        gt_x = ground_truth[:, body_idx, 0]
        gt_y = ground_truth[:, body_idx, 1]
        ax.plot(gt_x, gt_y, '-', color=colors[body_idx], linewidth=2, label='Ground Truth', alpha=0.8)
        ax.scatter([gt_x[0]], [gt_y[0]], color=colors[body_idx], s=100, marker='o', zorder=5)
        ax.scatter([gt_x[-1]], [gt_y[-1]], color=colors[body_idx], s=100, marker='x', zorder=5)
        
        # Predicted trajectory
        pred_x = predictions_original[:, body_idx, 0]
        pred_y = predictions_original[:, body_idx, 1]
        ax.plot(pred_x, pred_y, '--', color='black', linewidth=2, label='Predicted', alpha=0.8)
        ax.scatter([pred_x[-1]], [pred_y[-1]], color='black', s=100, marker='*', zorder=5)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'{body_names[body_idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Trajectory Prediction vs Ground Truth ({traj_type.title()} - {n_steps} steps)', fontsize=14)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory comparison to {save_path}")


def plot_error_over_time(
    model_path='checkpoints/best_model.pt',
    data_path='data/three_body_trajectories.csv',
    metadata_path='data/trajectory_metadata.csv',
    scaler_path='data/scaler.pkl',
    save_path='results/error_over_time.png',
    n_steps=100
):
    """
    Plot prediction error as function of prediction horizon.
    Shows how error accumulates in autoregressive prediction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    if config['model_type'] == 'v1':
        model = ThreeBodyTransformer(**{k: v for k, v in config.items() if k != 'model_type'})
    else:
        model = ThreeBodyTransformerV2(**{k: v for k, v in config.items() if k != 'model_type'})
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    seq_len = config['seq_len']
    
    # Calculate errors for stable and chaotic trajectories
    stable_errors = []
    chaotic_errors = []
    
    df = pd.read_csv(data_path)
    state_columns = [f'body{b}_{c}' for b in range(1,4) for c in ['x','y','z','vx','vy','vz']]
    
    for _, row in metadata.iterrows():
        traj_id = row['id']
        traj_type = row['type']
        
        traj = df[df['trajectory_id'] == traj_id]
        if len(traj) < seq_len + n_steps:
            continue
        
        states = traj[state_columns].values.reshape(-1, 3, 6)
        states_flat = states.reshape(-1, 18)
        states_scaled = scaler.transform(states_flat).reshape(-1, 3, 6)
        
        initial_seq = torch.tensor(states_scaled[:seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        current_seq = initial_seq.clone()
        
        step_errors = []
        
        with torch.no_grad():
            for step in range(n_steps):
                pred = model(current_seq)
                
                # Calculate error at this step
                gt = torch.tensor(states_scaled[seq_len + step], dtype=torch.float32).to(device)
                error = ((pred[0] - gt) ** 2).mean().item()
                step_errors.append(error)
                
                # Update sequence
                current_seq = torch.cat([current_seq[:, 1:, :, :], pred.unsqueeze(1)], dim=1)
        
        if traj_type == 'stable':
            stable_errors.append(step_errors)
        else:
            chaotic_errors.append(step_errors)
    
    # Average errors
    stable_errors = np.array(stable_errors)
    chaotic_errors = np.array(chaotic_errors)
    
    stable_mean = stable_errors.mean(axis=0)
    stable_std = stable_errors.std(axis=0)
    chaotic_mean = chaotic_errors.mean(axis=0)
    chaotic_std = chaotic_errors.std(axis=0)
    
    steps = np.arange(1, n_steps + 1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(steps, stable_mean, 'g-', linewidth=2, label='Stable')
    ax.fill_between(steps, stable_mean - stable_std, stable_mean + stable_std, 
                    color='green', alpha=0.2)
    
    ax.plot(steps, chaotic_mean, 'r-', linewidth=2, label='Chaotic')
    ax.fill_between(steps, chaotic_mean - chaotic_std, chaotic_mean + chaotic_std,
                    color='red', alpha=0.2)
    
    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('MSE (normalized)')
    ax.set_title('Prediction Error vs Horizon: Stable vs Chaotic Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error over time plot to {save_path}")


def plot_all_trajectories(
    data_path='data/three_body_trajectories.csv',
    save_path='results/all_trajectories.png'
):
    """Plot sample trajectories from the dataset."""
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = ['red', 'blue', 'green']
    
    # Plot figure-8 (id=0)
    ax = axes[0, 0]
    traj = df[df['trajectory_id'] == 0]
    for body in range(1, 4):
        ax.plot(traj[f'body{body}_x'], traj[f'body{body}_y'], 
                color=colors[body-1], linewidth=1.5, alpha=0.8)
    ax.set_title('Figure-8 (Stable)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # Plot Lagrange (id=1)
    ax = axes[0, 1]
    traj = df[df['trajectory_id'] == 1]
    for body in range(1, 4):
        ax.plot(traj[f'body{body}_x'], traj[f'body{body}_y'],
                color=colors[body-1], linewidth=1.5, alpha=0.8)
    ax.set_title('Lagrange Triangle (Stable)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # Plot some chaotic ones
    chaotic_ids = df[df['trajectory_type'] == 'chaotic']['trajectory_id'].unique()[:4]
    for idx, (ax, traj_id) in enumerate(zip(axes.flat[2:], chaotic_ids)):
        traj = df[df['trajectory_id'] == traj_id]
        for body in range(1, 4):
            ax.plot(traj[f'body{body}_x'], traj[f'body{body}_y'],
                    color=colors[body-1], linewidth=1, alpha=0.6)
        ax.set_title(f'Chaotic #{idx+1}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sample Trajectories from Dataset', fontsize=14)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved all trajectories to {save_path}")


def generate_all_visualizations():
    """Generate all visualization plots."""
    print("\n=== Generating Visualizations ===\n")
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Training curves
    if Path('checkpoints/training_results.json').exists():
        plot_training_curves()
    
    # Trajectory samples
    if Path('data/three_body_trajectories.csv').exists():
        plot_all_trajectories()
    
    # Model predictions
    if Path('checkpoints/best_model.pt').exists():
        # Stable trajectory (figure-8)
        plot_trajectory_comparison(trajectory_id=0, 
                                   save_path='results/trajectory_stable.png',
                                   n_steps=50)
        
        # Chaotic trajectory
        plot_trajectory_comparison(trajectory_id=5,
                                   save_path='results/trajectory_chaotic.png',
                                   n_steps=50)
        
        # Error over time
        plot_error_over_time(n_steps=100)
    
    print("\n=== All visualizations complete ===")


if __name__ == '__main__':
    generate_all_visualizations()
