"""
Evaluation and Hypothesis Testing for 3-Body Transformer.

Tests the hypothesis: Can transformer attention learn gravitational interactions
and predict the 3-body problem?

Key questions:
1. Does the model perform better on stable vs chaotic trajectories?
2. How does prediction error grow over time (Lyapunov-like behavior)?
3. Can attention weights reveal learned body-body interactions?
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from collections import defaultdict

from model import ThreeBodyTransformer, ThreeBodyTransformerV2
from dataset import create_data_loaders, ThreeBodyDataset


def evaluate_model(
    model_path='checkpoints/best_model.pt',
    data_path='data/three_body_trajectories.csv',
    metadata_path='data/trajectory_metadata.csv',
    scaler_path='data/scaler.pkl',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Comprehensive model evaluation.
    """
    print("=" * 60)
    print("3-BODY TRANSFORMER EVALUATION")
    print("=" * 60)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    print(f"\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
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
    metadata = pd.read_csv(metadata_path)
    
    state_columns = [f'body{b}_{c}' for b in range(1,4) for c in ['x','y','z','vx','vy','vz']]
    seq_len = config['seq_len']
    
    results = {
        'single_step': {},
        'multi_step': {},
        'per_body': {},
        'per_trajectory_type': {}
    }
    
    # ===== SINGLE-STEP EVALUATION =====
    print("\n" + "=" * 60)
    print("SINGLE-STEP PREDICTION ANALYSIS")
    print("=" * 60)
    
    stable_errors = []
    chaotic_errors = []
    per_body_errors = defaultdict(list)
    
    for _, row in metadata.iterrows():
        traj_id = row['id']
        traj_type = row['type']
        
        traj = df[df['trajectory_id'] == traj_id]
        states = traj[state_columns].values.reshape(-1, 3, 6)
        states_flat = states.reshape(-1, 18)
        states_scaled = scaler.transform(states_flat).reshape(-1, 3, 6)
        
        # Single-step predictions
        traj_errors = []
        with torch.no_grad():
            for start_idx in range(len(states_scaled) - seq_len - 1):
                x = torch.tensor(states_scaled[start_idx:start_idx + seq_len],
                               dtype=torch.float32).unsqueeze(0).to(device)
                y_true = torch.tensor(states_scaled[start_idx + seq_len],
                                    dtype=torch.float32).to(device)
                
                y_pred = model(x)[0]
                
                # Overall error
                error = ((y_pred - y_true) ** 2).mean().item()
                traj_errors.append(error)
                
                # Per-body error
                for body_idx in range(3):
                    body_error = ((y_pred[body_idx] - y_true[body_idx]) ** 2).mean().item()
                    per_body_errors[body_idx].append(body_error)
        
        avg_error = np.mean(traj_errors)
        if traj_type == 'stable':
            stable_errors.append(avg_error)
        else:
            chaotic_errors.append(avg_error)
    
    stable_mean = np.mean(stable_errors)
    stable_std = np.std(stable_errors)
    chaotic_mean = np.mean(chaotic_errors)
    chaotic_std = np.std(chaotic_errors)
    
    print(f"\nStable Trajectories:")
    print(f"  Mean MSE: {stable_mean:.6f} ± {stable_std:.6f}")
    print(f"  N trajectories: {len(stable_errors)}")
    
    print(f"\nChaotic Trajectories:")
    print(f"  Mean MSE: {chaotic_mean:.6f} ± {chaotic_std:.6f}")
    print(f"  N trajectories: {len(chaotic_errors)}")
    
    ratio = chaotic_mean / stable_mean if stable_mean > 0 else float('inf')
    print(f"\nChaotic/Stable Error Ratio: {ratio:.2f}x")
    
    results['single_step'] = {
        'stable_mean': stable_mean,
        'stable_std': stable_std,
        'chaotic_mean': chaotic_mean,
        'chaotic_std': chaotic_std,
        'chaotic_stable_ratio': ratio
    }
    
    # Per-body analysis
    print(f"\nPer-Body Errors:")
    for body_idx in range(3):
        mean_err = np.mean(per_body_errors[body_idx])
        print(f"  Body {body_idx + 1}: {mean_err:.6f}")
        results['per_body'][f'body_{body_idx + 1}'] = mean_err
    
    # ===== MULTI-STEP (AUTOREGRESSIVE) EVALUATION =====
    print("\n" + "=" * 60)
    print("MULTI-STEP (AUTOREGRESSIVE) PREDICTION ANALYSIS")
    print("=" * 60)
    
    horizons = [10, 25, 50, 100]
    
    for horizon in horizons:
        print(f"\n--- Horizon: {horizon} steps ---")
        
        stable_final_errors = []
        chaotic_final_errors = []
        
        for _, row in metadata.iterrows():
            traj_id = row['id']
            traj_type = row['type']
            
            traj = df[df['trajectory_id'] == traj_id]
            if len(traj) < seq_len + horizon:
                continue
            
            states = traj[state_columns].values.reshape(-1, 3, 6)
            states_flat = states.reshape(-1, 18)
            states_scaled = scaler.transform(states_flat).reshape(-1, 3, 6)
            
            # Autoregressive prediction
            current_seq = torch.tensor(states_scaled[:seq_len],
                                      dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                for step in range(horizon):
                    pred = model(current_seq)
                    current_seq = torch.cat([current_seq[:, 1:, :, :], pred.unsqueeze(1)], dim=1)
                
                # Final prediction error
                y_true = torch.tensor(states_scaled[seq_len + horizon - 1],
                                    dtype=torch.float32).to(device)
                final_error = ((pred[0] - y_true) ** 2).mean().item()
            
            if traj_type == 'stable':
                stable_final_errors.append(final_error)
            else:
                chaotic_final_errors.append(final_error)
        
        stable_mean_h = np.mean(stable_final_errors) if stable_final_errors else 0
        chaotic_mean_h = np.mean(chaotic_final_errors) if chaotic_final_errors else 0
        
        print(f"  Stable MSE:  {stable_mean_h:.6f}")
        print(f"  Chaotic MSE: {chaotic_mean_h:.6f}")
        
        results['multi_step'][f'horizon_{horizon}'] = {
            'stable': stable_mean_h,
            'chaotic': chaotic_mean_h
        }
    
    # ===== HYPOTHESIS EVALUATION =====
    print("\n" + "=" * 60)
    print("HYPOTHESIS EVALUATION")
    print("=" * 60)
    
    print("""
Hypothesis: Can transformer attention learn gravitational interactions
and predict the 3-body problem?

Key Findings:
""")
    
    # Finding 1: Overall learning
    print("1. LEARNING CAPABILITY")
    if stable_mean < 0.01:
        print("   ✓ Model successfully learns to predict body trajectories")
        print(f"   → Single-step MSE on stable orbits: {stable_mean:.6f}")
    else:
        print("   △ Model shows limited learning")
        print(f"   → Single-step MSE: {stable_mean:.6f}")
    
    # Finding 2: Stable vs Chaotic
    print("\n2. STABLE vs CHAOTIC TRAJECTORIES")
    if ratio > 1.5:
        print(f"   ✓ Model handles stable orbits better ({ratio:.1f}x lower error)")
        print("   → This matches physical intuition: chaotic systems are harder to predict")
    else:
        print(f"   △ Similar performance on stable and chaotic (ratio: {ratio:.2f}x)")
    
    # Finding 3: Horizon growth
    print("\n3. ERROR ACCUMULATION WITH HORIZON")
    horizons_data = results['multi_step']
    if len(horizons_data) >= 2:
        h10 = horizons_data.get('horizon_10', {}).get('chaotic', 0)
        h100 = horizons_data.get('horizon_100', {}).get('chaotic', 0)
        if h10 > 0:
            growth = h100 / h10
            print(f"   → Chaotic error growth (10→100 steps): {growth:.1f}x")
            if growth > 10:
                print("   ✓ Exhibits Lyapunov-like exponential error growth")
            else:
                print("   △ Sub-exponential error growth observed")
    
    # Finding 4: Per-body consistency
    print("\n4. PER-BODY PREDICTION CONSISTENCY")
    body_errors = [results['per_body'][f'body_{i+1}'] for i in range(3)]
    max_ratio = max(body_errors) / min(body_errors) if min(body_errors) > 0 else float('inf')
    if max_ratio < 2:
        print(f"   ✓ Consistent across all bodies (max ratio: {max_ratio:.2f}x)")
    else:
        print(f"   △ Inconsistent per-body performance (ratio: {max_ratio:.2f}x)")
    
    # Overall verdict
    print("\n" + "-" * 60)
    print("OVERALL VERDICT:")
    if stable_mean < 0.01 and ratio > 1.2:
        print("✓ HYPOTHESIS SUPPORTED")
        print("  The transformer architecture can learn to predict 3-body dynamics")
        print("  with attention capturing body-body interactions. However, chaotic")
        print("  trajectories remain fundamentally harder due to sensitivity to")
        print("  initial conditions (as expected from chaos theory).")
    elif stable_mean < 0.1:
        print("△ HYPOTHESIS PARTIALLY SUPPORTED")
        print("  The model learns some structure but may need more data/capacity")
        print("  to fully capture gravitational dynamics.")
    else:
        print("✗ HYPOTHESIS NOT WELL SUPPORTED")
        print("  Model struggles to learn the underlying dynamics.")
    print("-" * 60)
    
    # Save results
    results_path = Path('results/evaluation_results.json')
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == '__main__':
    evaluate_model()
