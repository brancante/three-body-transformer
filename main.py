#!/usr/bin/env python3
"""
Three-Body Transformer - Main Entry Point

A transformer-based approach to predicting the chaotic 3-body problem.

Usage:
    python main.py --all           # Run everything (data gen, train, evaluate)
    python main.py --generate      # Generate dataset only
    python main.py --train         # Train model only
    python main.py --evaluate      # Evaluate model only
    python main.py --visualize     # Generate visualizations only
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Three-Body Transformer: Predicting Chaos with Attention'
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--generate', action='store_true',
                       help='Generate synthetic dataset')
    parser.add_argument('--train', action='store_true',
                       help='Train the transformer model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained model')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    # Training options
    parser.add_argument('--model', type=str, default='v1', choices=['v1', 'v2'],
                       help='Model architecture (v1=hierarchical, v2=joint)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--embed-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--n-layers', type=int, default=4,
                       help='Number of transformer layers')
    
    # Data options
    parser.add_argument('--n-chaotic', type=int, default=50,
                       help='Number of chaotic trajectories to generate')
    parser.add_argument('--n-steps', type=int, default=2000,
                       help='Steps per trajectory')
    
    args = parser.parse_args()
    
    # Default to --all if no action specified
    if not any([args.all, args.generate, args.train, args.evaluate, args.visualize]):
        args.all = True
    
    print("=" * 60)
    print("THREE-BODY TRANSFORMER")
    print("Predicting Chaos with Attention")
    print("=" * 60)
    
    # === STEP 1: Generate Data ===
    if args.all or args.generate:
        print("\n[STEP 1/4] Generating synthetic 3-body trajectories...")
        print("-" * 60)
        
        from data_generator import generate_dataset
        generate_dataset(
            output_dir='data',
            n_chaotic=args.n_chaotic,
            n_steps=args.n_steps
        )
        
        print("✓ Data generation complete")
    
    # === STEP 2: Train Model ===
    if args.all or args.train:
        print("\n[STEP 2/4] Training transformer model...")
        print("-" * 60)
        
        # Check if data exists
        if not Path('data/three_body_trajectories.csv').exists():
            print("ERROR: No data found. Run with --generate first.")
            sys.exit(1)
        
        from train import train
        model, history, results = train(
            model_type=args.model,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs
        )
        
        print("✓ Training complete")
    
    # === STEP 3: Evaluate ===
    if args.all or args.evaluate:
        print("\n[STEP 3/4] Evaluating model performance...")
        print("-" * 60)
        
        # Check if model exists
        if not Path('checkpoints/best_model.pt').exists():
            print("ERROR: No trained model found. Run with --train first.")
            sys.exit(1)
        
        from evaluate import evaluate_model
        results = evaluate_model()
        
        print("✓ Evaluation complete")
    
    # === STEP 4: Visualize ===
    if args.all or args.visualize:
        print("\n[STEP 4/4] Generating visualizations...")
        print("-" * 60)
        
        from visualize import generate_all_visualizations
        generate_all_visualizations()
        
        print("✓ Visualizations complete")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("""
Output locations:
  - Data:          data/
  - Model:         checkpoints/
  - Results:       results/
  - Visualizations: results/*.png
""")


if __name__ == '__main__':
    main()
