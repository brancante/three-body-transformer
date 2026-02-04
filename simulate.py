#!/usr/bin/env python3
"""
Three-Body Transformer Simulation App

Interactive CLI and optional web interface for running 3-body simulations
using the trained transformer model.

Usage:
    # Run simulation with default initial conditions (figure-8)
    python simulate.py

    # Custom initial conditions via JSON file
    python simulate.py --config my_config.json

    # Custom positions and velocities directly
    python simulate.py --positions "0,0,0;1,0,0;0,1,0" --velocities "0,0.5,0;0,-0.5,0;0,0,0"

    # Launch web interface
    python simulate.py --web

Author: Gustavo Brancante
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Local imports
from model import ThreeBodyTransformer, ThreeBodyTransformerV2
from data_generator import (
    figure_eight_initial_conditions,
    lagrange_initial_conditions,
    chaotic_initial_conditions,
    three_body_ode,
    gravitational_acceleration
)
from scipy.integrate import solve_ivp


class ThreeBodySimulator:
    """
    Main simulator class that combines numerical integration with
    transformer-based prediction.
    """
    
    def __init__(
        self,
        model_path: str = 'checkpoints/best_model.pt',
        scaler_path: str = 'data/scaler.pkl',
        device: str = 'auto'
    ):
        """
        Initialize the simulator.
        
        Args:
            model_path: Path to the trained model checkpoint
            scaler_path: Path to the saved scaler for normalization
            device: 'cpu', 'cuda', or 'auto' (auto-detect)
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model and scaler
        self.model = None
        self.scaler = None
        self.config = None
        self._load_model()
        self._load_scaler()
        
        # Default simulation parameters
        self.seq_len = 10
        self.dt = 0.025  # Time step (matches training data: t_max=50, n_steps=2000)
        
    def _load_model(self):
        """Load the trained transformer model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # Determine model type and create architecture
        model_type = self.config.get('model_type', 'v1')
        
        if model_type == 'v2':
            self.model = ThreeBodyTransformerV2(
                embed_dim=self.config.get('embed_dim', 128),
                n_heads=self.config.get('n_heads', 8),
                n_layers=self.config.get('n_layers', 4),
                dim_feedforward=self.config.get('dim_feedforward', 256),
                dropout=0.0  # No dropout in inference
            )
        else:
            self.model = ThreeBodyTransformer(
                embed_dim=self.config.get('embed_dim', 128),
                n_heads=self.config.get('n_heads', 8),
                n_layers=self.config.get('n_layers', 4),
                dim_feedforward=self.config.get('dim_feedforward', 256),
                dropout=0.0
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded model ({model_type}) from {self.model_path}")
        print(f"  Device: {self.device}")
        
    def _load_scaler(self):
        """Load the data scaler for normalization."""
        if not self.scaler_path.exists():
            print(f"⚠ Scaler not found at {self.scaler_path}, using identity transform")
            self.scaler = None
            return
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Loaded scaler from {self.scaler_path}")
    
    def _normalize(self, states: np.ndarray) -> np.ndarray:
        """Normalize states using the trained scaler."""
        if self.scaler is None:
            return states
        original_shape = states.shape
        flat = states.reshape(-1, 18)
        normalized = self.scaler.transform(flat)
        return normalized.reshape(original_shape)
    
    def _denormalize(self, states: np.ndarray) -> np.ndarray:
        """Denormalize states back to physical units."""
        if self.scaler is None:
            return states
        original_shape = states.shape
        flat = states.reshape(-1, 18)
        denormalized = self.scaler.inverse_transform(flat)
        return denormalized.reshape(original_shape)
    
    def generate_seed_trajectory(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        n_steps: int = 10,
        dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate initial seed trajectory using numerical integration.
        
        Args:
            positions: Initial positions (3, 3) for 3 bodies, 3D coords
            velocities: Initial velocities (3, 3)
            masses: Body masses (3,)
            n_steps: Number of steps to generate (default: seq_len)
            dt: Time step (default: self.dt)
        
        Returns:
            Trajectory of shape (n_steps, 3, 6) - [positions, velocities]
        """
        if dt is None:
            dt = self.dt
        
        t_span = (0, (n_steps + 1) * dt)
        t_eval = np.linspace(0, t_span[1], n_steps + 1)
        
        state0 = np.concatenate([positions.flatten(), velocities.flatten()])
        
        sol = solve_ivp(
            three_body_ode,
            t_span,
            state0,
            args=(masses, 1.0),
            t_eval=t_eval,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )
        
        # Parse into (n_steps, 3, 6) format
        trajectory = np.zeros((n_steps + 1, 3, 6))
        for body_idx in range(3):
            trajectory[:, body_idx, 0] = sol.y[body_idx * 3]      # x
            trajectory[:, body_idx, 1] = sol.y[body_idx * 3 + 1]  # y
            trajectory[:, body_idx, 2] = sol.y[body_idx * 3 + 2]  # z
            trajectory[:, body_idx, 3] = sol.y[9 + body_idx * 3]      # vx
            trajectory[:, body_idx, 4] = sol.y[9 + body_idx * 3 + 1]  # vy
            trajectory[:, body_idx, 5] = sol.y[9 + body_idx * 3 + 2]  # vz
        
        return trajectory[:n_steps]  # Return exactly n_steps
    
    @torch.no_grad()
    def predict_next_state(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict the next state given a sequence of states.
        
        Args:
            sequence: Input sequence of shape (seq_len, 3, 6)
        
        Returns:
            Predicted next state of shape (3, 6)
        """
        # Normalize
        normalized = self._normalize(sequence)
        
        # Convert to tensor
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        prediction = self.model(x)
        
        # Convert back to numpy and denormalize
        pred_np = prediction.cpu().numpy()[0]  # (3, 6)
        pred_np = pred_np.reshape(1, 3, 6)
        denormalized = self._denormalize(pred_np)
        
        return denormalized[0]
    
    def simulate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        n_prediction_steps: int = 100,
        compare_numerical: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Run a full simulation starting from initial conditions.
        
        Args:
            positions: Initial positions (3, 3)
            velocities: Initial velocities (3, 3)
            masses: Body masses (3,)
            n_prediction_steps: Number of steps to predict
            compare_numerical: Also compute numerical solution for comparison
            verbose: Print progress
        
        Returns:
            Dictionary containing:
                - 'predicted': Predicted trajectory (n_steps, 3, 6)
                - 'numerical': Numerical trajectory (if compare_numerical)
                - 'times': Time array
                - 'mse': Mean squared error vs numerical (if compare_numerical)
        """
        if verbose:
            print(f"\n🚀 Running simulation for {n_prediction_steps} steps...")
        
        # Generate seed trajectory
        seed = self.generate_seed_trajectory(positions, velocities, masses, self.seq_len)
        
        # Initialize prediction buffer
        total_steps = self.seq_len + n_prediction_steps
        predicted = np.zeros((total_steps, 3, 6))
        predicted[:self.seq_len] = seed
        
        # Autoregressive prediction
        for step in range(n_prediction_steps):
            if verbose and (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{n_prediction_steps}")
            
            # Get last seq_len states
            input_seq = predicted[step:step + self.seq_len]
            
            # Predict next state
            next_state = self.predict_next_state(input_seq)
            predicted[self.seq_len + step] = next_state
        
        # Generate times
        times = np.arange(total_steps) * self.dt
        
        result = {
            'predicted': predicted,
            'times': times,
            'seed_length': self.seq_len
        }
        
        # Compare with numerical integration
        if compare_numerical:
            if verbose:
                print("  Computing numerical reference...")
            
            numerical = self.generate_seed_trajectory(
                positions, velocities, masses, 
                n_steps=total_steps, dt=self.dt
            )
            
            # Compute MSE (only for prediction steps, not seed)
            mse = np.mean((predicted[self.seq_len:] - numerical[self.seq_len:]) ** 2)
            
            result['numerical'] = numerical
            result['mse'] = mse
            
            if verbose:
                print(f"\n📊 Results:")
                print(f"  Total steps: {total_steps}")
                print(f"  Seed length: {self.seq_len}")
                print(f"  Prediction MSE: {mse:.6f}")
        
        return result
    
    def visualize(
        self,
        result: Dict,
        save_path: Optional[str] = None,
        show: bool = True,
        title: str = "Three-Body Simulation"
    ):
        """
        Visualize simulation results.
        
        Args:
            result: Output from simulate()
            save_path: Path to save the figure (optional)
            show: Whether to display the plot
            title: Plot title
        """
        predicted = result['predicted']
        times = result['times']
        seed_len = result['seed_length']
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 2D Trajectory plot
        ax1 = fig.add_subplot(2, 2, 1)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        labels = ['Body 1', 'Body 2', 'Body 3']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            # Seed (dashed)
            ax1.plot(predicted[:seed_len, i, 0], predicted[:seed_len, i, 1],
                    '--', color=color, alpha=0.5, linewidth=1)
            # Prediction (solid)
            ax1.plot(predicted[seed_len:, i, 0], predicted[seed_len:, i, 1],
                    '-', color=color, label=label, linewidth=1.5)
            # Starting point
            ax1.scatter(predicted[0, i, 0], predicted[0, i, 1],
                       s=100, c=color, marker='o', edgecolors='black', zorder=5)
            # Ending point
            ax1.scatter(predicted[-1, i, 0], predicted[-1, i, 1],
                       s=100, c=color, marker='s', edgecolors='black', zorder=5)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Predicted Trajectories (2D Projection)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Comparison with numerical (if available)
        ax2 = fig.add_subplot(2, 2, 2)
        if 'numerical' in result:
            numerical = result['numerical']
            for i, (color, label) in enumerate(zip(colors, labels)):
                ax2.plot(numerical[:, i, 0], numerical[:, i, 1],
                        '-', color=color, alpha=0.3, linewidth=2, label=f'{label} (Numerical)')
                ax2.plot(predicted[:, i, 0], predicted[:, i, 1],
                        '--', color=color, linewidth=1.5, label=f'{label} (Predicted)')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            ax2.set_title(f'Predicted vs Numerical (MSE: {result["mse"]:.6f})')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No numerical comparison\navailable',
                    ha='center', va='center', fontsize=14)
            ax2.set_title('Comparison')
        
        # 3. Position over time
        ax3 = fig.add_subplot(2, 2, 3)
        for i, (color, label) in enumerate(zip(colors, labels)):
            ax3.plot(times, predicted[:, i, 0], '-', color=color, label=f'{label} X', linewidth=1)
            ax3.plot(times, predicted[:, i, 1], '--', color=color, label=f'{label} Y', linewidth=1, alpha=0.7)
        ax3.axvline(x=times[seed_len], color='gray', linestyle=':', label='Prediction start', alpha=0.7)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Position')
        ax3.set_title('Position vs Time')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Error over time (if numerical available)
        ax4 = fig.add_subplot(2, 2, 4)
        if 'numerical' in result:
            numerical = result['numerical']
            errors = np.sqrt(np.sum((predicted - numerical) ** 2, axis=(1, 2)))
            ax4.plot(times[seed_len:], errors[seed_len:], 'r-', linewidth=1.5)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Total Position Error')
            ax4.set_title('Prediction Error Over Time')
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        else:
            ax4.text(0.5, 0.5, 'No error data\navailable',
                    ha='center', va='center', fontsize=14)
            ax4.set_title('Error Analysis')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_animation(
        self,
        result: Dict,
        save_path: str = 'simulation.gif',
        fps: int = 30,
        duration: float = 10.0
    ):
        """
        Create an animated GIF of the simulation.
        
        Args:
            result: Output from simulate()
            save_path: Path to save the GIF
            fps: Frames per second
            duration: Target duration in seconds
        """
        predicted = result['predicted']
        seed_len = result['seed_length']
        n_frames = int(fps * duration)
        
        # Sample frames evenly
        frame_indices = np.linspace(0, len(predicted) - 1, n_frames, dtype=int)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Set axis limits
        all_x = predicted[:, :, 0].flatten()
        all_y = predicted[:, :, 1].flatten()
        margin = 0.2 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Initialize plot elements
        trails = [ax.plot([], [], '-', color=c, alpha=0.5, linewidth=1)[0] for c in colors]
        bodies = [ax.scatter([], [], s=200, c=c, edgecolors='black', zorder=5) for c in colors]
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top', fontfamily='monospace')
        
        def animate(frame):
            idx = frame_indices[frame]
            
            for i in range(3):
                # Trail (last 50 points or from start)
                trail_start = max(0, idx - 50)
                trails[i].set_data(predicted[trail_start:idx+1, i, 0],
                                  predicted[trail_start:idx+1, i, 1])
                
                # Body position
                bodies[i].set_offsets([[predicted[idx, i, 0], predicted[idx, i, 1]]])
            
            time_text.set_text(f't = {result["times"][idx]:.2f}')
            
            return trails + bodies + [time_text]
        
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, blit=True)
        
        print(f"🎬 Creating animation ({n_frames} frames)...")
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"✓ Saved animation to {save_path}")
        
        plt.close()


def parse_array(s: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Parse a string like '0,0,0;1,0,0;0,1,0' into a numpy array."""
    rows = s.split(';')
    data = []
    for row in rows:
        values = [float(x.strip()) for x in row.split(',')]
        data.append(values)
    arr = np.array(data)
    return arr.reshape(shape)


def get_preset_initial_conditions(preset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get preset initial conditions by name."""
    presets = {
        'figure8': figure_eight_initial_conditions,
        'lagrange': lagrange_initial_conditions,
        'chaotic': lambda: chaotic_initial_conditions(seed=42),
        'chaotic2': lambda: chaotic_initial_conditions(seed=123),
        'chaotic3': lambda: chaotic_initial_conditions(seed=456),
    }
    
    if preset not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    ic = presets[preset]()
    return ic['positions'], ic['velocities'], ic['masses']


def main():
    parser = argparse.ArgumentParser(
        description='Three-Body Transformer Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with figure-8 preset
  python simulate.py --preset figure8

  # Run with custom conditions from JSON
  python simulate.py --config my_conditions.json

  # Custom positions and velocities
  python simulate.py --positions "-1,0,0;0,0,0;1,0,0" --velocities "0,0.5,0;0,-0.5,0;0,0,0"

  # Longer simulation with animation
  python simulate.py --preset lagrange --steps 500 --animate

  # Launch web interface
  python simulate.py --web
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Initial Conditions')
    input_group.add_argument('--preset', type=str, default='figure8',
                            help='Preset initial conditions: figure8, lagrange, chaotic, chaotic2, chaotic3')
    input_group.add_argument('--config', type=str,
                            help='JSON file with initial conditions')
    input_group.add_argument('--positions', type=str,
                            help='Initial positions: "x1,y1,z1;x2,y2,z2;x3,y3,z3"')
    input_group.add_argument('--velocities', type=str,
                            help='Initial velocities: "vx1,vy1,vz1;vx2,vy2,vz2;vx3,vy3,vz3"')
    input_group.add_argument('--masses', type=str, default='1,1,1',
                            help='Body masses: "m1,m2,m3" (default: 1,1,1)')
    
    # Simulation options
    sim_group = parser.add_argument_group('Simulation')
    sim_group.add_argument('--steps', type=int, default=200,
                          help='Number of prediction steps (default: 200)')
    sim_group.add_argument('--no-compare', action='store_true',
                          help='Skip numerical comparison')
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output', '-o', type=str,
                             help='Save trajectory to CSV file')
    output_group.add_argument('--plot', type=str,
                             help='Save plot to file (e.g., result.png)')
    output_group.add_argument('--animate', action='store_true',
                             help='Create animation GIF')
    output_group.add_argument('--gif-path', type=str, default='simulation.gif',
                             help='Animation output path (default: simulation.gif)')
    output_group.add_argument('--no-show', action='store_true',
                             help="Don't display the plot")
    output_group.add_argument('--json', action='store_true',
                             help='Output results as JSON')
    
    # Model options
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                            help='Path to model checkpoint')
    model_group.add_argument('--scaler', type=str, default='data/scaler.pkl',
                            help='Path to scaler file')
    model_group.add_argument('--device', type=str, default='auto',
                            help='Device: cpu, cuda, or auto')
    
    # Web interface
    parser.add_argument('--web', action='store_true',
                       help='Launch Gradio web interface')
    parser.add_argument('--port', type=int, default=7860,
                       help='Web interface port (default: 7860)')
    
    args = parser.parse_args()
    
    # Launch web interface if requested
    if args.web:
        launch_web_interface(args.model, args.scaler, args.device, args.port)
        return
    
    # Get initial conditions
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        positions = np.array(config['positions'])
        velocities = np.array(config['velocities'])
        masses = np.array(config.get('masses', [1.0, 1.0, 1.0]))
    elif args.positions and args.velocities:
        positions = parse_array(args.positions, (3, 3))
        velocities = parse_array(args.velocities, (3, 3))
        masses = np.array([float(x) for x in args.masses.split(',')])
    else:
        positions, velocities, masses = get_preset_initial_conditions(args.preset)
    
    print("═" * 60)
    print("   THREE-BODY TRANSFORMER SIMULATION")
    print("═" * 60)
    print(f"\n📍 Initial Positions:")
    for i, pos in enumerate(positions):
        print(f"   Body {i+1}: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
    print(f"\n🚀 Initial Velocities:")
    for i, vel in enumerate(velocities):
        print(f"   Body {i+1}: [{vel[0]:8.4f}, {vel[1]:8.4f}, {vel[2]:8.4f}]")
    print(f"\n⚖️  Masses: {masses}")
    
    # Initialize simulator
    simulator = ThreeBodySimulator(
        model_path=args.model,
        scaler_path=args.scaler,
        device=args.device
    )
    
    # Run simulation
    result = simulator.simulate(
        positions=positions,
        velocities=velocities,
        masses=masses,
        n_prediction_steps=args.steps,
        compare_numerical=not args.no_compare
    )
    
    # Output results
    if args.json:
        output = {
            'times': result['times'].tolist(),
            'predicted': result['predicted'].tolist(),
            'seed_length': result['seed_length'],
        }
        if 'mse' in result:
            output['mse'] = float(result['mse'])
        if 'numerical' in result:
            output['numerical'] = result['numerical'].tolist()
        print(json.dumps(output, indent=2))
    
    if args.output:
        # Save to CSV
        import pandas as pd
        data = {'time': result['times']}
        for i in range(3):
            for j, coord in enumerate(['x', 'y', 'z', 'vx', 'vy', 'vz']):
                data[f'body{i+1}_{coord}'] = result['predicted'][:, i, j]
        if 'numerical' in result:
            for i in range(3):
                for j, coord in enumerate(['x', 'y', 'z', 'vx', 'vy', 'vz']):
                    data[f'body{i+1}_{coord}_numerical'] = result['numerical'][:, i, j]
        pd.DataFrame(data).to_csv(args.output, index=False)
        print(f"✓ Saved trajectory to {args.output}")
    
    # Visualization
    if not args.json:
        simulator.visualize(
            result,
            save_path=args.plot,
            show=not args.no_show,
            title=f"Three-Body Simulation ({args.preset if not args.config else 'custom'})"
        )
    
    # Animation
    if args.animate:
        simulator.create_animation(result, save_path=args.gif_path)
    
    print("\n✓ Simulation complete!")


def launch_web_interface(model_path: str, scaler_path: str, device: str, port: int):
    """Launch Gradio web interface for the simulator."""
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio not installed. Install with: pip install gradio")
        print("   Or run without --web flag for CLI mode.")
        sys.exit(1)
    
    print("🌐 Launching web interface...")
    
    # Initialize simulator
    simulator = ThreeBodySimulator(
        model_path=model_path,
        scaler_path=scaler_path,
        device=device
    )
    
    def run_simulation(preset, steps, custom_json):
        """Gradio callback for running simulation."""
        try:
            # Parse initial conditions
            if custom_json.strip():
                config = json.loads(custom_json)
                positions = np.array(config['positions'])
                velocities = np.array(config['velocities'])
                masses = np.array(config.get('masses', [1.0, 1.0, 1.0]))
            else:
                positions, velocities, masses = get_preset_initial_conditions(preset)
            
            # Run simulation
            result = simulator.simulate(
                positions=positions,
                velocities=velocities,
                masses=masses,
                n_prediction_steps=int(steps),
                compare_numerical=True,
                verbose=False
            )
            
            # Create visualization
            fig = simulator.visualize(result, show=False, title=f"Three-Body Simulation ({preset})")
            
            # Format stats
            stats = f"""
**Simulation Results:**
- Total steps: {len(result['times'])}
- Seed length: {result['seed_length']}
- Prediction MSE: {result.get('mse', 'N/A'):.6f}
            """.strip()
            
            return fig, stats
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="Three-Body Transformer") as demo:
        gr.Markdown("# 🌌 Three-Body Transformer Simulation")
        gr.Markdown("Predict three-body gravitational dynamics using a trained transformer model.")
        
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=['figure8', 'lagrange', 'chaotic', 'chaotic2', 'chaotic3'],
                    value='figure8',
                    label="Preset Initial Conditions"
                )
                steps = gr.Slider(
                    minimum=50, maximum=1000, value=200, step=50,
                    label="Prediction Steps"
                )
                custom_json = gr.Textbox(
                    label="Custom Initial Conditions (JSON, optional)",
                    placeholder='{"positions": [[x1,y1,z1], ...], "velocities": [[vx1,vy1,vz1], ...]}',
                    lines=4
                )
                run_btn = gr.Button("🚀 Run Simulation", variant="primary")
            
            with gr.Column(scale=2):
                plot_output = gr.Plot(label="Trajectory Visualization")
                stats_output = gr.Markdown(label="Statistics")
        
        run_btn.click(
            run_simulation,
            inputs=[preset, steps, custom_json],
            outputs=[plot_output, stats_output]
        )
        
        gr.Markdown("""
        ### Presets:
        - **figure8**: Famous periodic figure-8 solution (stable)
        - **lagrange**: Equilateral triangle configuration (stable)
        - **chaotic**: Random initial conditions (chaotic behavior)
        
        ### Custom JSON Format:
        ```json
        {
            "positions": [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
            "velocities": [[0, 0.5, 0], [0, -0.5, 0], [0, 0, 0]],
            "masses": [1.0, 1.0, 1.0]
        }
        ```
        """)
    
    demo.launch(server_port=port, share=False)


if __name__ == '__main__':
    main()
