"""
3-Body Problem Data Generator

Generates trajectories of 3 gravitationally interacting bodies using
numerical integration (Runge-Kutta 4th order).

The 3-body problem is famous for its chaotic behavior - small changes
in initial conditions lead to vastly different outcomes.
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from pathlib import Path


def gravitational_acceleration(pos, masses, G=1.0):
    """
    Calculate gravitational accelerations for all bodies.
    
    Args:
        pos: positions array of shape (n_bodies, 3)
        masses: mass of each body
        G: gravitational constant
    
    Returns:
        accelerations array of shape (n_bodies, 3)
    """
    n_bodies = len(masses)
    acc = np.zeros_like(pos)
    
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                r_vec = pos[j] - pos[i]
                r_mag = np.linalg.norm(r_vec)
                # Softening to avoid singularity
                r_mag = max(r_mag, 0.01)
                acc[i] += G * masses[j] * r_vec / (r_mag ** 3)
    
    return acc


def three_body_ode(t, state, masses, G=1.0):
    """
    ODE function for 3-body problem.
    
    State vector: [x1,y1,z1, x2,y2,z2, x3,y3,z3, vx1,vy1,vz1, vx2,vy2,vz2, vx3,vy3,vz3]
    """
    n_bodies = 3
    positions = state[:n_bodies*3].reshape(n_bodies, 3)
    velocities = state[n_bodies*3:].reshape(n_bodies, 3)
    
    accelerations = gravitational_acceleration(positions, masses, G)
    
    # Derivative: [velocities, accelerations]
    return np.concatenate([velocities.flatten(), accelerations.flatten()])


def generate_trajectory(initial_conditions, masses, t_span, n_steps=1000, G=1.0):
    """
    Generate a single 3-body trajectory.
    
    Args:
        initial_conditions: dict with 'positions' and 'velocities' 
                           each of shape (3, 3) for 3 bodies, 3D coords
        masses: array of 3 masses
        t_span: (t_start, t_end)
        n_steps: number of time steps
        G: gravitational constant
    
    Returns:
        DataFrame with columns for time, positions, and velocities
    """
    pos0 = initial_conditions['positions'].flatten()
    vel0 = initial_conditions['velocities'].flatten()
    state0 = np.concatenate([pos0, vel0])
    
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    
    sol = solve_ivp(
        three_body_ode,
        t_span,
        state0,
        args=(masses, G),
        t_eval=t_eval,
        method='DOP853',  # High-order Runge-Kutta
        rtol=1e-10,
        atol=1e-12
    )
    
    if not sol.success:
        print(f"Warning: Integration failed - {sol.message}")
    
    # Parse results
    data = {'time': sol.t}
    
    for body_idx in range(3):
        body_name = f'body{body_idx + 1}'
        # Positions
        data[f'{body_name}_x'] = sol.y[body_idx * 3]
        data[f'{body_name}_y'] = sol.y[body_idx * 3 + 1]
        data[f'{body_name}_z'] = sol.y[body_idx * 3 + 2]
        # Velocities
        data[f'{body_name}_vx'] = sol.y[9 + body_idx * 3]
        data[f'{body_name}_vy'] = sol.y[9 + body_idx * 3 + 1]
        data[f'{body_name}_vz'] = sol.y[9 + body_idx * 3 + 2]
    
    return pd.DataFrame(data)


def figure_eight_initial_conditions():
    """
    Famous figure-8 periodic solution discovered by Chenciner and Montgomery.
    This is a stable, periodic orbit.
    """
    # Positions on a line
    positions = np.array([
        [-0.97000436, 0.24308753, 0.0],
        [0.0, 0.0, 0.0],
        [0.97000436, -0.24308753, 0.0]
    ])
    
    # Velocities (body 3 has opposite of sum of 1 and 2)
    v3 = np.array([0.93240737, 0.86473146, 0.0])
    velocities = np.array([
        v3 / 2,
        -v3,
        v3 / 2
    ])
    
    return {
        'positions': positions,
        'velocities': velocities,
        'masses': np.array([1.0, 1.0, 1.0]),
        'name': 'figure_eight'
    }


def chaotic_initial_conditions(seed=None):
    """
    Generate random initial conditions that typically lead to chaotic behavior.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random positions in a cube
    positions = np.random.uniform(-2, 2, (3, 3))
    # Set z to 0 for 2D motion (easier to visualize)
    positions[:, 2] = 0
    
    # Random velocities
    velocities = np.random.uniform(-0.5, 0.5, (3, 3))
    velocities[:, 2] = 0
    
    # Random masses (similar magnitude)
    masses = np.random.uniform(0.5, 1.5, 3)
    
    return {
        'positions': positions,
        'velocities': velocities,
        'masses': masses,
        'name': f'chaotic_{seed}'
    }


def lagrange_initial_conditions():
    """
    Lagrange's equilateral triangle solution.
    Three equal masses at vertices of equilateral triangle, rotating.
    """
    # Equilateral triangle vertices
    angle = 2 * np.pi / 3
    radius = 1.0
    positions = np.array([
        [radius * np.cos(0), radius * np.sin(0), 0],
        [radius * np.cos(angle), radius * np.sin(angle), 0],
        [radius * np.cos(2 * angle), radius * np.sin(2 * angle), 0]
    ])
    
    # Circular velocities (tangent to circle)
    omega = 0.5  # Angular velocity
    velocities = np.array([
        [-radius * omega * np.sin(0), radius * omega * np.cos(0), 0],
        [-radius * omega * np.sin(angle), radius * omega * np.cos(angle), 0],
        [-radius * omega * np.sin(2 * angle), radius * omega * np.cos(2 * angle), 0]
    ])
    
    return {
        'positions': positions,
        'velocities': velocities,
        'masses': np.array([1.0, 1.0, 1.0]),
        'name': 'lagrange_triangle'
    }


def generate_dataset(output_dir='data', n_chaotic=50, n_steps=2000, t_max=50.0):
    """
    Generate a complete dataset with stable and chaotic trajectories.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_trajectories = []
    metadata = []
    
    # Generate stable trajectories
    print("Generating stable trajectories...")
    
    # Figure-8
    ic = figure_eight_initial_conditions()
    df = generate_trajectory(
        ic, ic['masses'], (0, t_max), n_steps
    )
    df['trajectory_id'] = 0
    df['trajectory_type'] = 'stable'
    df['trajectory_name'] = ic['name']
    all_trajectories.append(df)
    metadata.append({'id': 0, 'type': 'stable', 'name': ic['name']})
    print(f"  Generated: {ic['name']}")
    
    # Lagrange triangle
    ic = lagrange_initial_conditions()
    df = generate_trajectory(
        ic, ic['masses'], (0, t_max), n_steps
    )
    df['trajectory_id'] = 1
    df['trajectory_type'] = 'stable'
    df['trajectory_name'] = ic['name']
    all_trajectories.append(df)
    metadata.append({'id': 1, 'type': 'stable', 'name': ic['name']})
    print(f"  Generated: {ic['name']}")
    
    # Generate chaotic trajectories
    print(f"Generating {n_chaotic} chaotic trajectories...")
    for i in range(n_chaotic):
        ic = chaotic_initial_conditions(seed=i + 42)
        df = generate_trajectory(
            ic, ic['masses'], (0, t_max), n_steps
        )
        traj_id = i + 2
        df['trajectory_id'] = traj_id
        df['trajectory_type'] = 'chaotic'
        df['trajectory_name'] = ic['name']
        all_trajectories.append(df)
        metadata.append({'id': traj_id, 'type': 'chaotic', 'name': ic['name']})
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_chaotic} chaotic trajectories")
    
    # Combine all trajectories
    full_dataset = pd.concat(all_trajectories, ignore_index=True)
    
    # Save
    full_dataset.to_csv(output_path / 'three_body_trajectories.csv', index=False)
    pd.DataFrame(metadata).to_csv(output_path / 'trajectory_metadata.csv', index=False)
    
    print(f"\nDataset saved to {output_path}/")
    print(f"Total trajectories: {len(metadata)}")
    print(f"Total timesteps: {len(full_dataset)}")
    
    return full_dataset, metadata


if __name__ == '__main__':
    generate_dataset()
