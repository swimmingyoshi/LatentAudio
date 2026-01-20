import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from latentaudio.explorer.walker import LatentWalker

def test_walk_displacement():
    walker = LatentWalker(latent_dim=128)
    n_steps = 10
    step_size = 1.0
    trials = 100

    print(f"Testing displacement over {n_steps} steps with step_size={step_size}")
    print("-" * 50)

    # Test Original behavior (momentum = 0, origin_pull = 0)
    displacements_orig = []
    for _ in range(trials):
        walk = walker.random_walk(n_steps=n_steps, step_size=step_size, momentum=0.0, origin_pull=0.0)
        dist = np.linalg.norm(walk[-1] - walk[0])
        displacements_orig.append(dist)
    
    avg_orig = np.mean(displacements_orig)
    print(f"Original Random Walk (Momentum 0.0): Avg Displacement = {avg_orig:.4f}")

    # Test New default behavior (momentum = 0.5, origin_pull = 0.1)
    displacements_new = []
    for _ in range(trials):
        walk = walker.random_walk(n_steps=n_steps, step_size=step_size, momentum=0.5, origin_pull=0.1)
        dist = np.linalg.norm(walk[-1] - walk[0])
        displacements_new.append(dist)
    
    avg_new = np.mean(displacements_new)
    print(f"Improved Momentum Walk (Momentum 0.5): Avg Displacement = {avg_new:.4f}")

    # Test High Momentum (momentum = 0.9)
    displacements_high = []
    for _ in range(trials):
        walk = walker.random_walk(n_steps=n_steps, step_size=step_size, momentum=0.9, origin_pull=0.1)
        dist = np.linalg.norm(walk[-1] - walk[0])
        displacements_high.append(dist)
    
    avg_high = np.mean(displacements_high)
    print(f"High Momentum Walk (Momentum 0.9):     Avg Displacement = {avg_high:.4f}")

    improvement = (avg_new / avg_orig - 1) * 100
    print("-" * 50)
    print(f"Improvement in exploration distance: {improvement:.1f}%")

if __name__ == "__main__":
    test_walk_displacement()
