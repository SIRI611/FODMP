#!/usr/bin/env python3
"""
Example script demonstrating how to use ManiSkills datasets with FODMP.

This script shows how to:
1. Load a ManiSkills dataset
2. Inspect dataset properties
3. Create data loaders
4. Iterate through batches
5. Apply filtering and normalization
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from FODMP.datasets import ManiSkillsDataset


def main():
    """Main example function."""
    print("=" * 60)
    print("MANISKILLS DATASET EXAMPLE")
    print("=" * 60)
    
    # Example dataset path (replace with your actual dataset)
    dataset_path = "/path/to/your/maniskills_dataset.h5"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please update the dataset_path variable with a valid ManiSkills dataset.")
        print("\nTo create a sample dataset, you can use the ManiSkills environment:")
        print("```python")
        print("import gym")
        print("from mani_skill.utils.wrappers import RecordEpisode")
        print("")
        print("env = gym.make('PickCube-v1', obs_mode='state')")
        print("env = RecordEpisode(env, output_dir='./recordings')")
        print("# ... collect demonstrations ...")
        print("```")
        return
    
    # 1. Load dataset
    print("\n1. Loading ManiSkills dataset...")
    dataset = ManiSkillsDataset(
        dataset_path=dataset_path,
        t_obs=2,                    # Observation horizon
        t_act=4,                    # Action horizon
        predict_past=False,         # Predict future actions
        normalize_actions=True,     # Normalize actions
        normalize_observations=True, # Normalize observations
        action_noise_std=0.01,      # Action noise for augmentation
        observation_noise_std=0.005, # Observation noise for augmentation
        position_control=True,       # Use position-based control
        movement_primitive_dim=7,    # Movement primitive dimension
        obs_mode="state"            # Observation mode
    )
    
    # 2. Inspect dataset properties
    print("\n2. Dataset properties:")
    info = dataset.get_dataset_info()
    print(f"   - Number of trajectories: {info['num_trajectories']}")
    print(f"   - Total timesteps: {info['total_timesteps']}")
    print(f"   - Average trajectory length: {info['avg_trajectory_length']:.1f}")
    print(f"   - Observation shape: {info['observation_shape']}")
    print(f"   - Action shape: {info['action_shape']}")
    print(f"   - Position control: {info['position_control']}")
    print(f"   - Movement primitive dim: {info['movement_primitive_dim']}")
    print(f"   - Observation mode: {info['obs_mode']}")
    
    # 3. Get normalization statistics
    print("\n3. Normalization statistics:")
    normalizer = dataset.get_normalizer()
    if 'actions' in normalizer:
        print(f"   - Action mean: {normalizer['actions']['mean']}")
        print(f"   - Action std: {normalizer['actions']['std']}")
    if 'observations' in normalizer:
        print(f"   - Observation mean: {normalizer['observations']['mean']}")
        print(f"   - Observation std: {normalizer['observations']['std']}")
    
    # 4. Inspect individual trajectories
    print("\n4. Individual trajectory information:")
    for i in range(min(3, len(dataset.data['trajectories']))):
        traj_info = dataset.get_trajectory_info(i)
        print(f"   Trajectory {i}:")
        print(f"     - Length: {traj_info['trajectory_length']}")
        print(f"     - Success rate: {traj_info['success_rate']:.3f}")
        print(f"     - Termination rate: {traj_info['termination_rate']:.3f}")
    
    # 5. Create data loader
    print("\n5. Creating data loader...")
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   - Batch size: 8")
    print(f"   - Number of batches: {len(dataloader)}")
    print(f"   - Total samples: {len(dataset)}")
    
    # 6. Iterate through batches
    print("\n6. Iterating through batches:")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Show only first 3 batches
            break
            
        print(f"   Batch {batch_idx}:")
        print(f"     - Observations shape: {batch['observations'].shape}")
        print(f"     - Actions shape: {batch['actions'].shape}")
        print(f"     - Observations range: [{batch['observations'].min():.3f}, {batch['observations'].max():.3f}]")
        print(f"     - Actions range: [{batch['actions'].min():.3f}, {batch['actions'].max():.3f}]")
        
        # Check for additional information
        if 'success' in batch:
            print(f"     - Success rate: {batch['success'].float().mean():.3f}")
        if 'terminated' in batch:
            print(f"     - Termination rate: {batch['terminated'].float().mean():.3f}")
    
    # 7. Filter successful trajectories
    print("\n7. Filtering successful trajectories...")
    try:
        successful_dataset = dataset.filter_successful_trajectories(min_success_rate=0.5)
        print(f"   - Original trajectories: {len(dataset.data['trajectories'])}")
        print(f"   - Successful trajectories: {len(successful_dataset.data['trajectories'])}")
        print(f"   - Success rate threshold: 0.5")
    except Exception as e:
        print(f"   - Filtering failed: {e}")
        print("   - This might be because success information is not available in the dataset")
    
    # 8. Demonstrate different configurations
    print("\n8. Different dataset configurations:")
    
    # Configuration 1: Larger horizons
    print("   Configuration 1: Larger horizons")
    dataset_large = ManiSkillsDataset(
        dataset_path=dataset_path,
        t_obs=3,
        t_act=6,
        normalize_actions=True,
        normalize_observations=True
    )
    print(f"     - Samples with t_obs=3, t_act=6: {len(dataset_large)}")
    
    # Configuration 2: Past prediction
    print("   Configuration 2: Past prediction")
    dataset_past = ManiSkillsDataset(
        dataset_path=dataset_path,
        t_obs=2,
        t_act=4,
        predict_past=True,
        normalize_actions=True,
        normalize_observations=True
    )
    print(f"     - Samples with predict_past=True: {len(dataset_past)}")
    
    # Configuration 3: No normalization
    print("   Configuration 3: No normalization")
    dataset_no_norm = ManiSkillsDataset(
        dataset_path=dataset_path,
        t_obs=2,
        t_act=4,
        normalize_actions=False,
        normalize_observations=False
    )
    print(f"     - Samples without normalization: {len(dataset_no_norm)}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # 9. Training integration example
    print("\n9. Training integration example:")
    print("   To use this dataset for training, you can:")
    print("   ```python")
    print("   # Create data loader")
    print("   train_loader = DataLoader(dataset, batch_size=32, shuffle=True)")
    print("   ")
    print("   # Training loop")
    print("   for epoch in range(num_epochs):")
    print("       for batch in train_loader:")
    print("           observations = batch['observations']")
    print("           actions = batch['actions']")
    print("           # Your training code here")
    print("   ```")
    print("   ")
    print("   Or use the MPD training script:")
    print("   ```bash")
    print("   python train_mpd.py \\")
    print("       --config-name=mpd_maniskills_position \\")
    print("       --dataset_path=/path/to/dataset.h5 \\")
    print("       --num_epochs=100 \\")
    print("       --batch_size=32")
    print("   ```")


if __name__ == "__main__":
    main()
