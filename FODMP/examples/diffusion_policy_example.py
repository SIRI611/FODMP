#!/usr/bin/env python3
"""
Example script demonstrating how to use the Diffusion Policy implementation.

This script shows how to:
1. Train a diffusion policy on MetaWorld tasks
2. Evaluate the trained policy
3. Save and load checkpoints
4. Use the policy for inference

Usage:
    python examples/diffusion_policy_example.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from FODMP.workspace.dp_cnn_metaworld_workspace import DPCNNMetaWorldWorkspace
from FODMP.datasets.metaworld_dataset import MetaWorldDataset
from train_dp import train_with_config


def create_sample_dataset(
    num_trajectories: int = 10,
    trajectory_length: int = 100,
    obs_dim: int = 20,
    action_dim: int = 4,
    save_path: str = "./sample_dataset.npz"
) -> str:
    """
    Create a sample dataset for demonstration purposes.
    
    Args:
        num_trajectories: Number of trajectories to generate
        trajectory_length: Length of each trajectory
        obs_dim: Observation dimension
        action_dim: Action dimension
        save_path: Path to save the dataset
        
    Returns:
        Path to the saved dataset
    """
    print(f"Creating sample dataset with {num_trajectories} trajectories...")
    
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    
    for traj_idx in range(num_trajectories):
        # Generate random trajectory data
        traj_obs = np.random.randn(trajectory_length, obs_dim)
        traj_actions = np.random.randn(trajectory_length, action_dim)
        traj_rewards = np.random.randn(trajectory_length)
        traj_dones = np.zeros(trajectory_length, dtype=bool)
        traj_dones[-1] = True  # Last timestep is terminal
        traj_infos = [{"success": np.random.random() > 0.5} for _ in range(trajectory_length)]
        
        observations.append(traj_obs)
        actions.append(traj_actions)
        rewards.append(traj_rewards)
        dones.append(traj_dones)
        infos.append(traj_infos)
    
    # Save dataset
    np.savez(
        save_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        infos=infos,
        metadata={
            "num_trajectories": num_trajectories,
            "trajectory_length": trajectory_length,
            "obs_dim": obs_dim,
            "action_dim": action_dim
        }
    )
    
    print(f"Sample dataset saved to: {save_path}")
    return save_path


def demonstrate_dataset_loading(dataset_path: str):
    """Demonstrate dataset loading and inspection."""
    print("\n" + "="*50)
    print("DATASET LOADING DEMONSTRATION")
    print("="*50)
    
    # Load dataset
    dataset = MetaWorldDataset(
        dataset_path=dataset_path,
        t_obs=2,
        t_act=4,
        predict_past=False,
        normalize_actions=True,
        normalize_observations=True
    )
    
    # Print dataset info
    info = dataset.get_dataset_info()
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test data loading
    print(f"\nDataset size: {len(dataset)}")
    
    # Load a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Observations shape: {sample['observations'].shape}")
        print(f"  Actions shape: {sample['actions'].shape}")
        print(f"  Observation range: [{sample['observations'].min():.3f}, {sample['observations'].max():.3f}]")
        print(f"  Action range: [{sample['actions'].min():.3f}, {sample['actions'].max():.3f}]")


def demonstrate_training(dataset_path: str, config_path: str = None):
    """Demonstrate training a diffusion policy."""
    print("\n" + "="*50)
    print("TRAINING DEMONSTRATION")
    print("="*50)
    
    # Create a simple config if none provided
    if config_path is None:
        config_path = create_sample_config()
    
    # Train the model
    print("Starting training...")
    workspace = train_with_config(
        config_path=config_path,
        dataset_path=dataset_path,
        checkpoint_dir="./example_checkpoints",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_wandb=False,
        wandb_project="example-dp-training",
        wandb_run_name="example-run"
    )
    
    print("Training completed!")
    return workspace


def demonstrate_inference(workspace: DPCNNMetaWorldWorkspace, num_samples: int = 5):
    """Demonstrate inference with the trained model."""
    print("\n" + "="*50)
    print("INFERENCE DEMONSTRATION")
    print("="*50)
    
    if workspace.agent is None:
        print("No agent available for inference.")
        return
    
    # Set agent to evaluation mode
    workspace.agent.model.eval()
    workspace.agent.encoder.eval()
    
    # Create sample observations
    device = workspace.agent.device
    batch_size = num_samples
    t_obs = workspace.t_obs
    obs_dim = 20  # Should match dataset
    
    # Generate random observations
    observations = torch.randn(batch_size, t_obs, obs_dim).to(device)
    extra_inputs = {}
    
    print(f"Running inference on {num_samples} samples...")
    print(f"Input observation shape: {observations.shape}")
    
    with torch.no_grad():
        # Run inference
        actions = workspace.agent.predict(
            observation={"observations": observations},
            extra_inputs=extra_inputs
        )
    
    print(f"Output action shape: {actions.shape}")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Plot some results
    if num_samples >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot observations
        for i in range(min(2, num_samples)):
            axes[0, i].plot(observations[i].cpu().numpy())
            axes[0, i].set_title(f'Observations - Sample {i}')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Value')
        
        # Plot actions
        for i in range(min(2, num_samples)):
            axes[1, i].plot(actions[i].cpu().numpy())
            axes[1, i].set_title(f'Actions - Sample {i}')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('./inference_results.png')
        print("Inference results saved to: inference_results.png")


def demonstrate_evaluation(workspace: DPCNNMetaWorldWorkspace):
    """Demonstrate evaluation of the trained model."""
    print("\n" + "="*50)
    print("EVALUATION DEMONSTRATION")
    print("="*50)
    
    if workspace.agent is None:
        print("No agent available for evaluation.")
        return
    
    # Test the agent in the environment
    print("Testing agent in environment...")
    results = workspace.test_agent(workspace.agent, num_trajectories=5)
    
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")


def create_sample_config() -> str:
    """Create a sample configuration file."""
    config_content = """
# Sample configuration for diffusion policy training
env:
  _target_: metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
  env_name: "reach-v2"
  time_limit: 500

dataset:
  _target_: FODMP.datasets.metaworld_dataset.MetaWorldDataset
  dataset_path: null
  t_obs: 2
  t_act: 4
  predict_past: false
  normalize_actions: true
  normalize_observations: true

agent:
  _target_: movement_primitive_diffusion.agents.discrete_time_diffusion_agent.DiscreteTimeDiffusionAgent
  device: null
  t_obs: 2
  predict_past: false
  num_inference_steps: 10
  use_ema: true
  ema_config:
    decay: 0.9999
    min_decay: 0.0
    update_after_step: 0
    use_ema_warmup: true
    inv_gamma: 1.0
    power: 0.75
  optimizer_config:
    _target_: torch.optim.AdamW
    lr: 1.0e-4
    betas: [0.95, 0.999]
    eps: 1.0e-8
    weight_decay: 1.0e-6
  lr_scheduler_config:
    _target_: movement_primitive_diffusion.utils.lr_scheduler.get_scheduler
    name: cosine
    num_warmup_steps: 500
    num_training_steps: null
    last_epoch: -1

# Training parameters
t_obs: 2
t_act: 4
predict_past: false
num_epochs: 5  # Small number for example
batch_size: 8
learning_rate: 1.0e-4
save_checkpoint_every: 2
eval_every: 1
test_every: 2
num_test_trajectories: 3

# Video logging
num_upload_successful_videos: 2
num_upload_failed_videos: 2
show_images: false

# Device configuration
device: "cuda"

# Checkpoint directory
checkpoint_dir: "./example_checkpoints"
"""
    
    config_path = "./example_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Sample configuration saved to: {config_path}")
    return config_path


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Diffusion Policy Example")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and only demonstrate inference")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to dataset (will create sample if not provided)")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to configuration file")
    args = parser.parse_args()
    
    print("="*60)
    print("DIFFUSION POLICY IMPLEMENTATION DEMONSTRATION")
    print("="*60)
    
    # Create sample dataset if not provided
    if args.dataset_path is None:
        dataset_path = create_sample_dataset()
    else:
        dataset_path = args.dataset_path
    
    # Demonstrate dataset loading
    demonstrate_dataset_loading(dataset_path)
    
    workspace = None
    if not args.skip_training:
        # Demonstrate training
        workspace = demonstrate_training(dataset_path, args.config_path)
    else:
        print("\nSkipping training demonstration.")
        print("To demonstrate inference, you would need a trained model.")
        return
    
    if workspace is not None:
        # Demonstrate inference
        demonstrate_inference(workspace)
        
        # Demonstrate evaluation
        demonstrate_evaluation(workspace)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED")
    print("="*60)
    print("Files created:")
    print("  - Sample dataset: ./sample_dataset.npz")
    print("  - Sample config: ./example_config.yaml")
    print("  - Checkpoints: ./example_checkpoints/")
    print("  - Inference results: ./inference_results.png")


if __name__ == "__main__":
    main()
