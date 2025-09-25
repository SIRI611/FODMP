#!/usr/bin/env python3
"""
Training script for Movement Primitive Diffusion (MPD) on robotic manipulation environments.

This script provides a comprehensive training pipeline for movement primitive diffusion
using position-based control for robotic manipulation tasks in MetaWorld and ManiSkills environments.

Usage:
    # Train MPD on MetaWorld
    python train_mpd.py --config-name=mpd_metaworld_position --dataset_path=/path/to/metaworld_dataset
    
    # Train MPD on ManiSkills
    python train_mpd.py --config-name=mpd_maniskills_position --dataset_path=/path/to/maniskills_dataset
"""

import argparse
import hydra
import wandb
import torch
import numpy as np
import random
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from FODMP.workspace.mpd_metaworld_position_workspace import MPDMetaWorldPosition
from FODMP.workspace.mpd_maniskills_position import MPDManiSkillsPosition


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Movement Primitive Diffusion")
    
    # Core arguments
    parser.add_argument("--config-name", type=str, required=True,
                       help="Name of the configuration file")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the training dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run training on (cuda/cpu)")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for optimizer")
    
    # Evaluation parameters
    parser.add_argument("--eval_every", type=int, default=5,
                       help="Evaluate every N epochs")
    parser.add_argument("--test_every", type=int, default=20,
                       help="Test in environment every N epochs")
    parser.add_argument("--num_test_trajectories", type=int, default=10,
                       help="Number of test trajectories")
    
    # MPD specific parameters
    parser.add_argument("--movement_primitive_dim", type=int, default=7,
                       help="Dimension of movement primitives")
    parser.add_argument("--position_control", action="store_true", default=True,
                       help="Use position-based control")
    
    # Logging parameters
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Use WandB for logging")
    parser.add_argument("--wandb_project", type=str, default="fodmp-mpd",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_workspace_class(config_name: str):
    """Get the appropriate workspace class based on configuration name."""
    if "metaworld" in config_name.lower():
        return MPDMetaWorldPosition
    elif "maniskills" in config_name.lower():
        return MPDManiSkillsPosition
    else:
        raise ValueError(f"Unknown configuration: {config_name}. Expected 'metaworld' or 'maniskills' in config name.")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Override config with command line arguments
    cfg.dataset_path = args.dataset_path
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.device = args.device
    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.eval_every = args.eval_every
    cfg.test_every = args.test_every
    cfg.num_test_trajectories = args.num_test_trajectories
    cfg.movement_primitive_dim = args.movement_primitive_dim
    cfg.position_control = args.position_control
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"Initialized WandB logging: {wandb.run.url}")
    
    # Print configuration
    print("=" * 60)
    print("MOVEMENT PRIMITIVE DIFFUSION TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Configuration: {args.config_name}")
    print(f"Dataset path: {cfg.dataset_path}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"Device: {cfg.device}")
    print(f"Number of epochs: {cfg.num_epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Evaluation every: {cfg.eval_every} epochs")
    print(f"Testing every: {cfg.test_every} epochs")
    print(f"Number of test trajectories: {cfg.num_test_trajectories}")
    print(f"Movement primitive dimension: {cfg.movement_primitive_dim}")
    print(f"Position control: {cfg.position_control}")
    print(f"Random seed: {args.seed}")
    if args.resume_from:
        print(f"Resume from: {args.resume_from}")
    print("=" * 60)
    
    # Validate dataset path
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get workspace class
    workspace_class = get_workspace_class(args.config_name)
    
    # Initialize workspace
    print("Initializing MPD workspace...")
    workspace = workspace_class(
        env_config=cfg.env,
        agent_config=cfg.agent,
        dataset_config=cfg.dataset,
        t_act=cfg.get('t_act', 4),
        t_obs=cfg.get('t_obs', 2),
        predict_past=cfg.get('predict_past', False),
        num_upload_successful_videos=cfg.get('num_upload_successful_videos', 5),
        num_upload_failed_videos=cfg.get('num_upload_failed_videos', 5),
        show_images=cfg.get('show_images', False),
        movement_primitive_dim=cfg.movement_primitive_dim,
        position_control=cfg.position_control
    )
    
    # Load checkpoint if resuming
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from: {checkpoint_path}")
            workspace.load_checkpoint(str(checkpoint_path))
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    try:
        # Start training
        print("Starting MPD training...")
        workspace.train(
            dataset_path=cfg.dataset_path,
            num_epochs=cfg.num_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
            save_checkpoint_every=cfg.get('save_checkpoint_every', 10),
            eval_every=cfg.eval_every,
            test_every=cfg.test_every,
            num_test_trajectories=cfg.num_test_trajectories,
            checkpoint_dir=cfg.checkpoint_dir
        )
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving current state...")
        
        # Save interrupted state
        interrupted_checkpoint = checkpoint_dir / "interrupted_training.pt"
        workspace.agent.save_model(interrupted_checkpoint, save_optimizer=True, save_lr_scheduler=True)
        print(f"Interrupted state saved to: {interrupted_checkpoint}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Clean up
        workspace.close()
        if args.use_wandb and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
